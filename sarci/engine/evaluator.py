"""
Evaluation: build the full image x audio similarity matrix, then compute
R@1/5/10, median rank, and mean rank.

retrieval_metrics_paired() is a faithful port of acc_i2t2 / acc_t2i2 from
the official repo's utils.py -- it assumes the paper's layout (audio_per_image
captions per image, contiguous in the gallery, e.g. audio rows
[5*i : 5*i+5] all belong to image i). Use it with PairedAudioVisualDataset.

retrieval_metrics_by_label() is the class-based equivalent used by this
project's earlier scripts (ADVANCE dataset, no explicit pairing) -- a match
is any gallery item sharing the query's class label, not a fixed index
range. Use it with ClassPairedDataset / ValVisionDataset / ValAudioDataset.

shard_similarity() extracts vision/audio features once (backbones run once
each, matching the efficiency of the old compute_metrics in
"initial experiments/rn18_cnn14_iclm.py") and then shards the cheap
cross_learning + cosine_similarity call over both axes -- functionally
equivalent to the official shard_dis, but without re-running the frozen
backbones per shard.
"""

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def shard_similarity(model, vision_loader, audio_loader, device, shard_size=64):
    """vision_loader / audio_loader yield (tensor, label_or_index) batches.

    Returns (sim_matrix [N_images, M_audio], vision_labels, audio_labels) as numpy arrays.
    """
    model.eval()

    v_feats, v_labels = [], []
    for x, labels in tqdm(vision_loader, desc="vision features", leave=False):
        v_feats.append(model.encode_vision(x.to(device)).cpu())
        v_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)

    a_feats, a_labels = [], []
    for x, labels in tqdm(audio_loader, desc="audio features", leave=False):
        a_feats.append(model.encode_audio(x.to(device)).cpu())
        a_labels.extend(labels.numpy() if torch.is_tensor(labels) else labels)

    v_feats = torch.cat(v_feats, dim=0)
    a_feats = torch.cat(a_feats, dim=0)
    n, m = v_feats.size(0), a_feats.size(0)
    sim = np.zeros((n, m), dtype=np.float32)

    for i in tqdm(range(0, n, shard_size), desc="cross-interaction", leave=False):
        v_chunk = v_feats[i:i + shard_size].to(device)
        for j in range(0, m, shard_size):
            a_chunk = a_feats[j:j + shard_size].to(device)
            block = model.similarity_from_features(v_chunk, a_chunk).cpu().numpy()
            sim[i:i + v_chunk.size(0), j:j + a_chunk.size(0)] = block

    return sim, np.array(v_labels), np.array(a_labels)


def retrieval_metrics_paired(sim: np.ndarray, audio_per_image: int = 5):
    """Faithful port of acc_i2t2 (image->audio) and acc_t2i2 (audio->image).

    Assumes audio rows [audio_per_image*i : audio_per_image*i+audio_per_image]
    are the ground-truth captions for image i.
    """

    def _rank_stats(ranks):
        r1 = 100.0 * np.mean(ranks < 1)
        r5 = 100.0 * np.mean(ranks < 5)
        r10 = 100.0 * np.mean(ranks < 10)
        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        return {"R@1": r1, "R@5": r5, "R@10": r10, "medR": medr, "meanR": meanr}

    n_images = sim.shape[0]
    i2a_ranks = np.zeros(n_images)
    for idx in range(n_images):
        inds = np.argsort(sim[idx])[::-1]
        rank = min(
            np.where(inds == i)[0][0]
            for i in range(audio_per_image * idx, audio_per_image * idx + audio_per_image)
        )
        i2a_ranks[idx] = rank
    i2a = _rank_stats(i2a_ranks)

    a2i_sim = sim.T
    a2i_ranks = np.zeros(audio_per_image * n_images)
    for idx in range(n_images):
        for k in range(audio_per_image):
            inds = np.argsort(a2i_sim[audio_per_image * idx + k])[::-1]
            a2i_ranks[audio_per_image * idx + k] = np.where(inds == idx)[0][0]
    a2i = _rank_stats(a2i_ranks)

    mr = (i2a["R@1"] + i2a["R@5"] + i2a["R@10"] + a2i["R@1"] + a2i["R@5"] + a2i["R@10"]) / 6.0
    return i2a, a2i, mr


def retrieval_metrics_by_label(sim: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray,
                                k_values=(1, 5, 10)):
    """Class-based recall: a hit is any gallery item sharing the query's label."""
    hits = {k: 0 for k in k_values}
    top_k_max = max(k_values)
    for idx in range(len(query_labels)):
        inds = np.argsort(sim[idx])[::-1][:top_k_max]
        top_labels = gallery_labels[inds]
        for k in k_values:
            if query_labels[idx] in top_labels[:k]:
                hits[k] += 1
    return {f"R@{k}": 100.0 * v / len(query_labels) for k, v in hits.items()}
