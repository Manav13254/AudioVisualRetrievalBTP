import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import argparse
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity

try:
    import clip
except ImportError:
    print("Error: CLIP not found.")
    sys.exit()

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    # Pointing to the new finetuned weights
    parser.add_argument("--model_path", type=str, default="best_finetune_clip_cnn14.pth")
    parser.add_argument("--perplexity", type=int, default=30)
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error loading Cnn14.")
    sys.exit()

# ==========================================
# 1. MODEL DEFINITION (Must Match Training)
# ==========================================
class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        self.clip_model = self.clip_model.float()
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # We don't need grad for inference, so just forward
        features = self.clip_model.encode_image(x)
        features = features.float()
        return F.normalize(self.head(features), p=2, dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output_dict = self.base_model(x)
        return F.normalize(self.head(output_dict['embedding']), p=2, dim=1)

class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = VisionEncoder()
        self.audio_model = AudioEncoder()

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def extract_features(model, v_dir, a_dir):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    
    v_feats, a_feats = [], []
    v_labels, a_labels = [], []
    
    print("--- Extracting Features ---")
    with torch.no_grad():
        for i, cls in enumerate(tqdm(classes)):
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                try:
                    img = Image.open(f).convert('RGB')
                    img = clip_transform(img).unsqueeze(0).to(DEVICE)
                    v_feats.append(model.vision_model(img).cpu().numpy())
                    v_labels.append(i)
                except: continue
                
            for f in glob.glob(os.path.join(a_dir, cls, "*.wav")):
                try:
                    wav, sr = torchaudio.load(f)
                    if sr!=32000: wav=T.Resample(sr,32000)(wav)
                    if wav.shape[1]<320000: wav=F.pad(wav,(0,320000-wav.shape[1]))
                    else: wav=wav[:,:320000]
                    if wav.shape[0]>1: wav=torch.mean(wav,dim=0,keepdim=True)
                    wav = wav.to(DEVICE)
                    a_feats.append(model.audio_model(wav).cpu().numpy())
                    a_labels.append(i)
                except: continue
                
    return np.vstack(v_feats), np.vstack(a_feats), np.array(v_labels), np.array(a_labels), classes

# ==========================================
# 3. PLOTS
# ==========================================
def plot_joint_tsne(v_feats, a_feats, v_labels, a_labels, class_names):
    print("Generating Joint t-SNE...")
    combined_feats = np.vstack([v_feats, a_feats])
    combined_labels = np.concatenate([v_labels, a_labels])
    domains = np.concatenate([np.zeros(len(v_labels)), np.ones(len(a_labels))])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=args.perplexity)
    reduced = tsne.fit_transform(combined_feats)
    
    plt.figure(figsize=(15, 12))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    
    for i, cls in enumerate(class_names):
        mask = (combined_labels == i) & (domains == 0)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors[i]], marker='^', alpha=0.6, label=f"{cls} (Img)" if i<5 else "")
        mask = (combined_labels == i) & (domains == 1)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], c=[colors[i]], marker='o', edgecolors='k', alpha=0.6, label=f"{cls} (Aud)" if i<5 else "")
        
    plt.title("Joint t-SNE (Fine-Tuned CLIP+Cnn14)")
    plt.tight_layout()
    plt.savefig("finetune_clip_tsne_joint.png")
    return reduced 

def plot_class_subplots(reduced_all, v_labels, a_labels, class_names):
    print("Generating Per-Class Subplots...")
    split_idx = len(v_labels)
    v_2d = reduced_all[:split_idx]
    a_2d = reduced_all[split_idx:]
    
    num_classes = len(class_names)
    cols = 4
    rows = math.ceil(num_classes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    x_min, x_max = reduced_all[:, 0].min(), reduced_all[:, 0].max()
    y_min, y_max = reduced_all[:, 1].min(), reduced_all[:, 1].max()
    margin = 5
    
    for i, cls in enumerate(class_names):
        ax = axes[i]
        v_mask = (v_labels == i)
        a_mask = (a_labels == i)
        ax.scatter(v_2d[v_mask, 0], v_2d[v_mask, 1], c='royalblue', marker='^', s=80, alpha=0.7, label='Vision')
        ax.scatter(a_2d[a_mask, 0], a_2d[a_mask, 1], c='darkorange', marker='o', s=80, alpha=0.7, label='Audio')
        ax.set_title(f"{cls}", fontsize=14, fontweight='bold')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0: ax.legend(loc='upper right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig("finetune_clip_tsne_subplots.png", dpi=300)

def plot_single_confusion(query_feats, gallery_feats, query_labels, gallery_labels, class_names, title, filename, ylabel, xlabel, cmap='Purples'):
    print(f"Generating Confusion Matrix: {title}...")
    sim = cosine_similarity(query_feats, gallery_feats)
    y_true, y_pred = [], []
    for i in range(len(query_labels)):
        top_idx = np.argmax(sim[i])
        y_true.append(query_labels[i])
        y_pred.append(gallery_labels[top_idx])
        
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 
    cm_norm = cm.astype('float') / row_sums
    
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap=cmap, ax=ax, xticks_rotation='vertical', values_format='.2f')
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename)

# ==========================================
# 4. MAIN
# ==========================================
def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")
    
    model = CrossModalModel().to(DEVICE)
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    else:
        print("Warning: Model weights not found!")

    v_feats, a_feats, v_lbls, a_lbls, classes = extract_features(model, VAL_V, VAL_A)
    
    print("\nCalculating Metrics...")
    sim_matrix = cosine_similarity(v_feats, a_feats)
    
    # I2A
    r1_i, r5_i, r10_i = 0, 0, 0
    for i in range(len(v_lbls)):
        indices = np.argsort(sim_matrix[i])[::-1]
        top = a_lbls[indices[:10]]
        if v_lbls[i] in top[:1]: r1_i += 1
        if v_lbls[i] in top[:5]: r5_i += 1
        if v_lbls[i] in top[:10]: r10_i += 1
    
    # A2I
    r1_a, r5_a, r10_a = 0, 0, 0
    sim_T = sim_matrix.T
    for i in range(len(a_lbls)):
        indices = np.argsort(sim_T[i])[::-1]
        top = v_lbls[indices[:10]]
        if a_lbls[i] in top[:1]: r1_a += 1
        if a_lbls[i] in top[:5]: r5_a += 1
        if a_lbls[i] in top[:10]: r10_a += 1

    n_v, n_a = len(v_lbls), len(a_lbls)
    r1_i, r5_i, r10_i = (r1_i/n_v)*100, (r5_i/n_v)*100, (r10_i/n_v)*100
    r1_a, r5_a, r10_a = (r1_a/n_a)*100, (r5_a/n_a)*100, (r10_a/n_a)*100
    
    avg_r1 = (r1_i + r1_a) / 2
    avg_r5 = (r5_i + r5_a) / 2
    avg_r10 = (r10_i + r10_a) / 2

    print("\n" + "="*40)
    print("   FINAL RETRIEVAL RESULTS (FINE-TUNED CLIP+CNN14)")
    print("="*40)
    print(f"Image -> Audio  | R@1: {r1_i:.2f}% | R@5: {r5_i:.2f}% | R@10: {r10_i:.2f}%")
    print(f"Audio -> Image  | R@1: {r1_a:.2f}% | R@5: {r5_a:.2f}% | R@10: {r10_a:.2f}%")
    print("-" * 40)
    print(f"AVERAGE         | R@1: {avg_r1:.2f}% | R@5: {avg_r5:.2f}% | R@10: {avg_r10:.2f}%")
    print("="*40 + "\n")
    
    reduced_coords = plot_joint_tsne(v_feats, a_feats, v_lbls, a_lbls, classes)
    plot_class_subplots(reduced_coords, v_lbls, a_lbls, classes)
    
    plot_single_confusion(
        query_feats=v_feats, gallery_feats=a_feats, 
        query_labels=v_lbls, gallery_labels=a_lbls, 
        class_names=classes, 
        title="Image-to-Audio (Fine-Tuned CLIP)", 
        filename="finetune_clip_confusion_img2aud.png",
        ylabel="True Image Class",
        xlabel="Retrieved Audio Class",
        cmap='Blues'
    )
    
    plot_single_confusion(
        query_feats=a_feats, gallery_feats=v_feats, 
        query_labels=a_lbls, gallery_labels=v_lbls, 
        class_names=classes, 
        title="Audio-to-Image (Fine-Tuned CLIP)", 
        filename="finetune_clip_confusion_aud2img.png",
        ylabel="True Audio Class",
        xlabel="Retrieved Image Class",
        cmap='Oranges'
    )

if __name__ == "__main__":
    main()