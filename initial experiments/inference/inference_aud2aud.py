import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import sys

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--model_path", type=str, default="best_aud2aud.pth")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'utils'))
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error loading Cnn14.")
    sys.exit()

# ==========================================
# 1. MODEL
# ==========================================
class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output_dict = self.base_model(x)
        out = self.head(output_dict['embedding'])
        return F.normalize(out, p=2, dim=1)

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def load_audio(path):
    try:
        wav, sr = torchaudio.load(path)
        if sr != 32000: wav = T.Resample(sr, 32000)(wav)
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
        else: wav = wav[:, :320000]
        return wav
    except: return torch.zeros(1, 320000)

def extract_features(model, data_dir):
    model.eval()
    embeddings = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    print("--- Extracting Audio Features ---")
    with torch.no_grad():
        for label_idx, class_name in enumerate(tqdm(class_names)):
            files = glob.glob(os.path.join(data_dir, class_name, "*.wav"))
            for path in files:
                wav = load_audio(path).unsqueeze(0).to(DEVICE)
                emb = model(wav).cpu().numpy()
                embeddings.append(emb)
                labels.append(label_idx)
                    
    return np.vstack(embeddings), np.array(labels), class_names

# ==========================================
# 3. VISUALIZATION
# ==========================================
def plot_tsne(embeddings, labels, class_names):
    print("Generating t-SNE Plot...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                    c=[colors[i]], label=class_names[lbl], s=20, alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE of Audio Embeddings (Frozen Cnn14)")
    plt.tight_layout()
    plt.savefig("aud2aud_tsne.png")
    print("Saved: aud2aud_tsne.png")

def plot_confusion_matrix(y_true, y_pred, class_names):
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap='Oranges', ax=ax, xticks_rotation='vertical', values_format='.2f')
    
    plt.title("Audio Retrieval Confusion Matrix (Top-1 Match - Normalized)")
    plt.tight_layout()
    plt.savefig("aud2aud_confusion_matrix.png")
    print("Saved: aud2aud_confusion_matrix.png")

# ==========================================
# 4. MAIN
# ==========================================
def main():
    test_dir = os.path.join(args.data_root, "test", "sound")
    
    model = AudioEncoder().to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"Loaded weights from {args.model_path}")
    else:
        print("Warning: No weights found. Using random init.")

    feats, labels, class_names = extract_features(model, test_dir)
    sim_matrix = feats @ feats.T 
    np.fill_diagonal(sim_matrix, -1)
    
    y_true, y_pred = [], []
    r1, r5, r10 = 0, 0, 0
    n = len(labels)
    
    for i in range(n):
        scores = sim_matrix[i]
        sorted_indices = np.argsort(scores)[::-1]
        
        true_cls = labels[i]
        top_10 = labels[sorted_indices[:10]]
        
        if true_cls in top_10[:1]: r1 += 1
        if true_cls in top_10[:5]: r5 += 1
        if true_cls in top_10[:10]: r10 += 1
        
        y_true.append(true_cls)
        y_pred.append(labels[sorted_indices[0]])
        
    print("\n=== Audio-to-Audio Retrieval Results ===")
    print(f"R@1:  {r1/n*100:.2f}%")
    print(f"R@5:  {r5/n*100:.2f}%")
    print(f"R@10: {r10/n*100:.2f}%")
    
    plot_tsne(feats, labels, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()