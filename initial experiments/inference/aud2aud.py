import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import glob
import argparse
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_aud2aud.pth")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")

if not os.path.exists(REPO_PATH):
    print(f"CRITICAL ERROR: '{REPO_PATH}' folder missing.")
    sys.exit()

sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'utils'))
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print("Error loading Cnn14. Check paths.")
    sys.exit()

# ==========================================
# 1. MODEL (Frozen Cnn14)
# ==========================================
class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # Freeze Backbone
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Trainable Head
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
# 2. DATASETS
# ==========================================
class TripletAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_auds = {c: glob.glob(os.path.join(root_dir, c, "*.wav")) for c in self.classes}
        self.classes = [c for c in self.classes if len(self.class_to_auds[c]) > 1]

    def __len__(self):
        return sum(len(auds) for auds in self.class_to_auds.values())

    def __getitem__(self, index):
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        anc_path, pos_path = random.sample(self.class_to_auds[target_class], 2)
        neg_path = random.choice(self.class_to_auds[neg_class])
        
        return self._load(anc_path), self._load(pos_path), self._load(neg_path)
    
    def _load(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: return torch.zeros(1, 320000)

class ValidationGalleryDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for i, cls in enumerate(self.classes):
            files = glob.glob(os.path.join(root_dir, cls, "*.wav"))
            for f in files:
                self.paths.append(f)
                self.labels.append(i)

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        return self._load(self.paths[index]), self.labels[index]

    def _load(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: return torch.zeros(1, 320000)

# ==========================================
# 3. METRICS
# ==========================================
def calculate_metrics(model, dataloader, device):
    model.eval()
    feats = []
    labels = []
    
    with torch.no_grad():
        for auds, lbls in dataloader:
            auds = auds.to(device)
            emb = model(auds)
            feats.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
            
    feats = np.vstack(feats)
    labels = np.array(labels)
    
    sim_matrix = cosine_similarity(feats)
    np.fill_diagonal(sim_matrix, -1)
    
    r1, r5, r10 = 0, 0, 0
    n = len(labels)
    
    for i in range(n):
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        top_k = labels[sorted_indices[:10]]
        true_lbl = labels[i]
        
        if true_lbl in top_k[:1]: r1 += 1
        if true_lbl in top_k[:5]: r5 += 1
        if true_lbl in top_k[:10]: r10 += 1
        
    return (r1/n)*100, (r5/n)*100, (r10/n)*100

def validate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            loss = criterion(model(anc), model(pos), model(neg))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    if not os.path.exists(args.audio_weights):
        print(f"Error: Weights '{args.audio_weights}' not found.")
        return

    train_dir = os.path.join(args.data_root, "train", "sound")
    val_dir = os.path.join(args.data_root, "test", "sound")
    
    train_ds = TripletAudioDataset(train_dir)
    val_triplet_ds = TripletAudioDataset(val_dir)
    val_gallery_ds = ValidationGalleryDataset(val_dir)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loss_loader = DataLoader(val_triplet_ds, batch_size=32, shuffle=False, num_workers=4)
    val_metric_loader = DataLoader(val_gallery_ds, batch_size=32, shuffle=False, num_workers=4)
    
    model = AudioEncoder().to(DEVICE)
    
    # Load Cnn14 Weights
    ckpt = torch.load(args.audio_weights, map_location=DEVICE)
    if 'model' in ckpt: ckpt = ckpt['model']
    model.base_model.load_state_dict(ckpt, strict=False)
    print("Pretrained Cnn14 Weights Loaded.")
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    best_r5 = 0.0
    print(f"--- Training Audio-to-Audio Model ---")
    
    for epoch in range(20):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20 [Train]")
        
        for anc, pos, neg in pbar:
            anc, pos, neg = anc.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            
            optimizer.zero_grad()
            loss = criterion(model(anc), model(pos), model(neg))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate_loss(model, val_loss_loader, criterion, DEVICE)
        
        print(f"Calculating Metrics...")
        r1, r5, r10 = calculate_metrics(model, val_metric_loader, DEVICE)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  R@1: {r1:.2f}% | R@5: {r5:.2f}% | R@10: {r10:.2f}%")
        
        if r5 > best_r5:
            best_r5 = r5
            torch.save(model.state_dict(), args.save_path)
            print(f"  🔥 New Best Model (R@5: {best_r5:.2f}%) Saved!")
        print("-" * 50)

if __name__ == "__main__":
    main()