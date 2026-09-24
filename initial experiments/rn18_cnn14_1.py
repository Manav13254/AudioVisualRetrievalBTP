import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import glob
import argparse
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ==========================================
# 0. SETUP & PATHS (Adjusted to match your working script)
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    # Defaulting to the structure that worked in your img2img script
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_cross_modal.pth")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Audioset Repo Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.join(CURRENT_DIR, "audioset_tagging_cnn")

# Adding paths for the AudioSet backbone
sys.path.append(REPO_PATH)
sys.path.append(os.path.join(REPO_PATH, 'pytorch'))

try:
    from pytorch.models import Cnn14
except ImportError:
    print(f"Error loading Cnn14. Ensure '{REPO_PATH}' contains the model definitions.")
    sys.exit()

# ==========================================
# 1. MODELS (Contrastive Learning Architectures)
# ==========================================
# These classes implement Siamese-style encoders
class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Cnn14 is a standard architecture for AudioSet tagging
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, 
                                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
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

class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = VisionEncoder()
        self.audio_model = AudioEncoder()

# ==========================================
# 2. DATASET (Matching your working structure)
# ==========================================
class BidirectionalDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir
        self.audio_dir = audio_dir
        self.transform = transform
        
        # Intersecting classes to ensure alignment
        v_classes = set(os.listdir(vision_dir))
        a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        
        # Glob patterns to find files exactly like your working script
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}

    def __len__(self): 
        return sum(len(imgs) for imgs in self.class_to_imgs.values())

    def _process_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: 
            return torch.zeros(1, 320000)

    def __getitem__(self, index):
        # Sampling logic for Triplet Loss
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        img_path = random.choice(self.class_to_imgs[target_class])
        aud_pos_path = random.choice(self.class_to_auds[target_class])
        img_neg_path = random.choice(self.class_to_imgs[neg_class])
        aud_neg_path = random.choice(self.class_to_auds[neg_class])
        
        img_anchor = Image.open(img_path).convert('RGB')
        img_neg = Image.open(img_neg_path).convert('RGB')
        
        if self.transform:
            img_anchor = self.transform(img_anchor)
            img_neg = self.transform(img_neg)
            
        aud_pos = self._process_audio(aud_pos_path)
        aud_neg = self._process_audio(aud_neg_path)
        
        return img_anchor, img_neg, aud_pos, aud_neg

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. METRICS (Recall@K)
# ==========================================
def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10]):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    v_feats, a_feats, labels = [], [], []
    
    # Feature Extraction (Standard Evaluation Protocol)
    with torch.no_grad():
        for i, cls in enumerate(classes):
            v_files = glob.glob(os.path.join(v_dir, cls, "*.*"))
            for f in v_files:
                try:
                    img = Image.open(f).convert('RGB')
                    img = vision_transform(img).unsqueeze(0).to(DEVICE)
                    v_feats.append(model.vision_model(img).cpu().numpy())
                    labels.append(i)
                except: continue
    
    a_feats_dict = {i: [] for i in range(len(classes))}
    for i, cls in enumerate(classes):
        a_files = glob.glob(os.path.join(a_dir, cls, "*.wav"))
        for f in a_files:
            try:
                wav, sr = torchaudio.load(f)
                if sr!=32000: wav=T.Resample(sr,32000)(wav)
                if wav.shape[1]<320000: wav=F.pad(wav,(0,320000-wav.shape[1]))
                else: wav=wav[:,:320000]
                if wav.shape[0]>1: wav=torch.mean(wav,dim=0,keepdim=True)
                wav = wav.to(DEVICE)
                with torch.no_grad():
                    a_feats_dict[i].append(model.audio_model(wav).cpu().numpy())
            except: continue

    final_a_feats, final_a_labels = [], []
    for cls_idx, feats in a_feats_dict.items():
        for f in feats:
            final_a_feats.append(f)
            final_a_labels.append(cls_idx)
            
    v_feats = np.vstack(v_feats)
    final_a_feats = np.vstack(final_a_feats)
    v_labels, a_labels = np.array(labels), np.array(final_a_labels)
    
    sim_matrix = cosine_similarity(v_feats, final_a_feats)
    
    # Image to Audio (I2A) and Audio to Image (A2I) Retrieval metrics
    def calculate_r_at_k(matrix, q_labels, g_labels):
        scores = {k: 0 for k in k_values}
        for idx in range(len(q_labels)):
            sorted_indices = np.argsort(matrix[idx])[::-1]
            for k in k_values:
                if q_labels[idx] in g_labels[sorted_indices[:k]]: scores[k] += 1
        return {k: (v / len(q_labels)) * 100 for k, v in scores.items()}

    i2a = calculate_r_at_k(sim_matrix, v_labels, a_labels)
    a2i = calculate_r_at_k(sim_matrix.T, a_labels, v_labels)
    
    return i2a, a2i

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    # Folder names set to match your successful script's 'vision' and 'sound' directories
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = CrossModalModel().to(DEVICE)
    
    # Security-focused checkpoint loading (weights_only=True)
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"WARNING: Weights not found at {args.audio_weights}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    # Triplet Loss is the standard for alignment tasks
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    best_avg_r5 = 0.0
    print(f"--- Starting Audio-Visual Alignment ---")
    
    for epoch in range(20):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
        
        for img_anc, img_neg, aud_pos, aud_neg in pbar:
            img_anc, img_neg = img_anc.to(DEVICE), img_neg.to(DEVICE)
            aud_pos, aud_neg = aud_pos.to(DEVICE), aud_neg.to(DEVICE)
            
            optimizer.zero_grad()
            v_a, v_n = model.vision_model(img_anc), model.vision_model(img_neg)
            a_p, a_n = model.audio_model(aud_pos), model.audio_model(aud_neg)
            
            # Bidirectional Triplet Loss for cross-modal alignment
            loss = criterion(v_a, a_p, a_n) + criterion(a_p, v_a, v_n)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print("Validating Cross-Modal Retrieval...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  I->A: R@1 {i2a[1]:.1f}% | R@5 {i2a[5]:.1f}%")
        print(f"  A->I: R@1 {a2i[1]:.1f}% | R@5 {a2i[5]:.1f}%")
        
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            torch.save(model.state_dict(), args.save_path)
            print("  🔥 Best Model Alignment Saved!")

if __name__ == "__main__":
    main()