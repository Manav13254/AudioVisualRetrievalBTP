import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from tqdm import tqdm
import glob
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--save_path", type=str, default="best_img2img.pth")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MODEL
# ==========================================
class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze Backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Trainable Head
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

# ==========================================
# 2. DATASETS
# ==========================================
class TripletImageDataset(Dataset):
    """ Yields (Anchor, Positive, Negative) for Training/Val Loss """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_imgs = {c: glob.glob(os.path.join(root_dir, c, "*.*")) for c in self.classes}
        self.classes = [c for c in self.classes if len(self.class_to_imgs[c]) > 1]

    def __len__(self):
        return sum(len(imgs) for imgs in self.class_to_imgs.values())

    def __getitem__(self, index):
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        anc_path, pos_path = random.sample(self.class_to_imgs[target_class], 2)
        neg_path = random.choice(self.class_to_imgs[neg_class])
        
        return self._load(anc_path), self._load(pos_path), self._load(neg_path)
    
    def _load(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img

class ValidationGalleryDataset(Dataset):
    """ Yields Single Images for Metrics Calculation """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for i, cls in enumerate(self.classes):
            files = glob.glob(os.path.join(root_dir, cls, "*.*"))
            for f in files:
                self.paths.append(f)
                self.labels.append(i)

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[index]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. METRIC CALCULATION UTILS
# ==========================================
def calculate_metrics(model, dataloader, device):
    model.eval()
    feats = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            emb = model(imgs)
            feats.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
            
    feats = np.vstack(feats)
    labels = np.array(labels)
    
    # Cosine Similarity Matrix
    sim_matrix = cosine_similarity(feats)
    
    # Metrics
    r1, r5, r10 = 0, 0, 0
    n = len(labels)
    
    # Fill diagonal with -1 (ignore self-match)
    np.fill_diagonal(sim_matrix, -1)
    
    for i in range(n):
        # Sort indices by similarity descending
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        # Check if ground truth label is in top K
        top_k_labels = labels[sorted_indices[:10]]
        true_label = labels[i]
        
        if true_label in top_k_labels[:1]: r1 += 1
        if true_label in top_k_labels[:5]: r5 += 1
        if true_label in top_k_labels[:10]: r10 += 1
        
    return (r1/n)*100, (r5/n)*100, (r10/n)*100

def validate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anc, pos, neg in dataloader:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
            emb_a = model(anc)
            emb_p = model(pos)
            emb_n = model(neg)
            loss = criterion(emb_a, emb_p, emb_n)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    train_dir = os.path.join(args.data_root, "train", "vision")
    val_dir = os.path.join(args.data_root, "test", "vision") # Using 'test' as 'val' for metrics
    
    # Datasets
    train_ds = TripletImageDataset(train_dir, transform=transform)
    val_triplet_ds = TripletImageDataset(val_dir, transform=transform)
    val_gallery_ds = ValidationGalleryDataset(val_dir, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loss_loader = DataLoader(val_triplet_ds, batch_size=32, shuffle=False, num_workers=4)
    val_metric_loader = DataLoader(val_gallery_ds, batch_size=32, shuffle=False, num_workers=4)
    
    model = VisionEncoder().to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    best_r5 = 0.0
    
    print(f"--- Training Image-to-Image Model ---")
    print(f"Train Size: {len(train_ds)} triplets | Val Size: {len(val_gallery_ds)} images")
    
    for epoch in range(20):
        # 1. Train
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
        
        # 2. Validation Loss
        avg_val_loss = validate_loss(model, val_loss_loader, criterion, DEVICE)
        
        # 3. Validation Metrics (R@K)
        print(f"Calculating Metrics...")
        r1, r5, r10 = calculate_metrics(model, val_metric_loader, DEVICE)
        
        # 4. Logging & Saving
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