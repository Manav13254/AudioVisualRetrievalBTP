import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--model_path", type=str, default="best_img2img.pth")
    return parser.parse_args()

args = get_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. MODEL DEFINITION (Must match training)
# ==========================================
class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=None) # Load structure only
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
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
# 2. FEATURE EXTRACTION
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(model, data_dir):
    model.eval()
    embeddings = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    print("--- Extracting Features ---")
    with torch.no_grad():
        for label_idx, class_name in enumerate(tqdm(class_names)):
            img_paths = glob.glob(os.path.join(data_dir, class_name, "*.*"))
            for path in img_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                    emb = model(img_tensor).cpu().numpy()
                    embeddings.append(emb)
                    labels.append(label_idx)
                except:
                    continue
                    
    return np.vstack(embeddings), np.array(labels), class_names

# ==========================================
# 3. VISUALIZATIONS
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
    plt.title("t-SNE of Image Embeddings (Frozen ResNet18)")
    plt.tight_layout()
    plt.savefig("img2img_tsne.png")
    print("Saved: img2img_tsne.png")

def plot_confusion_matrix(y_true, y_pred, class_names):
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for better visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='.2f')
    
    plt.title("Retrieval Confusion Matrix (Top-1 Match - Normalized)")
    plt.tight_layout()
    plt.savefig("img2img_confusion_matrix.png")
    print("Saved: img2img_confusion_matrix.png")

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================
def main():
    test_dir = os.path.join(args.data_root, "test", "vision")
    
    # Load Model
    model = VisionEncoder().to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"Loaded weights from {args.model_path}")
    else:
        print("Warning: No weights found, using random init.")

    # 1. Extract All Features
    feats, labels, class_names = extract_features(model, test_dir)
    
    # 2. Compute Similarity Matrix (N x N)
    print("Computing Similarity Matrix...")
    # Shape: (N_samples, N_samples)
    sim_matrix = feats @ feats.T 
    
    # 3. Calculate Recall & Predictions for Confusion Matrix
    y_true = []
    y_pred = []
    
    r1, r5, r10 = 0, 0, 0
    n_samples = len(labels)
    
    # Fill diagonal with -1 to ignore self-match
    np.fill_diagonal(sim_matrix, -1)
    
    # For every image, treat it as a Query
    for i in range(n_samples):
        # Sort by similarity (Highest to Lowest)
        scores = sim_matrix[i]
        
        # Get indices of top matches
        sorted_indices = np.argsort(scores)[::-1]
        
        # Ground Truth Class
        true_cls = labels[i]
        
        # Top-K Retrieved Classes
        top_10_cls = labels[sorted_indices[:10]]
        
        # Check Hits
        if true_cls in top_10_cls[:1]: r1 += 1
        if true_cls in top_10_cls[:5]: r5 += 1
        if true_cls in top_10_cls[:10]: r10 += 1
        
        # For Confusion Matrix: Take the Class of the Top-1 result
        pred_cls = labels[sorted_indices[0]]
        
        y_true.append(true_cls)
        y_pred.append(pred_cls)
        
    # 4. Print Metrics
    print("\n=== Image-to-Image Retrieval Results ===")
    print(f"R@1:  {r1/n_samples*100:.2f}%")
    print(f"R@5:  {r5/n_samples*100:.2f}%")
    print(f"R@10: {r10/n_samples*100:.2f}%")
    
    # 5. Generate Plots
    plot_tsne(feats, labels, class_names)
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()