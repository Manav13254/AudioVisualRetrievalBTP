import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import torchaudio
import torchaudio.transforms as T
import argparse
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    # Defaulting to the name your training script saves
    parser.add_argument("--model_path", type=str, default="best_attention_encoders.pth")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for ICLM interaction")
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
    print("Error loading Cnn14. Ensure the audioset repository is correctly cloned.")
    sys.exit()

# ==========================================
# 1. ATTENTION & INTERACTION MODULES
# ==========================================
class QUATER_ATTENTION(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(QUATER_ATTENTION, self).__init__()
        self.fc_h = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_h = nn.BatchNorm2d(in_planes // ratio)
        self.relu_h = nn.ReLU()
        self.conv_h_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_w = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn_w = nn.BatchNorm2d(in_planes // ratio)
        self.relu_w = nn.ReLU()
        self.conv_w_sptial = nn.Conv2d(2 * (in_planes // ratio), in_planes // ratio, 7, padding=3, bias=False)

        self.fc_general = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h_avg = torch.mean(x, dim=3, keepdim=True)
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_w_avg = torch.mean(x, dim=2, keepdim=True)
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)

        x_h_avg = self.relu_h(self.bn_h(self.fc_h(x_h_avg)))
        x_h_max = self.relu_h(self.bn_h(self.fc_h(x_h_max)))
        x_w_avg = self.relu_w(self.bn_w(self.fc_w(x_w_avg)))
        x_w_max = self.relu_w(self.bn_w(self.fc_w(x_w_max)))

        x_h_cat_sp = self.conv_h_sptial(torch.cat([x_h_avg, x_h_max], dim=1))
        x_w_cat_sp = self.conv_w_sptial(torch.cat([x_w_avg, x_w_max], dim=1))

        x_general = self.fc_general(x_h_cat_sp * x_w_cat_sp)
        return x * self.sigmoid(x_general)

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.F_w = nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()   

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        return x * s_h.expand_as(x) * s_w.expand_as(x)

class ICLM(nn.Module):
    def __init__(self, embed_dim=128, heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5

        self.v_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.a_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, v_o, a_o):
        B = v_o.size(0)

        def reshape_heads(x):
            return x.view(B, self.heads, 1, self.embed_dim // self.heads)

        vq = reshape_heads(self.v_q(v_o))
        vk = reshape_heads(self.v_k(v_o))
        vv = reshape_heads(self.v_v(v_o))

        aq = reshape_heads(self.a_q(a_o))
        ak = reshape_heads(self.a_k(a_o))
        av = reshape_heads(self.a_v(a_o))

        dots_v_ai = torch.einsum('bhid,bhjd->bhij', aq, vk) * self.scale
        attn_v_ai = dots_v_ai.softmax(dim=-1)
        v_ai = torch.einsum('bhij,bhjd->bhid', attn_v_ai, vv).reshape(B, self.embed_dim)

        dots_a_vi = torch.einsum('bhid,bhjd->bhij', vq, ak) * self.scale
        attn_a_vi = dots_a_vi.softmax(dim=-1)
        a_vi = torch.einsum('bhij,bhjd->bhid', attn_a_vi, av).reshape(B, self.embed_dim)

        v_local = torch.sigmoid(a_vi) * v_ai
        v_global = torch.sigmoid(a_o) * v_ai
        v_final = v_local + v_global + v_o

        a_fine = torch.sigmoid(v_ai) * a_o
        a_iteration = torch.sigmoid(a_vi) * a_o
        a_final = a_fine + a_iteration + a_o

        return F.normalize(v_final, p=2, dim=1), F.normalize(a_final, p=2, dim=1)

# ==========================================
# 2. ENCODERS & MODEL ARCHITECTURE
# ==========================================
class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=None) 
        
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.f0conv = nn.Conv2d(64, 64, 3, 2, 1)
        self.f01conv = nn.Conv2d(128, 128, 7, 4, 3)
        self.f2conv = nn.Conv2d(128, 128, 3, 2, 1)
        self.fusionconv = nn.Conv2d(512, 512, 3, 2, 1)
        
        self.quater_att_fusion = QUATER_ATTENTION(512)
        self.quater_att_f4 = QUATER_ATTENTION(512)
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.stem[0](x); x = self.stem[1](x); f0 = self.stem[2](x); x = self.stem[3](f0)
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        
        f0_d = self.f0conv(f0)
        f01 = self.f01conv(torch.cat([f0_d, f1], 1))
        f2_d = self.f2conv(f2)
        f23 = torch.cat([f2_d, f3], 1)
        
        fusion = self.fusionconv(torch.cat([f01, f23], 1))
        fusion = self.quater_att_fusion(fusion)
        f4_att = self.quater_att_f4(f4)
        
        final = f4_att * torch.sigmoid(fusion) + torch.sigmoid(f4_att) * fusion + f4_att
        out = F.adaptive_avg_pool2d(final, (1, 1)).flatten(1)
        return F.normalize(self.project(out), p=2, dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        self.ca_block = CA_Block(2048)
        self.attn_pool = nn.Sequential(nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1))
        self.project = nn.Linear(4096, embedding_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        
        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.base_model.bn0(x); x = x.transpose(1, 3)
        
        x = self.base_model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        x = self.ca_block(x)
        x = torch.mean(x, dim=3).transpose(1, 2)
        w = self.attn_pool(x)
        mu = torch.sum(x * w, dim=1)
        var = torch.sum(w * (x - mu.unsqueeze(1))**2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))
        
        out = torch.cat([mu, std], 1)
        return F.normalize(self.project(out), p=2, dim=1)

class CrossModalModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_model = VisionEncoder(embedding_dim)
        self.audio_model = AudioEncoder(embedding_dim)
        self.iclm = ICLM(embed_dim=embedding_dim)

# ==========================================
# 3. BASE FEATURE EXTRACTION
# ==========================================
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_base_features(model, v_dir, a_dir):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    
    v_base, a_base = [], []
    v_labels, a_labels = [], []
    
    print("--- Extracting Base Features ---")
    with torch.no_grad():
        for i, cls in enumerate(tqdm(classes)):
            # Vision
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                try:
                    img = Image.open(f).convert('RGB')
                    img = vision_transform(img).unsqueeze(0).to(DEVICE)
                    v_base.append(model.vision_model(img).cpu())
                    v_labels.append(i)
                except: continue
            
            # Audio
            for f in glob.glob(os.path.join(a_dir, cls, "*.wav")):
                try:
                    wav, sr = torchaudio.load(f)
                    if sr!=32000: wav=T.Resample(sr,32000)(wav)
                    if wav.shape[1]<320000: wav=F.pad(wav,(0,320000-wav.shape[1]))
                    else: wav=wav[:,:320000]
                    if wav.shape[0]>1: wav=torch.mean(wav,dim=0,keepdim=True)
                    wav = wav.to(DEVICE)
                    a_base.append(model.audio_model(wav).cpu())
                    a_labels.append(i)
                except: continue
                
    return torch.cat(v_base, dim=0), torch.cat(a_base, dim=0), np.array(v_labels), np.array(a_labels), classes

# ==========================================
# 4. PLOTTING FUNCTIONS
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
        
    plt.title("Joint t-SNE (ICLM Base Features): Vision (▲) vs Audio (●)")
    plt.tight_layout()
    plt.savefig("iclm_tsne_joint.png")
    print("Saved iclm_tsne_joint.png")
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
        
        ax.set_title(f"Class: {cls}", fontsize=14, fontweight='bold')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0: ax.legend(loc='upper right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig("iclm_tsne_subplots.png", dpi=300)
    print("Saved iclm_tsne_subplots.png")

def plot_single_confusion(sim_matrix, query_labels, gallery_labels, class_names, title, filename, ylabel, xlabel, cmap='Purples'):
    print(f"Generating Confusion Matrix: {title}...")
    
    y_true, y_pred = [], []
    for i in range(len(query_labels)):
        top_idx = np.argmax(sim_matrix[i])
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
    print(f"Saved {filename}")

# ==========================================
# 5. MAIN EVALUATION
# ==========================================
def main():
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")
    
    model = CrossModalModel().to(DEVICE)
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    else:
        print("Warning: Model weights not found! Check your path.")
        sys.exit()

    # 1. Extract Base Features
    v_base, a_base, v_lbls, a_lbls, classes = extract_base_features(model, VAL_V, VAL_A)
    
    # 2. Compute Cross-Modal Similarity Matrix via ICLM
    N, M = v_base.size(0), a_base.size(0)
    sim_matrix = np.zeros((N, M))
    
    print("\n--- Running ICLM Cross-Interaction ---")
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(N), desc="Images against Audios"):
            v_row = v_base[i].unsqueeze(0).to(DEVICE)
            
            for j in range(0, M, args.batch_size):
                a_chunk = a_base[j:j+args.batch_size].to(DEVICE)
                v_chunk = v_row.expand(a_chunk.size(0), -1)
                
                v_final, a_final = model.iclm(v_chunk, a_chunk)
                sims = F.cosine_similarity(v_final, a_final, dim=1).cpu().numpy()
                sim_matrix[i, j:j+args.batch_size] = sims
    
    # 3. Calculate Retrieval Metrics
    print("\nCalculating Metrics...")
    
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
    print("   FINAL RETRIEVAL RESULTS (ICLM MODEL)")
    print("="*40)
    print(f"Image -> Audio  | R@1: {r1_i:.2f}% | R@5: {r5_i:.2f}% | R@10: {r10_i:.2f}%")
    print(f"Audio -> Image  | R@1: {r1_a:.2f}% | R@5: {r5_a:.2f}% | R@10: {r10_a:.2f}%")
    print("-" * 40)
    print(f"AVERAGE         | R@1: {avg_r1:.2f}% | R@5: {avg_r5:.2f}% | R@10: {avg_r10:.2f}%")
    print("="*40 + "\n")
    
    # 4. Plotting
    reduced_coords = plot_joint_tsne(v_base.numpy(), a_base.numpy(), v_lbls, a_lbls, classes)
    plot_class_subplots(reduced_coords, v_lbls, a_lbls, classes)
    
    plot_single_confusion(
        sim_matrix=sim_matrix, 
        query_labels=v_lbls, gallery_labels=a_lbls, 
        class_names=classes, 
        title="Image-to-Audio (ICLM Model)", 
        filename="iclm_confusion_img2aud.png",
        ylabel="True Image Class (Query)",
        xlabel="Predicted Audio Class (Retrieved)",
        cmap='Blues'
    )
    
    plot_single_confusion(
        sim_matrix=sim_matrix.T, 
        query_labels=a_lbls, gallery_labels=v_lbls, 
        class_names=classes, 
        title="Audio-to-Image (ICLM Model)", 
        filename="iclm_confusion_aud2img.png",
        ylabel="True Audio Class (Query)",
        xlabel="Predicted Image Class (Retrieved)",
        cmap='Oranges'
    )

if __name__ == "__main__":
    main()