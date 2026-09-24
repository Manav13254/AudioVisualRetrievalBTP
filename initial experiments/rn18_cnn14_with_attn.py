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
# 0. SETUP
# ==========================================
def get_args():
    parser = argparse.ArgumentParser()
    home_dir = os.path.expanduser("~")
    default_root = os.path.join(home_dir, "ADVANCE_DATA_split")
    
    parser.add_argument("--data_root", type=str, default=default_root)
    parser.add_argument("--audio_weights", type=str, default="audioset_tagging_cnn/Cnn14_mAP=0.431.pth")
    parser.add_argument("--save_path", type=str, default="best_attention_encoders.pth")
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
# 1. ATTENTION MODULES (The "Original Paper" Parts)
# ==========================================

# --- A. Quaternion Attention (Vision) ---
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

# --- B. Coordinate Attention (Audio) ---
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

# ==========================================
# 2. UPGRADED ENCODERS
# ==========================================

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze Backbone
        for param in resnet.parameters():
            param.requires_grad = False
            
        # We need individual layers to do Multi-Scale Fusion
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Fusion Layers (The "Multi-Scale" part)
        self.f0conv = nn.Conv2d(64, 64, 3, 2, 1)
        self.f01conv = nn.Conv2d(128, 128, 7, 4, 3)
        self.f2conv = nn.Conv2d(128, 128, 3, 2, 1)
        self.fusionconv = nn.Conv2d(512, 512, 3, 2, 1)
        
        # Quaternion Attention Modules
        self.quater_att_fusion = QUATER_ATTENTION(512)
        self.quater_att_f4 = QUATER_ATTENTION(512)
        
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # 1. Backbone Features
        x = self.stem[0](x); x = self.stem[1](x); f0 = self.stem[2](x); x = self.stem[3](f0)
        f1 = self.layer1(x); f2 = self.layer2(f1); f3 = self.layer3(f2); f4 = self.layer4(f3)
        
        # 2. Fusion
        f0_d = self.f0conv(f0)
        f01 = self.f01conv(torch.cat([f0_d, f1], 1))
        f2_d = self.f2conv(f2)
        f23 = torch.cat([f2_d, f3], 1)
        
        fusion = self.fusionconv(torch.cat([f01, f23], 1))
        
        # 3. Attention
        fusion = self.quater_att_fusion(fusion)
        f4_att = self.quater_att_f4(f4)
        
        # 4. Gated Combination
        final = f4_att * torch.sigmoid(fusion) + torch.sigmoid(f4_att) * fusion + f4_att
        
        out = F.adaptive_avg_pool2d(final, (1, 1)).flatten(1)
        return F.normalize(self.project(out), p=2, dim=1)

class AudioEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.base_model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        
        # Freeze Backbone
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Coordinate Attention
        self.ca_block = CA_Block(2048)
        
        # Attentive Statistics Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )
        
        # Input size is 4096 because ASP concatenates Mean (2048) + Std (2048)
        self.project = nn.Linear(4096, embedding_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        
        # 1. Manual Backbone Forward
        x = self.base_model.spectrogram_extractor(x)
        x = self.base_model.logmel_extractor(x)
        x = x.transpose(1, 3); x = self.base_model.bn0(x); x = x.transpose(1, 3)
        
        x = self.base_model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.base_model.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        
        # 2. Coordinate Attention
        x = self.ca_block(x)
        
        # 3. Attentive Statistics Pooling
        x = torch.mean(x, dim=3).transpose(1, 2) # (Batch, Time, Channels)
        w = self.attn_pool(x)
        mu = torch.sum(x * w, dim=1)
        var = torch.sum(w * (x - mu.unsqueeze(1))**2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))
        
        out = torch.cat([mu, std], 1)
        return F.normalize(self.project(out), p=2, dim=1)

class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = VisionEncoder()
        self.audio_model = AudioEncoder()

# ==========================================
# 3. DATASET & METRICS (Standard)
# ==========================================
class BidirectionalDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, transform=None):
        self.vision_dir = vision_dir; self.audio_dir = audio_dir; self.transform = transform
        v_classes = set(os.listdir(vision_dir)); a_classes = set(os.listdir(audio_dir))
        self.classes = sorted(list(v_classes.intersection(a_classes)))
        self.class_to_imgs = {c: glob.glob(os.path.join(vision_dir, c, "*.*")) for c in self.classes}
        self.class_to_auds = {c: glob.glob(os.path.join(audio_dir, c, "*.wav")) for c in self.classes}

    def __len__(self): return sum(len(imgs) for imgs in self.class_to_imgs.values())

    def _process_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            if sr != 32000: wav = T.Resample(sr, 32000)(wav)
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            if wav.shape[1] < 320000: wav = F.pad(wav, (0, 320000 - wav.shape[1]))
            else: wav = wav[:, :320000]
            return wav
        except: return torch.zeros(1, 320000)

    def __getitem__(self, index):
        target_class = random.choice(self.classes)
        neg_class = random.choice([c for c in self.classes if c != target_class])
        
        img_path = random.choice(self.class_to_imgs[target_class])
        aud_pos_path = random.choice(self.class_to_auds[target_class])
        img_neg_path = random.choice(self.class_to_imgs[neg_class])
        aud_neg_path = random.choice(self.class_to_auds[neg_class])
        
        img_anchor = Image.open(img_path).convert('RGB')
        img_neg = Image.open(img_neg_path).convert('RGB')
        
        if self.transform: img_anchor = self.transform(img_anchor); img_neg = self.transform(img_neg)
        aud_pos = self._process_audio(aud_pos_path); aud_neg = self._process_audio(aud_neg_path)
        return img_anchor, img_neg, aud_pos, aud_neg

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def compute_metrics(model, v_dir, a_dir, k_values=[1, 5, 10]):
    model.eval()
    classes = sorted(os.listdir(v_dir))
    v_feats, a_feats, labels = [], [], []
    
    with torch.no_grad():
        for i, cls in enumerate(classes):
            for f in glob.glob(os.path.join(v_dir, cls, "*.*")):
                try:
                    img = Image.open(f).convert('RGB')
                    img = vision_transform(img).unsqueeze(0).to(DEVICE)
                    v_feats.append(model.vision_model(img).cpu().numpy())
                    labels.append(i)
                except: continue
                
    a_feats_dict = {i: [] for i in range(len(classes))}
    for i, cls in enumerate(classes):
        for f in glob.glob(os.path.join(a_dir, cls, "*.wav")):
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
            
    v_feats = np.vstack(v_feats); final_a_feats = np.vstack(final_a_feats)
    v_labels = np.array(labels); a_labels = np.array(final_a_labels)
    
    sim_matrix = cosine_similarity(v_feats, final_a_feats)
    
    def calc_r(matrix, q_lbl, g_lbl):
        res = {k: 0 for k in k_values}
        for idx in range(len(q_lbl)):
            indices = np.argsort(matrix[idx])[::-1]
            top = g_lbl[indices[:10]]
            for k in k_values:
                if q_lbl[idx] in top[:k]: res[k] += 1
        return {k: (v/len(q_lbl))*100 for k,v in res.items()}

    i2a = calc_r(sim_matrix, v_labels, a_labels)
    a2i = calc_r(sim_matrix.T, a_labels, v_labels)
    return i2a, a2i

# ==========================================
# 4. MAIN
# ==========================================
def main():
    TRAIN_V = os.path.join(args.data_root, "train", "vision")
    TRAIN_A = os.path.join(args.data_root, "train", "sound")
    VAL_V = os.path.join(args.data_root, "test", "vision")
    VAL_A = os.path.join(args.data_root, "test", "sound")

    model = CrossModalModel().to(DEVICE)
    
    # Load Original Weights for Audio Backbone
    if os.path.exists(args.audio_weights):
        print(f"Loading AudioSet Weights: {args.audio_weights}")
        ckpt = torch.load(args.audio_weights, map_location=DEVICE, weights_only=True)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.audio_model.base_model.load_state_dict(state_dict, strict=False)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    train_ds = BidirectionalDataset(TRAIN_V, TRAIN_A, transform=vision_transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    
    best_avg_r5 = 0.0
    print(f"--- Training with Attention Encoders (Backbones Frozen) ---")
    
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
            
            loss = criterion(v_a, a_p, a_n) + criterion(a_p, v_a, v_n)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print("Validating...")
        i2a, a2i = compute_metrics(model, VAL_V, VAL_A)
        avg_r5 = (i2a[5] + a2i[5]) / 2
        
        print(f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f}")
        print(f"  I->A: R@1 {i2a[1]:.1f} | R@5 {i2a[5]:.1f}")
        print(f"  A->I: R@1 {a2i[1]:.1f} | R@5 {a2i[5]:.1f}")
        
        if avg_r5 > best_avg_r5:
            best_avg_r5 = avg_r5
            torch.save(model.state_dict(), args.save_path)
            print("  🔥 Best Attention Model Saved!")

if __name__ == "__main__":
    main()