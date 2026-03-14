import sys, os, h5py, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

# ==========================================
# ⚙️ H200 하이퍼 파라미터 & 저장 설정
# ==========================================
AUG_LEVEL = "high"         # 🔥 REAL 오판을 줄이기 위해 무조건 'high' 권장
DATA_RATIO = 1.0
BATCH_SIZE = 64            
ACCUMULATION_STEPS = 4     # 64 * 4 = 256 배치 효과
LEARNING_RATE = 2e-4       
EPOCHS = 50
NUM_WORKERS = 0            

# ⭐ 새로운 Attention 모델 저장 경로
MODEL_SAVE_PATH = "../data/processed/h200_attention_model.pth"
# ==========================================

# [1] 데이터 증강 (REAL 영상 오판 방지용 화질 저하 집중)
def get_augmentation(level):
    if level == "high":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(degrees=15),
            # 🔥 핵심: 야생의 REAL 영상(압축, 노이즈)에 내성을 갖도록 블러 추가
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
        ])
    return None

# [2] 데이터셋 클래스 (기존과 동일)
class DeepfakeDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['y'])

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        video = self.file['x'][idx]
        label = self.file['y'][idx]
        video = torch.from_numpy(video).float()
        if self.transform:
            seed = random.randint(0, 2**32)
            transformed =[]
            for frame in video:
                random.seed(seed); torch.manual_seed(seed)
                transformed.append(self.transform(frame))
            video = torch.stack(transformed)
        return video, label

# [3] 🧠 Attention 기반 설명 가능한 모델 구조 (새로 추가)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, rnn_output):
        attn_weights = torch.softmax(self.attention(rnn_output), dim=1) # (B, Seq_len, 1)
        context = torch.sum(attn_weights * rnn_output, dim=1)           # (B, Hidden)
        return context, attn_weights.squeeze(-1)

class ExplainableDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ExplainableDeepfakeModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT') # ImageNet 가중치 유지
        self.backbone.classifier = nn.Identity()
        
        self.rnn = nn.GRU(1280, 256, num_layers=2, batch_first=True)
        self.attention = TemporalAttention(256)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        
        # 🔥 프레임별 결과를 뽑아내기 위한 레이어
        self.frame_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)
        features = features.view(b, s, -1)
        
        rnn_out, _ = self.rnn(features)
        
        context, attn_weights = self.attention(rnn_out)
        video_logits = self.fc(context)
        
        # 프레임별 로짓 추출
        frame_logits = self.frame_fc(rnn_out) 
        
        return video_logits, frame_logits, attn_weights

#[4] 학습 메인 함수
def train():
    device = torch.device("cuda")
    print(f"🔥 H200 모드 가동 | XAI(설명 가능한 AI) 학습 시작 | 저장명: {os.path.basename(MODEL_SAVE_PATH)}")

    full_ds = DeepfakeDataset("../data/processed/dataset_balanced.h5", transform=get_augmentation(AUG_LEVEL))
    num_samples = int(len(full_ds) * DATA_RATIO)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    subset_ds = Subset(full_ds, indices[:num_samples])
    
    t_size = int(0.8 * len(subset_ds))
    v_size = len(subset_ds) - t_size
    train_ds, val_ds = torch.utils.data.random_split(subset_ds,[t_size, v_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model = ExplainableDeepfakeModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0
        num_batches = len(train_loader)
        
        for i, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                # 3개의 아웃풋 반환
                video_logits, frame_logits, _ = model(videos)
                
                # 1. 전체 비디오 손실 계산
                loss_video = criterion(video_logits, labels)
                
                # 2. 개별 프레임 손실 계산 (16프레임 모두에 정답 라벨 부여)
                # 라벨 형태 변환: (Batch) -> (Batch, 16) -> (Batch * 16)
                b, seq_len, _ = frame_logits.shape
                frame_labels = labels.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                loss_frame = criterion(frame_logits.reshape(-1, 2), frame_labels)
                
                # 🔥 최종 Loss: 비디오 예측값과 프레임 예측값의 손실을 합산 (가중치 0.5 부여)
                loss = (loss_video + 0.5 * loss_frame) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            train_loss += loss.item() * ACCUMULATION_STEPS

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        # 검증 (Validation)
        model.eval()
        correct = 0
        with torch.no_grad():
            for vv, ll in val_loader:
                vv, ll = vv.to(device), ll.to(device)
                with torch.amp.autocast('cuda'):
                    # 검증 시에는 전체 비디오 로짓(video_logits)만 사용해 정확도 측정
                    video_logits, _, _ = model(vv)
                    pred = video_logits.argmax(1)
                correct += (pred == ll).sum().item()
        
        acc = (correct / v_size) * 100
        print(f"📈 Loss: {train_loss/num_batches:.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 Best Model Saved! ({acc:.2f}%)")

if __name__ == "__main__":
    train()