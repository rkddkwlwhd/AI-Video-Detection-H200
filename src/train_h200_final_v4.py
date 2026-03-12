import sys, os, h5py, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm
# NumPy 2.2.6을 지우고 1.x 대의 마지막 안정 버전으로 설치합니다.
# pip install --user "numpy<2"

# ==========================================
# ⚙️ H200 하이퍼 파라미터 & 저장 설정
# ==========================================
AUG_LEVEL = "medium"
DATA_RATIO = 1.0
BATCH_SIZE = 64            # 물리적 배치 (IndexMath 에러 방지)
ACCUMULATION_STEPS = 4     # 논리적 배치 (64 * 4 = 256 효과)
LEARNING_RATE = 2e-4       # 256 배치에 최적화된 학습률
EPOCHS = 50
NUM_WORKERS = 0            # Bus error 방지

# ⭐ 모델 저장 경로 수정 (H200의 위엄을 담아!)
MODEL_SAVE_PATH = "../data/processed/h200_final_model.pth"
# ==========================================

# [1] 데이터 증강 (온라인 방식 - 실시간 변형)
def get_augmentation(level):
    if level == "low":
        return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    elif level == "medium":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=10)
        ])
    return None

# [2] 데이터셋 클래스 (Lazy Loading 및 x, y 키 적용)
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
            transformed = []
            for frame in video:
                random.seed(seed); torch.manual_seed(seed)
                transformed.append(self.transform(frame))
            video = torch.stack(transformed)
        return video, label

# [3] 모델 구조 (EfficientNet-B0 + GRU)
class CNNGRUModel(nn.Module):
    def __init__(self):
        super(CNNGRUModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(1280, 256, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)
        )
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)
        features = features.view(b, s, -1)
        _, hidden = self.rnn(features)
        return self.fc(hidden[-1])

# [4] 학습 메인 함수
def train():
    device = torch.device("cuda")
    print(f"🔥 H200 모드 가동 | 저장명: {os.path.basename(MODEL_SAVE_PATH)}")

    full_ds = DeepfakeDataset("../data/processed/dataset_balanced.h5", transform=get_augmentation(AUG_LEVEL))
    num_samples = int(len(full_ds) * DATA_RATIO)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    subset_ds = Subset(full_ds, indices[:num_samples])
    
    t_size = int(0.8 * len(subset_ds))
    v_size = len(subset_ds) - t_size
    train_ds, val_ds = torch.utils.data.random_split(subset_ds, [t_size, v_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model = CNNGRUModel().to(device)
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
                outputs = model(videos)
                loss = criterion(outputs, labels) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            train_loss += loss.item() * ACCUMULATION_STEPS

            # ⭐ [개선] 정해진 단계가 찼거나, 데이터로더의 마지막 배치일 때 업데이트
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
                    pred = model(vv).argmax(1)
                correct += (pred == ll).sum().item()
        
        acc = (correct / v_size) * 100
        print(f"📈 Loss: {train_loss/num_batches:.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 Best Model Saved! ({acc:.2f}%)")

if __name__ == "__main__":
    train()