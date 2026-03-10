import sys, os, h5py, torch, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

# ==========================================
# ⚙️ 실험 제어판 (이 설정으로 성능 변화를 테스트하세요)
# ==========================================
AUG_LEVEL = "medium"  # 증강 강도: "low", "medium", "high"
DATA_RATIO = 1.0      # 데이터 사용량: 0.1(10%) ~ 1.0(100%)
BATCH_SIZE = 256      # H200 권장 배치
LEARNING_RATE = 1e-4
EPOCHS = 50
NUM_WORKERS = 12      # CPU 코어 활용도 상향
# ==========================================

# [1] 데이터 증강 세부 설정 및 설명
def get_augmentation(level):
    """
    - low: 좌우 반전만 적용 (가장 기초적인 변형)
    - medium: 반전 + 색상 변조 + 미세 회전 (일반적인 딥페이크 학습용)
    - high: 강한 변조 + 블러 추가 (모델이 극한의 상황을 견디게 함)
    """
    if level == "low":
        return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    elif level == "medium":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # 조명 차이 극복
            transforms.RandomRotation(degrees=10)               # 머리 각도 극복
        ])
    elif level == "high":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomRotation(degrees=20),
            transforms.GaussianBlur(kernel_size=3)              # 화질 저하 상황 가정
        ])
    return None

# [2] H200 최적화 데이터셋 클래스 (Lazy Loading 적용)
class DeepfakeDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None # 워커 프로세스별로 별도 핸들링
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['y']) # 기존 'labels' -> 'y'로 수정

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None: # 워커가 처음 실행될 때 파일 오픈 (I/O 가속)
            self.file = h5py.File(self.h5_path, 'r')
            
        video = self.file['x'][idx] # 'videos' -> 'x'로 수정
        label = self.file['y'][idx] # 'labels' -> 'y'로 수정
        video = torch.from_numpy(video).float()
        
        if self.transform:
            # 16프레임 전체에 동일한 랜덤 값을 적용하기 위해 시드 고정
            seed = random.randint(0, 2**32)
            transformed = []
            for frame in video:
                random.seed(seed); torch.manual_seed(seed)
                transformed.append(self.transform(frame))
            video = torch.stack(transformed)
            
        return video, label

# [3] 모델 구조 (EfficientNet + GRU)
class CNNGRUModel(nn.Module):
    def __init__(self):
        super(CNNGRUModel, self).__init__()
        # Pretrained weight 사용으로 학습 초기 안정성 확보
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

def train():
    device = torch.device("cuda")
    print(f"🚀 실험 시작 | 증강: {AUG_LEVEL} | 데이터량: {DATA_RATIO*100}%")

    # 전체 데이터셋 로드 및 증강 적용
    full_ds = DeepfakeDataset("../data/processed/dataset_balanced.h5", transform=get_augmentation(AUG_LEVEL))
    
    # 데이터 개수에 따른 성능 변화 테스트를 위한 샘플링
    num_samples = int(len(full_ds) * DATA_RATIO)
    indices = list(range(len(full_ds)))
    random.shuffle(indices)
    subset_ds = Subset(full_ds, indices[:num_samples])
    
    # 학습/검증 분할
    t_size = int(0.8 * len(subset_ds))
    v_size = len(subset_ds) - t_size
    train_ds, val_ds = torch.utils.data.random_split(subset_ds, [t_size, v_size])

    # H200 가속 옵션 적용 (pin_memory, prefetch)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model = CNNGRUModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision 가속기

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                loss = criterion(model(videos), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # 검증 (Validation)
        model.eval()
        correct = 0
        with torch.no_grad():
            for vv, ll in val_loader:
                vv, ll = vv.to(device), ll.to(device)
                with torch.amp.autocast('cuda'):
                    pred = model(vv).argmax(1)
                correct += (pred == ll).sum().item()
        
        acc = correct / v_size * 100
        print(f"📈 Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "../data/processed/v100_final_model.pth")
            print(f"🌟 Best Saved! ({acc:.2f}%)")

if __name__ == "__main__":
    train()