import sys
import os
import site

# [1] H200 전용 경로 및 시스템 라이브러리 충돌 방지 설정
user_site_packages = site.getusersitepackages()
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

sys.modules['transformer_engine'] = None # H200 가속 라이브러리 충돌 방지

torch_lib_path = os.path.join(user_site_packages, "torch", "lib")
if os.path.exists(torch_lib_path):
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# [2] 필수 라이브러리 임포트
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 기존 모델 클래스 가져오기 (train_xai.py 파일이 같은 폴더에 있어야 함)
from train_xai import ExplainableDeepfakeModel

# [3] 하이퍼파라미터 설정 (H200 최적화: 32비트 연산 오류 방지용)
BATCH_SIZE = 32           # 물리적 배치 사이즈 (연산 에러 방지용)
ACCUMULATION_STEPS = 4     # 그래디언트 누적 (32 * 4 = 128 배치 효과)
EPOCHS = 30                # 전체 학습 횟수
LR = 1e-4                  # 학습률

# 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(BASE_DIR, "data_audio", "processed", "audio_dataset_v2.h5")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "data_audio", "processed", "deepvoice_attention_model.pth")

# [4] 데이터셋 클래스 정의
class DeepfakeAudioDataset(Dataset):
    def __init__(self, h5_path, train_mode=True):
        self.h5_path = h5_path
        self.train_mode = train_mode
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['y'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            x = torch.from_numpy(f['x'][idx]).float()
            y = torch.tensor(f['y'][idx]).long()
        
        # 성별 편향 억제용 미세 노이즈 추가 (훈련 모드일 때만)
        if self.train_mode:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            
        return x, y

# [5] 학습 메인 함수
def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎙️ H200 딥보이스 탐지 학습 가동 (Batch: {BATCH_SIZE}x{ACCUMULATION_STEPS})")

    # 데이터 로더 준비 (Shared Memory 에러 방지를 위해 num_workers=0 설정)
    full_dataset = DeepfakeAudioDataset(H5_PATH)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 모드 설정
    train_ds.dataset.train_mode = True
    val_ds.dataset.train_mode = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True, num_workers=0)

    # 모델 및 최적화 도구 설정
    model = ExplainableDeepfakeModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # H200 가속을 위한 Mixed Precision (최신 PyTorch 방식 적용)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad() # 누적 단계 시작 전 초기화
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (videos, labels) in enumerate(pbar):
            videos, labels = videos.to(device), labels.to(device)

            # Mixed Precision 연산
            with torch.amp.autocast('cuda'):
                video_logits, frame_logits, _ = model(videos)
                
                # 전체 영상 손실
                loss_video = criterion(video_logits, labels)
                
                # 프레임별 손실 (16개 조각)
                b, s, _ = frame_logits.shape
                frame_labels = labels.unsqueeze(1).expand(-1, s).reshape(-1)
                loss_frame = criterion(frame_logits.reshape(-1, 2), frame_labels)
                
                # 최종 Loss 계산 (그래디언트 누적을 위해 단계수로 나눔)
                loss = (loss_video + 0.5 * loss_frame) / ACCUMULATION_STEPS

            # 역전파 (그래디언트 누적)
            scaler.scale(loss).backward()

            # 정해진 단계마다 가중치 업데이트
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=train_loss/(i+1))

        # --- 검증 (Validation) ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for vv, ll in val_loader:
                vv, ll = vv.to(device), ll.to(device)
                with torch.amp.autocast('cuda'):
                    v_logits, _, _ = model(vv)
                    correct += (v_logits.argmax(1) == ll).sum().item()
        
        acc = (correct / val_size) * 100
        print(f"📊 Validation Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🌟 최고 점수 갱신! 모델 저장됨: {acc:.2f}%")

if __name__ == "__main__":
    train_audio_model()