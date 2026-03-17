import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 기존에 작성했던 모델과 데이터셋 클래스를 그대로 import 하거나 정의합니다.
# (파일 상단에 ExplainableDeepfakeModel, DeepfakeDataset 클래스가 있다고 가정)

# [설정]
H5_PATH = "../data_audio/processed/audio_dataset.h5"
MODEL_SAVE_PATH = "../data_audio/processed/deepvoice_attention_model.pth"
BATCH_SIZE = 64
EPOCHS = 30

def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎙️ H200 딥보이스 탐지 학습 가동...")

    # 1. 데이터 로드 (기존 DeepfakeDataset 클래스 재활용)
    from train_xai import DeepfakeDataset, ExplainableDeepfakeModel
    
    dataset = DeepfakeDataset(H5_PATH, transform=None) # 스펙트로그램은 이미 정규화됨
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

    # 2. 모델 및 최적화 설정
    model = ExplainableDeepfakeModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. 학습 루프 (기존 train_xai.py의 로직과 동일)
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            video_logits, frame_logits, _ = model(videos)
            
            loss_video = criterion(video_logits, labels)
            # 프레임(스펙트로그램 조각)별 손실 계산
            b, s, _ = frame_logits.shape
            frame_labels = labels.unsqueeze(1).expand(-1, s).reshape(-1)
            loss_frame = criterion(frame_logits.reshape(-1, 2), frame_labels)
            
            loss = loss_video + 0.5 * loss_frame
            loss.backward()
            optimizer.step()

        # 검증
        model.eval()
        correct = 0
        with torch.no_grad():
            for vv, ll in val_loader:
                v_logits, _, _ = model(vv.to(device))
                correct += (v_logits.argmax(1) == ll.to(device)).sum().item()
        
        acc = (correct / val_size) * 100
        print(f"Epoch {epoch+1} | Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("🌟 모델 저장됨!")

if __name__ == "__main__":
    train_audio_model()