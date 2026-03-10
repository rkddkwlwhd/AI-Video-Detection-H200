# 프레임별 신뢰도 시각화
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# HDF5/환경 설정
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# 1. 모델 정의 (학습 시와 동일)
class CNNGRUModel(nn.Module):
    def __init__(self):
        super(CNNGRUModel, self).__init__()
        self.backbone = models.efficientnet_b0()
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(input_size=self.feature_dim, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        _, hidden = self.rnn(features)
        return self.fc(hidden[-1])

def analyze_video_flow(video_path, model_path, window_size=16, step=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNGRUModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        all_frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    total_f = len(all_frames)
    probabilities = []
    timestamps = []

    print(f"🧐 분석 중: {os.path.basename(video_path)} (총 {total_f} 프레임)")

    # 슬라이딩 윈도우 방식으로 구간별 확률 추출
    for i in range(0, total_f - window_size, step):
        window = all_frames[i : i + window_size]
        input_tensor = torch.from_numpy(np.transpose(window, (0, 3, 1, 2))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0][1].item() # FAKE일 확률
            probabilities.append(prob)
            timestamps.append(i + window_size//2) # 윈도우 중앙점 기록

    # 시각화
    plt.figure(figsize=(15, 6))
    
    # 1. 상단: 대표 프레임들 나열
    num_samples = 5
    for i in range(num_samples):
        idx = int(i * (total_f / num_samples))
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(all_frames[idx])
        plt.title(f"Frame {idx}")
        plt.axis('off')

    # 2. 하단: 확률 그래프
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, probabilities, color='red', marker='o', linestyle='-', markersize=4)
    plt.axhline(y=0.5, color='gray', linestyle='--') # 0.5 기준선
    plt.ylim(0, 1.1)
    plt.xlabel('Frame Number')
    plt.ylabel('Fake Probability')
    plt.title(f'Analysis: {os.path.basename(video_path)}')
    plt.grid(True, alpha=0.3)

    save_name = f"analysis_{os.path.basename(video_path)}.png"
    plt.tight_layout()
    plt.savefig(f"../data/processed/{save_name}")
    print(f"✅ 분석 그래프 저장 완료: data/processed/{save_name}")

if __name__ == "__main__":
    MODEL = "../data/processed/v100_baseline.pth"
    # 오답이 나왔던 real/video_0004.mp4를 넣어보세요
    TARGET = "../data/raw/real/video_0004.mp4" 
    
    if os.path.exists(TARGET):
        analyze_video_flow(TARGET, MODEL)