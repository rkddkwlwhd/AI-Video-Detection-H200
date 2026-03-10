import sys, os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from tqdm import tqdm
import contextlib

# 1. 환경 설정 및 로그 차단
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['AV_LOG_LEVEL'] = 'quiet'

class Color:
    GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'
    BLUE = '\033[94m'; BOLD = '\033[1m'; END = '\033[0m'; CYAN = '\033[96m'

# [모델 구조 - 97.5% 버전]
class CNNGRUModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNGRUModel, self).__init__()
        self.backbone = models.efficientnet_b0()
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(input_size=self.feature_dim, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)
        features = features.view(b, s, -1)
        _, hidden = self.rnn(features)
        return self.fc(hidden[-1])

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
        try: yield
        finally: os.dup2(old_stderr, sys.stderr.fileno())

def preprocess_video(video_path):
    with suppress_stderr():
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        frames = []
        interval = max(1, total_frames // 16)
        for i in range(16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
    if len(frames) != 16: return None
    return torch.from_numpy(np.transpose(np.array(frames), (0, 3, 1, 2))).unsqueeze(0)

def run_detailed_inference(folder, model_path, expected_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNGRUModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    files = sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])
    wrong_list = []      # 틀린 영상
    low_conf_list = []   # 맞았지만 신뢰도 낮은 영상
    correct_count = 0

    print(f"\n{Color.BOLD}{Color.CYAN}📂 분석 시작: {folder} (정답: {expected_label}){Color.END}")
    print(f"{'파일명':<35} | {'결과':<10} | {'신뢰도'}")
    print("-" * 65)

    for f in tqdm(files, desc="Processing", leave=True):
        input_tensor = preprocess_video(os.path.join(folder, f))
        if input_tensor is None: continue

        with torch.no_grad():
            output = model(input_tensor.to(device))
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        
        res_label = "REAL" if pred.item() == 0 else "FAKE"
        score = conf.item() * 100
        
        is_correct = (res_label == expected_label)
        c_code = Color.GREEN if is_correct else Color.RED
        
        # 결과 저장 로직
        if not is_correct:
            wrong_list.append((f, res_label, score))
        elif score < 85.0:
            low_conf_list.append((f, res_label, score))
        
        if is_correct: correct_count += 1
        
        # 실시간 출력
        status = "" if is_correct else f"{Color.RED}[WRONG]{Color.END}"
        if is_correct and score < 85.0: status = f"{Color.YELLOW}[LOW]{Color.END}"
        
        print(f"{f[:33]:<35} | {c_code}{res_label:<10}{Color.END} | {score:6.2f}% {status}")

    acc = (correct_count / len(files)) * 100 if files else 0
    print("-" * 65)
    print(f"{Color.BOLD}📊 {expected_label} 폴더 정확도: {acc:.2f}%{Color.END}")
    
    return wrong_list, low_conf_list

if __name__ == "__main__":
    MODEL = "../data/processed/v100_final_model.pth"
    
    w_fake, l_fake = run_detailed_inference("../data/raw/fake", MODEL, "FAKE")
    w_real, l_real = run_detailed_inference("../data/raw/real", MODEL, "REAL")
    
    # [최종 종합 오답 노트]
    print(f"\n{Color.BOLD}{'='*25} 📋 최종 분석 리포트 {'='*25}{Color.END}")
    
    total_wrong = w_fake + w_real
    if total_wrong:
        print(f"\n{Color.RED}{Color.BOLD}❌ [오답 리스트] - 모델이 틀린 영상 ({len(total_wrong)}개):{Color.END}")
        for f, res, sc in total_wrong:
            print(f"  - {f:<35} | 판정: {res:<5} | 확신도: {sc:.2f}%")
            
    total_low = l_fake + l_real
    if total_low:
        print(f"\n{Color.YELLOW}{Color.BOLD}⚠️  [헷갈림 리스트] - 맞았으나 불안한 영상 ({len(total_low)}개):{Color.END}")
        for f, res, sc in total_low:
            print(f"  - {f:<35} | 판정: {res:<5} | 확신도: {sc:.2f}%")
            
    if not total_wrong and not total_low:
        print(f"\n{Color.GREEN}🎉 완벽합니다! 모든 영상을 높은 신뢰도로 맞혔습니다.{Color.END}")
    print(f"\n{Color.BOLD}{'='*70}{Color.END}")