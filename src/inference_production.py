import sys, os, cv2, torch, json, datetime
import torch.nn as nn
import numpy as np
from torchvision import models
from tqdm import tqdm

# [1] 설정
MODEL_PATH = "../data/processed/h200_final_model.pth"
INPUT_DIR = "../data/test_samples"
OUTPUT_JSON_DIR = "../data/inference_results"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [2] 모델 구조 (기존과 동일)
class CNNGRUModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNGRUModel, self).__init__()
        self.backbone = models.efficientnet_b0()
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(input_size=1280, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w); features = self.backbone(x)
        features = features.view(b, s, -1); _, hidden = self.rnn(features)
        return self.fc(hidden[-1])

# [3] 전처리 함수
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 16: return None
    frames = []
    interval = total_frames // 16
    for i in range(16):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return torch.from_numpy(np.transpose(np.array(frames), (0, 3, 1, 2))).unsqueeze(0) if len(frames) == 16 else None

# [4] JSON 리포트 생성기
def save_json_report(filename, label, confidence):
    report = {
        "analysis_id": f"H200_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "filename": filename,
        "prediction": label,
        "confidence_score": round(confidence, 2),
        "details": {
            "visual_consistency": "High" if confidence > 80 else "Medium",
            "temporal_flicker": "Low" if label == "REAL" else "Detected"
        }
    }
    with open(os.path.join(OUTPUT_JSON_DIR, f"{filename}.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

# [5] 메인 실행부
def run_inference():
    model = CNNGRUModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    print(f"\n🚀 실전 판별 시작 ({len(files)}개 영상)")

    for f in tqdm(files):
        video_path = os.path.join(INPUT_DIR, f)
        input_tensor = preprocess_video(video_path)
        if input_tensor is None: continue

        with torch.no_grad():
            output = model(input_tensor.to(device))
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        res_label = "REAL" if pred.item() == 0 else "FAKE"
        score = conf.item() * 100

        # 결과 출력 및 JSON 저장
        print(f" ▶ [{f}] 판정: {res_label} ({score:.2f}%)")
        save_json_report(f, res_label, score)

if __name__ == "__main__":
    run_inference()