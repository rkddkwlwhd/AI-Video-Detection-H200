import sys, os, cv2, torch, time
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# ⚙️ H200 하드웨어 최적화 설정
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth"
INPUT_DIR = "../data/test_samples"
OUTPUT_DIR = "../data/inference_results/gradcam_videos"

# H200 메모리를 고려한 배치 사이즈 상향 (32~64 권장)
BATCH_SIZE = 32  
# 전처리에 사용할 CPU 코어 개수 (서버 사양에 따라 4, 8, 16 조절)
NUM_WORKERS = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [모델 구조 ExplainableDeepfakeModel, TemporalAttention 클래스 동일]
# (이전 대화의 모델 정의 부분을 그대로 사용하세요)
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, rnn_output):
        attn_weights = torch.softmax(self.attention(rnn_output), dim=1)
        context = torch.sum(attn_weights * rnn_output, dim=1)
        return context, attn_weights.squeeze(-1)

class ExplainableDeepfakeModel(nn.Module):
    def __init__(self):
        super(ExplainableDeepfakeModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(1280, 256, num_layers=2, batch_first=True)
        self.attention = TemporalAttention(256)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
        self.frame_fc = nn.Linear(256, 2)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)
        features = features.view(b, s, -1)
        rnn_out, _ = self.rnn(features)
        context, attn_weights = self.attention(rnn_out)
        video_logits = self.fc(context)
        frame_logits = self.frame_fc(rnn_out)
        return video_logits, frame_logits, attn_weights

class GradCamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        bs, c, h, w = x.shape
        b = bs // 16
        x_5d = x.view(b, 16, c, h, w)
        _, frame_logits, _ = self.model(x_5d)
        return frame_logits.view(bs, 2)

# ==========================================
# 📂 데이터 파이프라인 (Dataset)
# ==========================================
class VideoWindowDataset(Dataset):
    def __init__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            f_224 = cv2.resize(frame, (224, 224))
            f_224 = cv2.cvtColor(f_224, cv2.COLOR_BGR2RGB)
            self.all_frames.append(np.float32(f_224) / 255.0)
        cap.release()
        
        self.all_frames = np.transpose(np.array(self.all_frames), (0, 3, 1, 2))
        self.total_count = len(self.all_frames)

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        # 현재 인덱스(idx)를 기준으로 16개 프레임 윈도우 생성
        win_indices = [max(0, min(self.total_count - 1, idx - 7 + j)) for j in range(16)]
        window = self.all_frames[win_indices]
        return torch.from_numpy(window), idx

# ==========================================
# 🚀 메인 렌더링 루프 (비동기 병렬 처리)
# ==========================================
def generate_ultimate_v5():
    # H200 최적화: 수치 안정성 및 에러 방지
    torch.backends.cudnn.enabled = False 
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()

    # 모델 준비
    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()
    base_model.rnn.train() 

    cam_model = GradCamWrapper(base_model)
    cam = GradCAM(model=cam_model, target_layers=[cam_model.model.backbone.features[-1]])

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    
    for f in files:
        print(f"\n🎬 {f} 분석 시작 (H200 최적화 파이프라인)")
        video_path = os.path.join(INPUT_DIR, f)
        
        # 1. 원본 비디오 정보 및 데이터셋 생성
        cap_info = cv2.VideoCapture(video_path)
        fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        
        dataset = VideoWindowDataset(video_path)
        # num_workers를 사용하여 CPU가 미리 데이터를 준비하게 함
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        out_path = os.path.join(OUTPUT_DIR, f"v5_ultimate_{f}")
        out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # 원본 프레임 읽기용
        cap_orig = cv2.VideoCapture(video_path)

        start_time = time.time()
        # 2. 병렬 루프 시작
        for windows, indices in tqdm(loader, desc="💎 Processing"):
            # windows shape: (B, 16, 3, 224, 224)
            input_tensor_5d = windows.to(device).float()
            B, S, C, H, W = input_tensor_5d.shape
            input_tensor_4d = input_tensor_5d.view(B * S, C, H, W)
            
            # Grad-CAM 연산
            targets = [ClassifierOutputTarget(1)] * (B * S)
            grayscale_cams = cam(input_tensor=input_tensor_4d, targets=targets)
            
            # 확률 계산
            with torch.no_grad():
                _, frame_logits, _ = base_model(input_tensor_5d)
                probs = torch.softmax(frame_logits[:, 7, :], dim=-1)[:, 1].cpu().numpy() * 100

            # 3. 결과 시각화 및 쓰기
            for b_idx in range(B):
                ret, orig_frame = cap_orig.read()
                if not ret: break

                current_cam = grayscale_cams[b_idx * 16 + 7]
                heatmap = cv2.resize(current_cam, (w, h), interpolation=cv2.INTER_CUBIC)
                
                base_img = np.float32(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)) / 255.0
                vis = show_cam_on_image(base_img, heatmap, use_rgb=True)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                # 가변 텍스트 크기
                p = probs[b_idx]
                color = (0, 0, 255) if p >= 70 else (0, 255, 0)
                cv2.putText(vis, f"FAKE: {p:.1f}%", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                out_video.write(vis)
                
            torch.cuda.empty_cache()

        cap_orig.release()
        out_video.release()
        print(f"✅ 완료: {time.time() - start_time:.1f}초")

if __name__ == "__main__":
    generate_ultimate_v5()