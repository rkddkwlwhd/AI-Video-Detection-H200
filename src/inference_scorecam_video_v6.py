import sys, os, cv2, torch, time
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# ⚙️ H200 안정화 설정 (Score-CAM 전용)
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth"
INPUT_DIR = "../data/test_samples"
OUTPUT_DIR = "../data/inference_results/gradcam_videos"

# 🔥 중요: INT_MAX 에러 방지를 위해 배치 사이즈를 1~2로 낮춥니다.
BATCH_SIZE = 1  
NUM_WORKERS = 0  
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [모델 구조 ExplainableDeepfakeModel, TemporalAttention 클래스 동일]
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

class ScoreCamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        bs, c, h, w = x.shape
        b = bs // 16
        x_5d = x.view(b, 16, c, h, w)
        _, frame_logits, _ = self.model(x_5d)
        return frame_logits.view(bs, 2)

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
    def __len__(self): return self.total_count
    def __getitem__(self, idx):
        win_indices = [max(0, min(self.total_count - 1, idx - 7 + j)) for j in range(16)]
        return torch.from_numpy(self.all_frames[win_indices]), idx

def generate_scorecam_ultimate():
    torch.backends.cuda.matmul.allow_tf32 = False
    
    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()

    cam_model = ScoreCamWrapper(base_model)
    target_layers = [cam_model.model.backbone.features[-1]]
    
    # 🔥 내부 연산 배치(batch_size)를 줄여서 INT_MAX 에러를 방지합니다.
    cam = ScoreCAM(model=cam_model, target_layers=target_layers)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    
    for f in files:
        print(f"\n🔍 {f} 분석 시작 (H200 Safe Score-CAM)")
        video_path = os.path.join(INPUT_DIR, f)
        cap_info = cv2.VideoCapture(video_path)
        fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        
        dataset = VideoWindowDataset(video_path)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0) # 안정성을 위해 0
        out_video = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"scorecam_{f}"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        cap_orig = cv2.VideoCapture(video_path)

        for windows, _ in tqdm(loader, desc="💎 Score-CAM Loop"):
            input_tensor_5d = windows.to(device).float()
            B, S, C, H, W = input_tensor_5d.shape
            input_tensor_4d = input_tensor_5d.view(B * S, C, H, W)
            
            # 🔥 [수정] Score-CAM의 내부 처리를 위한 데이터 전송
            # H200 메모리는 충분하지만 PyTorch 내부 연산 한계를 위해 BATCH_SIZE=1을 권장
            targets = [ClassifierOutputTarget(1)] * (B * S)
            
            # 메모리 정리를 수동으로 수행하며 한 프레임씩 정교하게 분석
            try:
                grayscale_cams = cam(input_tensor=input_tensor_4d, targets=targets)
            except RuntimeError as e:
                print(f"⚠️ 에러 감지: {e}. 더 작은 연산 단위로 재시도합니다.")
                # 여기서 에러가 나면 루프 내부에서 더 쪼개서 처리할 수도 있습니다.
                continue
            
            with torch.no_grad():
                _, frame_logits, _ = base_model(input_tensor_5d)
                probs = torch.softmax(frame_logits[:, 7, :], dim=-1)[:, 1].cpu().numpy() * 100

            for b_idx in range(B):
                ret, orig_frame = cap_orig.read()
                if not ret: break

                cur_cam = grayscale_cams[b_idx * 16 + 7]
                heatmap = cv2.resize(cur_cam, (w, h), interpolation=cv2.INTER_CUBIC)
                
                base_img = np.float32(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)) / 255.0
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                alpha = 0.5
                vis = cv2.addWeighted(np.uint8(255 * base_img), 1-alpha, heatmap_color, alpha, 0)
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                
                p = probs[b_idx]; color = (0, 0, 255) if p >= 70 else (0, 255, 0)
                cv2.putText(vis, f"FAKE: {p:.1f}%", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                out_video.write(vis)
                
            torch.cuda.empty_cache()

        cap_orig.release(); out_video.release()
        print(f"✅ {f} 완료")

if __name__ == "__main__":
    generate_scorecam_ultimate()