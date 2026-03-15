import sys, os, cv2, torch, time
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# ⚙️ H200 설정
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth"
INPUT_DIR = "../data/test_samples"
OUTPUT_DIR = "../data/inference_results/gradcam_videos"
BATCH_SIZE = 8  # Guided BP의 안정성을 위해 배치를 8로 낮춰 메모리 파편화 방지
NUM_WORKERS = 0  
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [모델 구조 및 Wrapper 클래스 동일]
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

# ==========================================
# 🚀 메인 실행부 (v6.7)
# ==========================================
def generate_guided_gradcam_ultimate():
    torch.backends.cudnn.enabled = False 
    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    for module in base_model.modules():
        if hasattr(module, 'inplace'): module.inplace = False
    base_model.eval()
    base_model.rnn.train()

    cam_model = GradCamWrapper(base_model)
    cam = GradCAM(model=cam_model, target_layers=[cam_model.model.backbone.features[-1]])
    gb_model = GuidedBackpropReLUModel(model=cam_model, device=device)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    
    for f in files:
        print(f"\n✨ {f} 분석 시작 (Guided Grad-CAM)")
        video_path = os.path.join(INPUT_DIR, f)
        cap_info = cv2.VideoCapture(video_path); fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_info.release()
        
        dataset = VideoWindowDataset(video_path)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        out_video = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"guided_{f}"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        cap_orig = cv2.VideoCapture(video_path)

        for windows, _ in tqdm(loader, desc="💎 Precision Analyzing"):
            input_tensor_5d = windows.to(device).float()
            B, S, C, H, W = input_tensor_5d.shape
            input_tensor_4d = input_tensor_5d.view(B * S, C, H, W)
            
            # Grad-CAM 추출
            grayscale_cams = cam(input_tensor=input_tensor_4d, targets=[ClassifierOutputTarget(1)] * (B * S))
            
            with torch.no_grad():
                _, frame_logits, _ = base_model(input_tensor_5d)
                probs = torch.softmax(frame_logits[:, 7, :], dim=-1)[:, 1].cpu().numpy() * 100

            # 🛠️ [핵심 수정] Guided Backprop을 루프 안에서 개별 혹은 소그룹으로 처리
            # 전체 배치를 한꺼번에 넣었을 때 발생하는 인덱싱 꼬임을 방지합니다.
            for b_idx in range(B):
                ret, orig_frame = cap_orig.read()
                if not ret: break

                # 현재 윈도우(16프레임)만 따로 추출해서 Guided Backprop 수행
                single_window_4d = input_tensor_5d[b_idx].view(16, C, H, W)
                single_gb = gb_model(single_window_4d, target_category=1)
                
                # Grad-CAM 결과 가져오기
                cur_cam = grayscale_cams[b_idx * 16 + 7] # (224, 224)
                
                # Guided Backprop에서 7번 프레임 추출 및 차원 보정
                # single_gb shape: (16, 3, 224, 224) 예상
                frame_gb = single_gb[7]
                if frame_gb.ndim == 4: frame_gb = frame_gb[0]
                if frame_gb.shape[0] == 3: frame_gb = np.transpose(frame_gb, (1, 2, 0))
                
                # 결합 및 시각화
                guided_gradcam = np.maximum(0, cur_cam[:, :, np.newaxis] * frame_gb)
                guided_gradcam -= np.min(guided_gradcam)
                guided_gradcam /= (np.max(guided_gradcam) + 1e-5)
                
                guided_res = cv2.resize(np.uint8(255 * guided_gradcam), (w, h), interpolation=cv2.INTER_CUBIC)
                alpha = 0.6
                result_img = cv2.addWeighted(orig_frame, 1-alpha, guided_res, alpha, 0)
                
                p = probs[b_idx]; color = (0, 0, 255) if p >= 70 else (0, 255, 0)
                cv2.putText(result_img, f"FAKE: {p:.1f}%", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                out_video.write(result_img)

        cap_orig.release(); out_video.release()
        print(f"✅ {f} 완료")

if __name__ == "__main__":
    generate_guided_gradcam_ultimate()