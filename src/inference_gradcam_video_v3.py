import sys, os, cv2, torch, time
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# ⚙️ 설정
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth"
INPUT_DIR = "../data/test_samples"
OUTPUT_DIR = "../data/inference_results/gradcam_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 모델 구조 (기존과 동일)
# ==========================================
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
        x = x.unsqueeze(0)
        _, frame_logits, _ = self.model(x)
        return frame_logits.squeeze(0)

# ==========================================
# 🎥 고해상도 렌더링 지원 전처리
# ==========================================
def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_224 = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # 분석용 224 사이즈만 메모리에 보관
        f_224 = cv2.resize(frame, (224, 224))
        f_224 = cv2.cvtColor(f_224, cv2.COLOR_BGR2RGB)
        frames_224.append(np.float32(f_224) / 255.0)
    
    cap.release()
    tensor_224 = torch.from_numpy(np.transpose(np.array(frames_224), (0, 3, 1, 2)))
    return tensor_224, fps, (width, height)

# ==========================================
# 🚀 메인 실행부
# ==========================================
def generate_high_res_gradcam_videos():
    torch.backends.cudnn.enabled = False 

    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()

    cam_model = GradCamWrapper(base_model).eval()
    target_layers = [cam_model.model.backbone.features[-1]]
    cam = GradCAM(model=cam_model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(1) for _ in range(16)]

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    print(f"\n✨ [고해상도 모드] Grad-CAM 렌더링 시작 ({len(files)}개)\n")

    for f in files:
        start_time = time.time()
        video_path = os.path.join(INPUT_DIR, f)
        
        # 1. 메타데이터 및 분석용 텐서(224) 가져오기
        tensor_224, fps, (orig_w, orig_h) = get_video_metadata(video_path)
        total_frames = len(tensor_224)
        
        # 2. 결과 저장을 위한 VideoWriter (원본 해상도 적용)
        out_path = os.path.join(OUTPUT_DIR, f"high_res_{f}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

        # 3. 원본 프레임을 다시 읽기 위한 캡처 객체
        cap_orig = cv2.VideoCapture(video_path)

        print(f"⏳ {f} 처리 중... ({orig_w}x{orig_h}, {fps} FPS)")
        
        for i in tqdm(range(total_frames), desc="Rendering", leave=False):
            ret, orig_frame = cap_orig.read()
            if not ret: break

            # 슬라이딩 윈도우 인덱스 계산
            window_indices = [max(0, min(total_frames - 1, i - 7 + j)) for j in range(16)]
            window_tensor = tensor_224[window_indices].to(device)

            # Grad-CAM 및 확률 추출
            with torch.no_grad():
                _, frame_logits, _ = base_model(window_tensor.unsqueeze(0))
                prob = torch.softmax(frame_logits, dim=-1)[0, 7, 1].item() * 100

            grayscale_cams = cam(input_tensor=window_tensor, targets=targets)
            current_heatmap = grayscale_cams[7] # 현재 프레임의 히트맵

            # 🔥 [핵심] 히트맵을 원본 해상도로 업스케일링
            heatmap_resized = cv2.resize(current_heatmap, (orig_w, orig_h))

            # 시각화 합성 (원본 이미지 기반)
            orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            base_img = np.float32(orig_frame_rgb) / 255.0
            
            visualization = show_cam_on_image(base_img, heatmap_resized, use_rgb=True)
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            # 고해상도에 맞춰 텍스트 크기 조절
            font_scale = orig_h / 500.0
            thickness = max(1, int(font_scale * 2))
            color = (0, 0, 255) if prob >= 70 else (0, 255, 255) if prob >= 50 else (0, 255, 0)
            
            cv2.putText(visualization, f"FAKE: {prob:.1f}%", (20, int(40 * font_scale)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

            out_video.write(visualization)
            
        cap_orig.release()
        out_video.release()
        print(f"✅ {f} 완료! ({time.time() - start_time:.1f}s)\n")

if __name__ == "__main__":
    generate_high_res_gradcam_videos()