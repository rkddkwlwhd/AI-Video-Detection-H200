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
# 🧠 모델 구조 (XAI 동일)
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
# 🎥 전처리: 모든 프레임 추출 (슬라이딩 윈도우용)
# ==========================================
def preprocess_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_tensor = []
    rgb_frames =[]

    while True:
        ret, frame = cap.read()
        if not ret: break
        # 모든 프레임을 224x224로 변환하여 저장
        frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb_frame)
        frames_tensor.append(np.float32(rgb_frame) / 255.0)

    cap.release()
    
    if len(frames_tensor) == 0:
        return None, None, None

    # (N, 3, 224, 224) 텐서로 변환
    tensor_all = torch.from_numpy(np.transpose(np.array(frames_tensor), (0, 3, 1, 2)))
    return tensor_all, rgb_frames, fps

# ==========================================
# 🚀 메인 실행부 (슬라이딩 윈도우 렌더링)
# ==========================================
def generate_smooth_gradcam_videos():
    # cuDNN RNN 버그 방지
    torch.backends.cudnn.enabled = False 

    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()

    cam_model = GradCamWrapper(base_model).eval()
    target_layers =[cam_model.model.backbone.features[-1]]
    cam = GradCAM(model=cam_model, target_layers=target_layers)

    targets =[ClassifierOutputTarget(1) for _ in range(16)]

    files =[f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    print(f"\n🎬 [초당 30프레임] 부드러운 고품질 Grad-CAM 렌더링 시작 ({len(files)}개)\n")

    for f in files:
        start_time = time.time()
        video_path = os.path.join(INPUT_DIR, f)
        
        # 영상의 "모든" 프레임 가져오기
        tensor_all, rgb_frames, fps = preprocess_all_frames(video_path)
        if tensor_all is None: continue

        total_frames = len(tensor_all)
        out_path = os.path.join(OUTPUT_DIR, f"smooth_heatmap_{f}")
        
        # 1초에 4장이 아니라, 원본 영상의 부드러운 FPS 그대로(예: 30) 세팅
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (224, 224))

        print(f"⏳ {f} 처리 중... (총 {total_frames} 프레임, {fps} FPS)")
        
        # 프레임 하나하나를 돌면서 슬라이딩 윈도우로 AI 분석
        for i in tqdm(range(total_frames), desc="Rendering", leave=False):
            
            # 🔥 [핵심] 현재 프레임(i)을 중심으로 앞뒤 16장짜리 윈도우 생성
            # 범위를 벗어나는 첫/끝부분은 가장자리 프레임으로 패딩 처리
            window_indices =[max(0, min(total_frames - 1, i - 7 + j)) for j in range(16)]
            window_tensor = tensor_all[window_indices].to(device) # (16, 3, 224, 224)

            # 1. 모델 예측 (윈도우 문맥 파악)
            with torch.no_grad():
                _, frame_logits, _ = base_model(window_tensor.unsqueeze(0))
                frame_probs = torch.softmax(frame_logits, dim=-1)[0, :, 1].cpu().numpy() * 100

            # 2. Grad-CAM 추출
            grayscale_cams = cam(input_tensor=window_tensor, targets=targets)

            # 3. 16장 중에서 "정중앙(7번째 인덱스)" 값이 바로 현재 프레임(i)의 결과
            center_idx = 7
            current_heatmap = grayscale_cams[center_idx]
            current_prob = frame_probs[center_idx]

            # 4. 시각화 합성
            base_img = np.float32(rgb_frames[i]) / 255.0
            visualization = show_cam_on_image(base_img, current_heatmap, use_rgb=True)
            
            # 텍스트 오버레이
            color = (0, 0, 255) if current_prob >= 70 else (0, 255, 255) if current_prob >= 50 else (0, 255, 0)
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.putText(visualization, f"FAKE: {current_prob:.1f}%", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            out_video.write(visualization)
            
        out_video.release()
        elapsed = time.time() - start_time
        print(f"✅ {f} 렌더링 완료! (소요 시간: {elapsed:.2f}초) -> {out_path}\n")

if __name__ == "__main__":
    generate_smooth_gradcam_videos()