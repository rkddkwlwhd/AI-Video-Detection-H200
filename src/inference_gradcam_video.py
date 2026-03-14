import sys, os, cv2, torch, time
import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# Grad-CAM 라이브러리
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
# 🧠 모델 구조 (XAI와 동일)
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

# 🔥 Grad-CAM을 위해 시계열 모델을 2D 배치 모델처럼 속이는 래퍼(Wrapper)
class GradCamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # x: (16, 3, 224, 224) -> 가짜 배치 추가 (1, 16, 3, 224, 224)
        x = x.unsqueeze(0)
        _, frame_logits, _ = self.model(x)
        return frame_logits.squeeze(0) # (16, 2) 반환

# ==========================================
# 🎥 전처리 (원본 프레임 이미지도 함께 반환)
# ==========================================
def preprocess_for_gradcam(video_path, seq_len=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames < seq_len: return None, None, None
    
    frames_tensor =[]
    rgb_frames =[] # 시각화 베이스 이미지
    
    interval = total_frames // seq_len
    for i in range(seq_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frames.append(np.float32(rgb_frame) / 255.0)
        frames_tensor.append(rgb_frames[-1])
        
    cap.release()
    if len(frames_tensor) != seq_len: return None, None, None
    
    tensor = torch.from_numpy(np.transpose(np.array(frames_tensor), (0, 3, 1, 2)))
    return tensor, rgb_frames, fps # tensor: (16, 3, 224, 224)

# ==========================================
# 🚀 메인 실행부 (cuDNN RNN 버그 완벽 해결)
# ==========================================
def generate_gradcam_videos():
    # 🔥 핵심 해결책: GRU가 eval() 모드일 때 backward()를 수행하려면 cuDNN을 꺼야 합니다.
    torch.backends.cudnn.enabled = False 

    base_model = ExplainableDeepfakeModel().to(device)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    base_model.eval()

    # GradCAM 설정 (타겟 레이어: EfficientNet의 마지막 CNN 레이어)
    cam_model = GradCamWrapper(base_model).eval()
    target_layers =[cam_model.model.backbone.features[-1]]
    
    cam = GradCAM(model=cam_model, target_layers=target_layers)

    # 타겟: 인덱스 1 (FAKE)에 대한 그라디언트를 추적
    targets =[ClassifierOutputTarget(1) for _ in range(16)]

    files =[f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    print(f"\n🎥 영상 프레임별 Grad-CAM 렌더링 시작 ({len(files)}개)\n")

    for f in files:
        start_time = time.time()
        video_path = os.path.join(INPUT_DIR, f)
        
        tensor_16, rgb_frames, fps = preprocess_for_gradcam(video_path)
        if tensor_16 is None: continue

        # 1. 원본 모델에서 프레임별 FAKE 확률 예측
        with torch.no_grad():
            _, frame_logits, _ = base_model(tensor_16.unsqueeze(0).to(device))
            frame_probs = torch.softmax(frame_logits, dim=-1)[0, :, 1].cpu().numpy() * 100

        # 2. Grad-CAM 히트맵 추출 (이제 에러 없이 정상 계산됨!)
        grayscale_cams = cam(input_tensor=tensor_16, targets=targets)

        # 3. 비디오 렌더링 설정
        out_path = os.path.join(OUTPUT_DIR, f"heatmap_{f}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, 4.0, (224, 224))

        # 4. 프레임 합성 및 영상 쓰기
        for i in range(16):
            # 원본 이미지 위에 히트맵 덧씌우기
            visualization = show_cam_on_image(rgb_frames[i], grayscale_cams[i], use_rgb=True)
            
            # 우측 상단에 현재 프레임의 FAKE 확률 텍스트 그리기
            prob = frame_probs[i]
            color = (0, 0, 255) if prob >= 70 else (0, 255, 255) if prob >= 50 else (0, 255, 0)
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.putText(visualization, f"FAKE: {prob:.1f}%", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            out_video.write(visualization)
            
        out_video.release()
        elapsed = time.time() - start_time
        print(f"✅ {f} 렌더링 완료! (소요 시간: {elapsed:.2f}초) -> {out_path}")

if __name__ == "__main__":
    generate_gradcam_videos()