import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# [1] H200 충돌 방지 및 경로 설정
sys.modules['transformer_engine'] = None

# 기존 모델 클래스 가져오기 (train_xai.py)
from train_xai import ExplainableDeepfakeModel

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data_audio", "processed", "deepvoice_attention_model.pth")
TEST_ROOT = os.path.join(BASE_DIR, "data_audio", "test_record") 
REPORT_DIR = os.path.join(BASE_DIR, "data_audio", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

SR = 16000
SEGMENTS = 16

# [2] 전처리 함수 (SciPy 대신 PyTorch STFT 사용 - 100% 동일한 결과 구현)
def preprocess_audio_torch(path):
    try:
        # 오디오 로드 (soundfile은 NumPy 2.x와 호환됨)
        y, sr = sf.read(path)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        
        # 4초 분량 조절
        target_len = SR * 4
        y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
        
        # 텐서로 변환
        y_t = torch.from_numpy(y).float()
        
        segment_len = len(y) // SEGMENTS
        spec_frames = []
        
        # Hann Window 설정 (SciPy 기본값과 동일)
        window = torch.hann_window(256)
        
        for i in range(SEGMENTS):
            chunk = y_t[i*segment_len : (i+1)*segment_len]
            
            # --- [SciPy Spectrogram을 Torch STFT로 완벽 대체] ---
            # n_fft=256, hop_length=128 (overlap 128)
            stft = torch.stft(chunk, n_fft=256, hop_length=128, win_length=256, 
                              window=window, return_complex=True)
            
            # Magnitude Squared (Power Spectrogram)
            spec = stft.abs().pow(2)
            
            # Log Scaled (dB)
            spec_db = 10 * torch.log10(spec + 1e-10)
            
            # --- [OpenCV Resize를 Torch Interpolate로 대체] ---
            # [Freq, Time] -> [1, 1, Freq, Time]
            spec_db = spec_db.unsqueeze(0).unsqueeze(0)
            spec_res = F.interpolate(spec_db, size=(224, 224), mode='bilinear', align_corners=False)
            
            # 정규화
            spec_res = spec_res.squeeze()
            s_min, s_max = spec_res.min(), spec_res.max()
            spec_norm = (spec_res - s_min) / (s_max - s_min + 1e-6)
            spec_frames.append(spec_norm)
            
        tensor_stack = torch.stack(spec_frames) # [16, 224, 224]
        # 3채널 복제 (EfficientNet 입력용)
        tensor_3ch = tensor_stack.unsqueeze(1).repeat(1, 3, 1, 1)
        return tensor_3ch.unsqueeze(0), y # [1, 16, 3, 224, 224]
        
    except Exception as e:
        print(f"\n❌ 분석 실패 ({os.path.basename(path)}): {e}")
        return None, None

# [3] 실행 메인 루프
def run_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 분석 시작 (NumPy {np.__version__} 환경)")
    
    model = ExplainableDeepfakeModel().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_files = []
    for root, dirs, files in os.walk(TEST_ROOT):
        for f in files:
            if f.lower().endswith('.wav'):
                test_files.append(os.path.join(root, f))

    print(f"🎙️ 총 {len(test_files)}개 파일 분석 리포트 생성 중...")

    for path in tqdm(test_files):
        f_name = os.path.basename(path)
        input_tensor, raw_audio = preprocess_audio_torch(path)
        if input_tensor is None: continue

        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            video_logits, frame_logits, attn_weights = model(input_tensor)
            
            probs = torch.softmax(video_logits, dim=1)[0]
            fake_prob = probs[1].item() * 100
            frame_probs = torch.softmax(frame_logits, dim=-1)[0, :, 1].cpu().numpy() * 100
            attn = attn_weights[0].cpu().numpy()

        # 시각화 리포트 생성
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, 4, len(raw_audio)), raw_audio, color='gray', alpha=0.5)
        decision = "[FAKE]" if fake_prob > 50 else "[REAL]"
        conf = fake_prob if fake_prob > 50 else 100 - fake_prob
        plt.title(f"File: {f_name}\nPrediction: {decision} ({conf:.1f}%)")
        
        plt.subplot(3, 1, 2)
        plt.bar(range(SEGMENTS), frame_probs, color=['red' if p > 50 else 'green' for p in frame_probs])
        plt.axhline(y=50, color='black', linestyle='--')
        plt.ylabel("Fake Prob %")
        
        plt.subplot(3, 1, 3)
        plt.plot(range(SEGMENTS), attn, marker='o', color='blue')
        plt.fill_between(range(SEGMENTS), attn, color='blue', alpha=0.2)
        plt.ylabel("Attention")
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, f"{f_name}_report.png"))
        plt.close()

    print(f"\n✅ 리포트 생성 완료! 위치: {REPORT_DIR}")

if __name__ == "__main__":
    run_report()