import os, sys, torch, cv2, time
import numpy as np
import soundfile as sf
from scipy import signal
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# [1] 터미널 컬러 설정
class Color:
    GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'
    BLUE = '\033[94m'; BOLD = '\033[1m'; END = '\033[0m'; CYAN = '\033[96m'

# [2] 모델 구조 정의 (ExplainableDeepvoiceModel)
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

class ExplainableDeepvoiceModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ExplainableDeepvoiceModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.rnn = nn.GRU(1280, 256, num_layers=2, batch_first=True)
        self.attention = TemporalAttention(256)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
        self.frame_fc = nn.Linear(256, num_classes)

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

# [3] Robust 오디오 전처리 (Scipy 방식)
SR = 16000
SEGMENTS = 16

def preprocess_audio(v_path):
    try:
        y, sr = sf.read(v_path)
        if sr != SR: return None
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        target_len = SR * 4
        y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
        
        segment_len = len(y) // SEGMENTS
        spec_frames = []
        for i in range(SEGMENTS):
            chunk = y[i*segment_len : (i+1)*segment_len]
            _, _, Sxx = signal.spectrogram(chunk, fs=SR, nperseg=256, noverlap=128)
            spec_db = 10 * np.log10(Sxx + 1e-10)
            spec_resized = cv2.resize(spec_db, (224, 224))
            spec_norm = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min() + 1e-6)
            spec_frames.append(spec_norm)
            
        tensor = np.stack(spec_frames)
        tensor_3ch = np.repeat(tensor[:, np.newaxis, :, :], 3, axis=1)
        return torch.from_numpy(tensor_3ch).unsqueeze(0).float()
    except:
        return None

# [4] 폴더 순회 및 성능 분석 핵심 함수
def run_auto_evaluation(base_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = ExplainableDeepvoiceModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    categories = ['real', 'fake']
    overall_results = []
    
    print(f"\n{Color.BOLD}{Color.CYAN}🚀 [H200 Audio Analysis] 딥보이스 자동 전수 검사 시작...{Color.END}")

    for cat in categories:
        folder_path = os.path.join(base_dir, cat)
        if not os.path.exists(folder_path):
            continue
            
        expected_label = cat.upper() # "REAL" 또는 "FAKE"
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        correct_count = 0
        cat_results = []

        print(f"\n📂 {Color.BOLD}분석 폴더: {cat} (정답: {expected_label}){Color.END}")
        print(f"{'파일명':<40} | {'판정':<6} | {'신뢰도':<8} | {'결과'}")
        print("-" * 75)

        for f in tqdm(files, desc=f"Processing {cat}", leave=False):
            audio_path = os.path.join(folder_path, f)
            input_tensor = preprocess_audio(audio_path)
            
            if input_tensor is None:
                continue

            with torch.no_grad():
                video_logits, _, _ = model(input_tensor.to(device))
                prob = torch.softmax(video_logits, dim=1)[0][1].item() * 100
                
            pred_label = "FAKE" if prob > 50 else "REAL"
            confidence = prob if pred_label == "FAKE" else (100 - prob)
            
            is_correct = (pred_label == expected_label)
            if is_correct: correct_count += 1
            
            res_str = f"{Color.GREEN}PASS{Color.END}" if is_correct else f"{Color.RED}FAIL{Color.END}"
            conf_color = Color.YELLOW if confidence < 80 else "" # 신뢰도 낮으면 노란색 표시
            
            print(f"{f[:38]:<40} | {pred_label:<6} | {conf_color}{confidence:>6.2f}%{Color.END} | {res_str}")
            
            cat_results.append({
                'file': f, 'pred': pred_label, 'actual': expected_label, 
                'conf': confidence, 'is_correct': is_correct
            })
        
        acc = (correct_count / len(files)) * 100 if files else 0
        overall_results.extend(cat_results)
        print(f"\n📊 {Color.BOLD}{cat.upper()} 정확도: {acc:.2f}% ({correct_count}/{len(files)}){Color.END}")

    # [5] 최종 종합 리포트 출력
    total = len(overall_results)
    total_correct = sum(1 for r in overall_results if r['is_correct'])
    total_acc = (total_correct / total) * 100 if total > 0 else 0
    
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*30} 📋 최종 종합 리포트 {'='*30}{Color.END}")
    print(f"✅ 전체 정확도 : {total_acc:.2f}% ({total_correct}/{total})")
    
    # 오답 리스트 (Hard Cases)
    wrong_cases = [r for r in overall_results if not r['is_correct']]
    if wrong_cases:
        print(f"\n{Color.RED}{Color.BOLD}❌ [오답 노트] 모델이 틀린 파일 ({len(wrong_cases)}개):{Color.END}")
        for r in wrong_cases:
            print(f"  - {r['file']:<40} (판정: {r['pred']} / 정답: {r['actual']} / 신뢰도: {r['conf']:.2f}%)")
            
    # 맞았지만 불안한 리스트 (Low Confidence)
    low_cases = [r for r in overall_results if r['is_correct'] and r['conf'] < 75]
    if low_cases:
        print(f"\n{Color.YELLOW}{Color.BOLD}⚠️ [정밀 검토 권장] 맞았으나 신뢰도가 낮은 파일 ({len(low_cases)}개):{Color.END}")
        for r in low_cases:
            print(f"  - {r['file']:<40} (신뢰도: {r['conf']:.2f}%)")

    print(f"\n{Color.BOLD}{Color.CYAN}{'='*75}{Color.END}\n")

if __name__ == "__main__":
    BASE_AUDIO_DIR = "../data_audio/standardized"
    MODEL_PATH = "../data_audio/processed/deepvoice_attention_model.pth"
    
    if os.path.exists(MODEL_PATH):
        run_auto_evaluation(BASE_AUDIO_DIR, MODEL_PATH)
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")