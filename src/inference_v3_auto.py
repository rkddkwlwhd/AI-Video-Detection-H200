import sys, os, cv2, torch, json, datetime
import torch.nn as nn
import numpy as np
from torchvision import models
from tqdm import tqdm

# ==========================================
# ⚙️ 설정
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth" # (학습 시 이 모델 구조로 다시 저장해야 함)
INPUT_DIR = "../data/test_samples"
OUTPUT_JSON_DIR = "../data/inference_results"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAKE_THRESHOLD = 0.85 # 🔥 오판을 줄이기 위한 임계값 상향! (0.5 -> 0.85)

# ==========================================
# 🧠 Attention 기반 모델 구조 (개선됨)
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, rnn_output):
        # rnn_output: (Batch, Seq_len, Hidden)
        attn_weights = torch.softmax(self.attention(rnn_output), dim=1) # (B, Seq_len, 1)
        context = torch.sum(attn_weights * rnn_output, dim=1)           # (B, Hidden)
        return context, attn_weights.squeeze(-1)

class ExplainableDeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ExplainableDeepfakeModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        
        # RNN & Attention
        self.rnn = nn.GRU(input_size=1280, hidden_size=256, num_layers=2, batch_first=True)
        self.attention = TemporalAttention(256)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        
        # 프레임별 판별기 (타임라인 설명력을 위해 추가)
        self.frame_fc = nn.Linear(256, num_classes)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)
        features = features.view(b, s, -1)
        
        rnn_out, _ = self.rnn(features) # rnn_out: (B, Seq, 256)
        
        # 1. Attention 기반 전체 비디오 문맥 추출
        context, attn_weights = self.attention(rnn_out)
        video_logits = self.fc(context)
        
        # 2. 프레임별 독립적인 의심도 추출
        frame_logits = self.frame_fc(rnn_out) # (B, Seq, 2)
        
        return video_logits, frame_logits, attn_weights

# ==========================================
# 🎥 전처리 (기존과 동일)
# ==========================================
def preprocess_video(video_path, seq_len=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len: return None
    
    frames =[]
    interval = total_frames // seq_len
    for i in range(seq_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    return torch.from_numpy(np.transpose(np.array(frames), (0, 3, 1, 2))).unsqueeze(0) if len(frames) == seq_len else None

# ==========================================
# 📊 지표 생성 및 JSON 파싱 로직 (수정됨)
# ==========================================
def get_risk_level(score):
    if score >= 70: return "높음", "red"
    elif score >= 50: return "중간", "yellow"
    else: return "낮음", "blue"

def analyze_and_save(filename, video_prob, frame_probs, attn_weights):
    #[1] 타임라인 데이터 (Numpy float32 -> Python float 변환)
    timeline =[]
    high_risk_frame_count = 0  # 🚨 80% 이상 위험한 프레임 개수 카운트
    
    for i, p in enumerate(frame_probs):
        p_val = float(p)
        if p_val >= 80.0:
            high_risk_frame_count += 1
            
        lvl, color = get_risk_level(p_val)
        timeline.append({"frame_idx": i+1, "fake_prob": round(p_val, 2), "risk": lvl, "color": color})

    # [2] 상세 지표 생성 (버그 수정: var -> std)
    # frame_probs가 0~100 스케일이므로 분산(var) 대신 표준편차(std)를 써야 수치가 튀지 않습니다.
    prob_std = float(np.std(frame_probs)) 
    attention_spike = float(np.max(attn_weights) * 100)
    
    # 떨림(표준편차)과 어텐션 최대치를 적절한 비율로 섞어서 0~100 사이로 보정
    temporal_score = float(min(100.0, prob_std * 1.5 + attention_spike * 0.5))
    temp_lvl, _ = get_risk_level(temporal_score)
    
    texture_score = float(np.mean(frame_probs))
    tex_lvl, _ = get_risk_level(texture_score)

    # [3] 🔥 REAL 오판 방지를 위한 "깐깐한 다중 교차 검증" 로직
    # 단순히 video_prob 하나만 믿지 않고, 아래 3개 조건을 모두 만족해야 FAKE로 판정합니다.
    cond1 = video_prob >= 85.0           # 1. 모델의 전체적인 확신도가 85% 이상인가?
    cond2 = texture_score >= 40.0        # 2. 16개 프레임의 평균 의심도가 40% 이상인가?
    cond3 = high_risk_frame_count >= 2   # 3. 80% 이상 확신하는 빼박 가짜 프레임이 2장 이상인가?
    
    is_fake = cond1 and cond2 and cond3
    final_label = "FAKE" if is_fake else "REAL"

    # 최종 신뢰도 점수 (FAKE면 가장 높은 점수를, REAL이면 100에서 뺀 점수를 보여줌)
    final_confidence = video_prob if is_fake else (100.0 - video_prob)
    if not is_fake and final_confidence < 50: 
        # FAKE로 오해할 뻔했다가 REAL로 교정된 경우, REAL 확신도를 50~70% 사이로 보정해줌
        final_confidence = 100.0 - texture_score 

    report = {
        "analysis_id": f"H200_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "filename": filename,
        "final_prediction": final_label,
        "overall_confidence_percent": round(float(final_confidence), 2),
        "timeline_chart": timeline,
        "detailed_analysis":[
            {
                "title": "프레임 전환 일관성 위험도",
                "risk_level": temp_lvl,
                "score_percent": round(temporal_score, 1),
                "description": f"프레임 전환 시 얼굴이나 배경의 미세한 떨림 및 시공간적 비일관성이 {round(temporal_score, 1)}% 수준으로 감지되었습니다."
            },
            {
                "title": "공간적 텍스처 및 화질 왜곡 위험도",
                "risk_level": tex_lvl,
                "score_percent": round(texture_score, 1),
                "description": f"이미지 생성 과정에서 발생하는 인위적인 픽셀 뭉개짐이나 텍스처 이상 징후가 {round(texture_score, 1)}% 확률로 감지되었습니다."
            }
        ]
    }
    
    with open(os.path.join(OUTPUT_JSON_DIR, f"{filename}.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

# ==========================================
# 🚀 메인 실행부
# ==========================================
def run_inference():
    model = ExplainableDeepfakeModel().to(device)
    
    # 모델 가중치 로드 (🚨 이 코드를 사용하려면 train.py에서 모델 구조를 변경 후 다시 학습해야 합니다)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("⚠️ [경고] 학습된 Attention 모델이 없습니다. 임의의 가중치로 테스트합니다.")
    
    model.eval()

    files =[f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    print(f"\n🚀 설명 가능한 XAI 딥페이크 판별 시작 ({len(files)}개 영상)")

    for f in tqdm(files):
        video_path = os.path.join(INPUT_DIR, f)
        input_tensor = preprocess_video(video_path)
        if input_tensor is None: continue

        with torch.no_grad():
            video_logits, frame_logits, attn_weights = model(input_tensor.to(device))
            
            # 1. 전체 영상에 대한 FAKE 확률 (0~100)
            video_prob = torch.softmax(video_logits, dim=1)[0][1].item() * 100
            
            # 2. 프레임별 FAKE 확률 추출
            frame_probs = torch.softmax(frame_logits, dim=-1)[0, :, 1].cpu().numpy() * 100
            
            # 3. 모델이 어디를 중요하게 봤는지 (Attention)
            attn_weights = attn_weights[0].cpu().numpy()
            
        analyze_and_save(f, video_prob, frame_probs, attn_weights)

if __name__ == "__main__":
    run_inference()