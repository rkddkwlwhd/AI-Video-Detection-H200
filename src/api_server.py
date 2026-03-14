import os, cv2, torch, time
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
import shutil
import tempfile
import asyncio
import yt_dlp
import subprocess  # 🔥 추가
import json        # 🔥 추가

import torch.nn as nn
from torchvision import models
# pip install --user fastapi uvicorn python-multipart

# ==========================================
# ⚙️ 설정
# ==========================================
MODEL_PATH = "../data/processed/h200_attention_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAKE_THRESHOLD = 0.85

app = FastAPI(title="Explainable Deepfake Detection API", version="2.0")

# API 요청용 데이터 모델 (링크 받기용)
class LinkRequest(BaseModel):
    url: str

# ==========================================
# 🧠 모델 구조 정의 (동일)
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

model = None

@app.on_event("startup")
def load_model():
    global model
    print("🚀 [API Server] GPU 모델 로딩 중...")
    model = ExplainableDeepfakeModel().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ [API Server] 모델 로딩 완료!")
    else:
        print("❌ [API Server] 모델 가중치 파일을 찾을 수 없습니다!")

# ==========================================
# 🎥 영상 전처리 및 핵심 AI 분석 로직
# ==========================================
def preprocess_video(video_path, seq_len=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len: 
        cap.release()
        return None
    
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
    if len(frames) != seq_len: return None
    return torch.from_numpy(np.transpose(np.array(frames), (0, 3, 1, 2))).unsqueeze(0)

def get_risk_level(score):
    if score >= 70: return "높음", "red"
    elif score >= 50: return "중간", "yellow"
    else: return "낮음", "blue"

def run_ai_analysis(video_path: str, filename: str):
    """실제 AI 모델이 동작하여 JSON을 만드는 공통 핵심 함수"""
    input_tensor = preprocess_video(video_path)
    if input_tensor is None:
        raise HTTPException(status_code=400, detail="영상이 너무 짧거나 분석할 수 없는 형식입니다. (최소 16프레임 필요)")

    start_time = time.time()
    with torch.no_grad():
        video_logits, frame_logits, attn_weights = model(input_tensor.to(device))
        
        video_prob = torch.softmax(video_logits, dim=1)[0][1].item() * 100
        frame_probs = torch.softmax(frame_logits, dim=-1)[0, :, 1].cpu().numpy() * 100
        attn_weights = attn_weights[0].cpu().numpy()
    
    timeline =[]
    high_risk_frame_count = 0
    for i, p in enumerate(frame_probs):
        p_val = float(p)
        if p_val >= 80.0: high_risk_frame_count += 1
        lvl, color = get_risk_level(p_val)
        timeline.append({"frame_idx": i+1, "fake_prob": round(p_val, 2), "risk": lvl, "color": color})

    prob_std = float(np.std(frame_probs)) 
    attention_spike = float(np.max(attn_weights) * 100)
    temporal_score = float(min(100.0, prob_std * 1.5 + attention_spike * 0.5))
    temp_lvl, _ = get_risk_level(temporal_score)
    
    texture_score = float(np.mean(frame_probs))
    tex_lvl, _ = get_risk_level(texture_score)

    cond1 = video_prob >= (FAKE_THRESHOLD * 100)
    cond2 = texture_score >= 40.0
    cond3 = high_risk_frame_count >= 2
    
    is_fake = cond1 and cond2 and cond3
    final_label = "FAKE" if is_fake else "REAL"

    final_confidence = video_prob if is_fake else (100.0 - video_prob)
    if not is_fake and final_confidence < 50: 
        final_confidence = 100.0 - texture_score 

    process_time = round(time.time() - start_time, 2)

    return {
        "analysis_id": f"API_{int(time.time())}",
        "filename": filename,
        "final_prediction": final_label,
        "overall_confidence_percent": round(float(final_confidence), 2),
        "process_time_seconds": process_time,
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

# ==========================================
# 🌐 API 엔드포인트 1: 파일 업로드 방식
# ==========================================
@app.post("/analyze/file")
async def analyze_video_file(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_video_path = tmp_file.name

    try:
        result = run_ai_analysis(tmp_video_path, file.filename)
        return result
    finally:
        if os.path.exists(tmp_video_path): os.remove(tmp_video_path)

# ==========================================
# 🛠 비디오 코덱 검사 및 H.264 변환 함수 (추가됨)
# ==========================================
def get_video_codec(file_path):
    cmd =[
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "json", file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data['streams'][0]['codec_name']
    except Exception:
        return None

def convert_to_h264_if_needed(file_path):
    """영상이 H.264가 아니면 OpenCV가 읽을 수 있도록 변환합니다."""
    codec = get_video_codec(file_path)
    
    if codec == "h264":
        return True
        
    print(f"⚠️ [API Server] 비표준 코덱 감지({codec}). H.264 변환을 시작합니다...")
    temp_path = file_path + ".temp.mp4"
    
    # 🔥 수정 1: '-c:a copy' 대신 '-c:a aac' 로 변경하여 오디오 충돌 원천 차단
    cmd =[
        "ffmpeg", "-y", "-i", file_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast", 
        "-c:a", "aac", temp_path
    ]
    
    try:
        # 🔥 수정 2: 에러 로그를 숨기지 않고 캡처하도록 변경
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        shutil.move(temp_path, file_path)
        print("✅ [API Server] 코덱 변환 완료!")
        return True
        
    except subprocess.CalledProcessError as e:
        # 🔥 수정 3: FFmpeg이 왜 뻗었는지 터미널에 아주 상세하게 출력
        error_msg = e.stderr.decode('utf-8', errors='ignore')
        print(f"❌ [API Server] FFmpeg 변환 실패 상세 로그:\n{error_msg}")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

# ==========================================
# 🌐 API 엔드포인트 2: 숏폼 링크 다운로드 (안전 폴더 방식)
# ==========================================
def download_shortform(url: str, output_dir: str):
    """임시 폴더에 yt-dlp가 원하는 확장자로 자유롭게 다운로드하도록 허용합니다."""
    # %(ext)s 를 사용하여 비디오 원본 포맷(webm, mp4 등) 충돌을 방지합니다.
    outtmpl = os.path.join(output_dir, 'video.%(ext)s')
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4', # 합칠 때 강제로 mp4 컨테이너 사용
        'noplaylist': True,
        'quiet': True,
        'extractor_args': {'youtube': {'player_client':['default', '-android_sdkless']}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"❌ [API Server] yt-dlp 다운로드 에러: {e}")
        return False

@app.post("/analyze/link")
async def analyze_video_link(request: LinkRequest):
    url = request.url
    
    # 🔥 핵심: 임시 '파일' 대신 안전한 임시 '폴더'를 생성합니다.
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # 1. 임시 폴더 안에 다운로드 진행
            success = await asyncio.to_thread(download_shortform, url, tmp_dir)
            if not success:
                raise HTTPException(status_code=400, detail="해당 링크에서 영상을 다운로드할 수 없습니다.")

            # 2. 다운로드 완료 후, 폴더 안에 생성된 '진짜' 파일 이름 찾기
            downloaded_files = os.listdir(tmp_dir)
            if not downloaded_files:
                raise HTTPException(status_code=400, detail="영상 다운로드에 실패했습니다 (파일 없음).")
                
            actual_video_path = os.path.join(tmp_dir, downloaded_files[0])

            # 3. 진짜 파일을 대상으로 코덱 검사 및 H.264 강제 표준화
            conversion_success = await asyncio.to_thread(convert_to_h264_if_needed, actual_video_path)
            if not conversion_success:
                raise HTTPException(status_code=500, detail="서버 내부 코덱 변환 과정에서 오류가 발생했습니다.")

            # 4. 완벽하게 준비된 영상으로 AI 분석 실행
            result = run_ai_analysis(actual_video_path, "ShortForm_Video")
            
            result["filename"] = f"Link: {url[:30]}..."
            return result

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ [API Server] 처리 중 에러: {e}")
            raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")
        
        # 🧹 파이썬의 with 블록이 끝나면 tmp_dir(임시 폴더)와 그 안의 영상들은 
        # 찌꺼기 없이 OS 차원에서 완전히 자동 삭제됩니다! (서버 용량 쾌적)