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

# ... (중간의 설정, 모델 클래스, load_model, run_ai_analysis 함수는 기존과 100% 동일하게 유지합니다) ...

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
    
    # 이미 h264면 변환 없이 패스 (API 속도 최적화)
    if codec == "h264":
        return True
        
    print(f"⚠️ [API Server] 비표준 코덱 감지({codec}). H.264 변환을 시작합니다...")
    temp_path = file_path + ".temp.mp4"
    
    # API 서버이므로 변환 속도가 생명! preset을 ultrafast로 설정
    cmd =[
        "ffmpeg", "-y", "-i", file_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast", 
        "-c:a", "copy", temp_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        shutil.move(temp_path, file_path) # 원본 덮어쓰기
        print("✅ [API Server] 코덱 변환 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [API Server] FFmpeg 변환 에러: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

# ==========================================
# 🌐 API 엔드포인트 2: 숏폼 링크 입력 방식 (수정됨)
# ==========================================
def download_shortform(url: str, output_path: str):
    """yt-dlp를 이용해 숏폼 다운로드"""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'noplaylist': True,
        'quiet': True,
        'extractor_args': {'youtube': {'player_client': ['default', '-android_sdkless']}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True
    except Exception as e:
        print(f"다운로드 에러: {e}")
        return False

@app.post("/analyze/link")
async def analyze_video_link(request: LinkRequest):
    url = request.url
    tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)

    try:
        # 1. 숏폼 다운로드
        success = await asyncio.to_thread(download_shortform, url, tmp_video_path)
        if not success or not os.path.exists(tmp_video_path):
            raise HTTPException(status_code=400, detail="해당 링크에서 영상을 다운로드할 수 없습니다.")

        # 🔥 2. 코덱 검사 및 H.264 강제 표준화 (추가된 부분)
        conversion_success = await asyncio.to_thread(convert_to_h264_if_needed, tmp_video_path)
        if not conversion_success:
            raise HTTPException(status_code=500, detail="서버 내부 코덱 변환 과정에서 오류가 발생했습니다.")

        # 3. AI 분석 실행
        filename = "ShortForm_Video.mp4"
        result = run_ai_analysis(tmp_video_path, filename)
        
        result["filename"] = f"Link: {url[:30]}..."
        return result

    finally:
        # 4. 임시 파일 삭제
        if os.path.exists(tmp_video_path): 
            os.remove(tmp_video_path)