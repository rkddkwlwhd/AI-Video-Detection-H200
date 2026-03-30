import sys
import os

# --- [1] H200 서버 전용 충돌 방지 설정 ---
# 시스템 라이브러리(transformer_engine)와의 버전 충돌을 원천 차단합니다.
sys.modules['transformer_engine'] = None

import site
import json
import re
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

# [2] 경로 강제 주입
# 시스템에 깔린 구버전 라이브러리보다 사용자가 설치한 최신 라이브러리를 먼저 검색하게 합니다.
user_site_packages = site.getusersitepackages()
if user_site_packages not in sys.path:
    sys.path.insert(0, user_site_packages)

# CUDA 라이브러리 경로 강제 연결 (libcudart 오류 방지)
torch_lib_path = os.path.join(user_site_packages, "torch", "lib")
if os.path.exists(torch_lib_path):
    os.environ["LD_LIBRARY_PATH"] = torch_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# 경로 주입 후 MeloTTS 로드
try:
    from melo.api import TTS
except ImportError:
    print("❌ MeloTTS 라이브러리를 찾을 수 없습니다. 'pip install --user melotts'를 확인하세요.")
    sys.exit(1)

# --- [3] 경로 및 환경 설정 ---
# 현재 파일(src/generate_fake_melo.py) 위치를 기준으로 절대 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_PATH = os.path.join(BASE_DIR, "data_audio", "metadata.json")
OUT_DIR = os.path.join(BASE_DIR, "data_audio", "fake")
TARGET_SR = 16000  # Real 데이터(KsponSpeech) 규격에 맞춤

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- [4] 텍스트 정제 함수 (KeyError 방지) ---
def clean_text(text):
    """MeloTTS가 인식하지 못하는 특수 기호(+) 및 KsponSpeech 마커 제거"""
    # 1. KsponSpeech 특유의 잡음 마커 제거 (b/, n/ 등)
    text = re.sub(r'[a-z]/', '', text)
    
    # 2. KsponSpeech의 (단어)/(발음) 형태에서 발음 부분만 남기거나 괄호 제거
    # 여기서는 안전하게 괄호와 슬래시를 제거합니다.
    text = re.sub(r'[\(\)\/]', ' ', text)
    
    # 3. 한글, 영어, 숫자, 기본 문장부호(. , ? !)와 공백만 남기고 모두 제거 (에러 원인 '+' 포함)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!]', ' ', text)
    
    # 4. 연속된 공백을 하나로 합침
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- [5] 메인 실행 함수 ---
def generate_fake():
    print(f"🚀 MeloTTS 로딩 중... (H200 GPU 모드: {device})")
    
    try:
        # 한국어 모델 로드 (최초 실행 시 자동 다운로드)
        model = TTS(language='KR', device=device)
        speaker_ids = model.hps.data.spk2id
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return

    # 1. 메타데이터 로드
    if not os.path.exists(METADATA_PATH):
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {METADATA_PATH}")
        return

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"🎙️ Fake 데이터 생성 시작 (총 {len(metadata)}개)...")

    # 2. 루프 실행
    for item in tqdm(metadata):
        raw_text = item['transcript']
        file_name = item['file_name']
        save_path = os.path.join(OUT_DIR, file_name)

        # 텍스트 정제 (KeyError 방지)
        text = clean_text(raw_text)
        if not text:
            continue

        try:
            # 음성 합성 (가짜 목소리 생성)
            model.tts_to_file(text, speaker_ids['KR'], save_path, speed=1.0)

            # 3. 리샘플링 및 저장 (Real 데이터 규격인 16kHz로 통일)
            y, sr = librosa.load(save_path, sr=None)
            if sr != TARGET_SR:
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(save_path, y_resampled, TARGET_SR)
                
        except Exception as e:
            # 에러 발생 시 멈추지 않고 기록 후 다음 파일로 진행
            print(f"\n⚠️ 스킵됨 ({file_name}): {e}")
            continue

    print(f"\n✅ 모든 Fake 데이터 생성 완료! -> {OUT_DIR}")

if __name__ == "__main__":
    generate_fake()