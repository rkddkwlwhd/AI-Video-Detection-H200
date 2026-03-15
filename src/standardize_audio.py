import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# [설정]
RAW_AUDIO_DIR = "../data_audio/raw"
STZ_AUDIO_DIR = "../data_audio/standardized"
MAX_WORKERS = os.cpu_count() 

# 오디오 표준 규격 (탐지 모델의 일반적인 요구사항)
SAMPLE_RATE = "16000"  # 16kHz (보이스 분석 최적)
CHANNELS = "1"         # Mono (딥보이스 분석 시 스테레오 정보는 노이즈가 됨)

def standardize_single_audio(file_info):
    category, file_name = file_info
    src_path = os.path.join(RAW_AUDIO_DIR, category, file_name)
    dst_dir = os.path.join(STZ_AUDIO_DIR, category)
    os.makedirs(dst_dir, exist_ok=True)
    
    # 확장자를 .wav로 통일
    base_name = os.path.splitext(file_name)[0]
    dst_path = os.path.join(dst_dir, f"{base_name}.wav")

    # FFmpeg 명령어: 샘플 레이트 변경(-ar), 채널 변경(-ac), 코덱(pcm_s16le)
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", SAMPLE_RATE,
        "-ac", CHANNELS,
        "-c:a", "pcm_s16le", 
        dst_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return f"✅ {file_name} 표준화 완료"
    except Exception as e:
        return f"❌ {file_name} 에러: {e}"

def main():
    print(f"🛠️ H200 오디오 병렬 표준화 시작 (규격: {SAMPLE_RATE}Hz, Mono)")
    
    tasks = []
    for cat in ['real', 'fake']:
        cat_path = os.path.join(RAW_AUDIO_DIR, cat)
        if os.path.exists(cat_path):
            files = [f for f in os.listdir(cat_path) if f.endswith((".wav", ".mp3", ".m4a"))]
            for f in files:
                tasks.append((cat, f))

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for res in tqdm(executor.map(standardize_single_audio, tasks), total=len(tasks), desc="Processing"):
            results.append(res)

    print(f"\n🏁 완료: {len(results)}개 오디오 가공됨.")
    print(f"📍 위치: {os.path.abspath(STZ_AUDIO_DIR)}")

if __name__ == "__main__":
    main()