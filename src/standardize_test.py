import os
import subprocess
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==========================================
# ⚙️ 실전 판별용 경로 설정
# ==========================================
# 원본 수집 폴더 (collector_test.py가 저장한 곳)
INPUT_DIR = "../data/test_raw/unlabeled" 

# 가공 완료 폴더 (inference_production.py가 읽을 곳)
OUTPUT_DIR = "../data/test_samples"

# H200 최적화: 코어 절반 사용
MAX_WORKERS = os.cpu_count() // 2 
# ==========================================

def get_video_codec(file_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "json", file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data['streams'][0]['codec_name']
    except:
        return None

def process_test_video(file_name):
    """영상을 검사하고 규격에 맞춰 OUTPUT_DIR로 내보내는 함수"""
    src_path = os.path.join(INPUT_DIR, file_name)
    dst_path = os.path.join(OUTPUT_DIR, file_name)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    codec = get_video_codec(src_path)
    
    # 1. 이미 h264라면? 변환 없이 복사만 수행 (시간 단축)
    if codec == "h264":
        shutil.copy2(src_path, dst_path)
        return f"🚚 복사 완료 (이미 h264): {file_name}"
    
    # 2. h264가 아니라면? 변환 수행
    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "copy", dst_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return f"✅ 변환 완료: {file_name}"
    except Exception as e:
        return f"❌ 변환 실패: {file_name} ({e})"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"⚠️ 입력 폴더가 없습니다: {INPUT_DIR}")
        return

    print(f"🚀 실전 데이터 가공 시작 (H200 Parallel)")
    print(f"📂 경로: {INPUT_DIR} -> {OUTPUT_DIR}")
    
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi"))]
    
    if not files:
        print("📭 가공할 영상이 없습니다.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for res in tqdm(executor.map(process_test_video, files), total=len(files), desc="Standardizing"):
            results.append(res)

    print("\n" + "="*50)
    print(f"🏁 가공 완료! (총 {len(files)}개 처리 완료)")
    print(f"📍 위치: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)

if __name__ == "__main__":
    main()