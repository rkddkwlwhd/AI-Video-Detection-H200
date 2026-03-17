import os, subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# [설정]
RAW_DIR = "../data_audio/test_raw"
DST_DIR = "../data_audio/test_standardized" # 여기가 최종 inference_audio_auto_report.py가 읽을 곳
SR = "16000"
CH = "1"

def standardize_worker(file_info):
    cat, name = file_info
    src = os.path.join(RAW_DIR, cat, name)
    out_dir = os.path.join(DST_DIR, cat)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".wav")
    
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", SR, "-ac", CH, "-c:a", "pcm_s16le", out_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False

if __name__ == "__main__":
    print("🛠️ [TEST DATA] 표준화 가공 시작...")
    tasks = []
    for cat in ['real', 'fake']:
        path = os.path.join(RAW_DIR, cat)
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.wav', '.mp3', '.m4a'))]
            for f in files: tasks.append((cat, f))
            
    with ProcessPoolExecutor() as exe:
        list(tqdm(exe.map(standardize_worker, tasks), total=len(tasks)))
    print(f"✅ 가공 완료: {DST_DIR}")