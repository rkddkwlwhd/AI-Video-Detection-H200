import os
import subprocess
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# [설정]
TARGET_FOLDERS = ["../data/raw/real", "../data/raw/fake"]
MAX_WORKERS = os.cpu_count() // 2  # 전체 CPU 코어의 절반 사용 (H200 최적화)

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

def process_single_video(file_info):
    """개별 영상 하나를 검사하고 필요시 변환하는 함수 (병렬 실행됨)"""
    folder, file = file_info
    file_path = os.path.join(folder, file)
    
    codec = get_video_codec(file_path)
    if codec and codec != "h264":
        temp_path = file_path + ".temp.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-c:a", "copy", temp_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            bak_path = file_path + ".bak"
            os.rename(file_path, bak_path)
            os.rename(temp_path, file_path)
            return f"✅ 변환 완료: {file}"
        except:
            if os.path.exists(temp_path): os.remove(temp_path)
            return f"❌ 변환 실패: {file}"
    return None # 이미 h264인 경우

def main():
    print(f"🚀 H200 병렬 가공 시작 (프로세스 수: {MAX_WORKERS})")
    
    tasks = []
    for folder in TARGET_FOLDERS:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith((".mp4", ".mkv", ".mov"))]
            for f in files:
                tasks.append((folder, f))

    # 병렬 처리 실행
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # tqdm으로 전체 진행 상황 표시
        for res in tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc="Total Progress"):
            if res:
                results.append(res)

    print("\n" + "="*50)
    print(f"🏁 작업 완료! (총 {len(tasks)}개 중 {len(results)}개 변환 진행)")
    print("="*50)

if __name__ == "__main__":
    main()