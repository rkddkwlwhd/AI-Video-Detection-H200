import os
import subprocess
import json
from tqdm import tqdm

# [설정] 변환할 폴더 경로
TARGET_FOLDERS = ["../data/raw/real", "../data/raw/fake"]

def get_video_codec(file_path):
    """ffprobe를 사용하여 영상의 현재 코덱을 확인합니다."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name", "-of", "json", file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data['streams'][0]['codec_name']
    except Exception:
        return None

def convert_to_h264(input_path):
    """영상을 H.264 코덱으로 변환합니다."""
    temp_path = input_path + ".temp.mp4"
    # -c:v libx264: H.264 코덱 사용
    # -crf 23: 화질 유지 (숫자가 낮을수록 고화질, 18~28 권장)
    # -c:a copy: 오디오는 변환 없이 그대로 복사
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy", temp_path
    ]
    
    try:
        # 로그를 끄고 조용히 변환 진행
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 변환 성공 시: 원본을 .bak로 백업하고 새 파일을 원본 이름으로 교체
        bak_path = input_path + ".bak"
        os.rename(input_path, bak_path)
        os.rename(temp_path, input_path)
        return True
    except subprocess.CalledProcessError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def main():
    print("🚀 비디오 코덱 최적화 작업을 시작합니다 (H.264 표준화)")
    
    for folder in TARGET_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠️ 폴더를 찾을 수 없습니다: {folder}")
            continue
            
        print(f"\n📂 대상 폴더: {folder}")
        files = [f for f in os.listdir(folder) if f.endswith((".mp4", ".mkv", ".mov"))]
        
        for file in tqdm(files, desc="Processing"):
            file_path = os.path.join(folder, file)
            
            # 1. 현재 코덱 확인
            codec = get_video_codec(file_path)
            
            # 2. h264가 아니면 변환 (av1, vp9 등)
            if codec and codec != "h264":
                success = convert_to_h264(file_path)
                if not success:
                    print(f"❌ 변환 실패: {file}")
            # else: 이미 h264면 건너뜀

    print("\n✅ 모든 작업이 완료되었습니다!")
    print("💡 원본 AV1 영상들은 '.bak' 확장자로 백업되었습니다.")
    print("💡 이제 모든 영상은 OpenCV에서 문제 없이 읽힙니다.")

if __name__ == "__main__":
    main()