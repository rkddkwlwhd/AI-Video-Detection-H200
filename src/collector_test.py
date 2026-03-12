import os
import sys

# 1. 사용자 설치 경로(~/.local) 최우선 설정
user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import yt_dlp

# ==========================================
# ⚙️ 경로 설정 (실전 판별용)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 중요: 판별 전 '세수'를 위해 test_raw 폴더로 먼저 모읍니다.
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "test_raw")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")
# ==========================================

def download_videos(link_file, category="new_samples"):
    """
    link_file: 다운로드할 링크가 담긴 txt 파일 경로
    category: 저장될 하위 폴더 이름
    """
    save_path = os.path.join(DATA_DIR, category)
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(link_file):
        print(f"❌ 오류: '{link_file}' 파일을 찾을 수 없습니다.")
        return

    with open(link_file, 'r') as f:
        links = [line.strip() for line in f.readlines() if line.strip()]

    total_links = len(links)
    print(f"📦 실전 판별 대상: 총 {total_links}개의 링크를 확인했습니다.")

    for idx, link in enumerate(links, start=1):
        numbering = f"{idx:04d}" 
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{save_path}/test_{numbering}_%(title)s.%(ext)s', # 파일명에 test 접두사 추가
            'noplaylist': True,
            'quiet': True,
            'cookiefile': COOKIE_PATH,
            'js_runtime': 'deno',
            'extractor_args': {
                'youtube': {'player_client': ['default', '-android_sdkless']}
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"🚀 [{numbering}/{total_links:04d}] 신규 영상 수집 중: {link}")
                ydl.download([link])
        except Exception as e:
            print(f"❌ [{numbering}] 수집 실패: {e}")

if __name__ == "__main__":
    # 판별하고 싶은 영상 링크들을 담은 파일 경로
    # (미리 links 폴더에 test_links.txt를 만들어주세요!)
    test_txt = os.path.join(BASE_DIR, "..", "links", "test_links.txt")
    
    # 실행
    download_videos(test_txt, "unlabeled")