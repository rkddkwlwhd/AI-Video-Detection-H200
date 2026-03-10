import os
import sys

# 1. 사용자 설치 경로(~/.local) 최우선 설정
user_site = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import yt_dlp

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")

# [전략: 기존 코드와 동일]

def download_videos(link_file, category):
    save_path = os.path.join(DATA_DIR, category)
    os.makedirs(save_path, exist_ok=True)
    
    with open(link_file, 'r') as f:
        links = [line.strip() for line in f.readlines() if line.strip()]

    total_links = len(links)
    print(f"📦 {category} 카테고리: 총 {total_links}개의 링크를 확인했습니다.")

    for idx, link in enumerate(links, start=1):
        # 숫자를 4자리(0001, 0002...)로 변경
        numbering = f"{idx:04d}" 
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            # 파일명 맨 앞에 4자리 번호 삽입
            'outtmpl': f'{save_path}/{numbering}_%(title)s.%(ext)s',
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
                # 진행 상황 출력 (예: [0001/1200])
                print(f"🚀 [{numbering}/{total_links:04d}] 다운로드 중: {link}")
                ydl.download([link])
        except Exception as e:
            # 실패하더라도 어떤 번호의 링크가 실패했는지 정확히 출력
            print(f"❌ [{numbering}] 다운로드 실패: {e}")

# [후략: 기존 실행 로직 동일]

if __name__ == "__main__":
    # 실행 로직은 동일
    real_txt = os.path.join(BASE_DIR, "..", "links", "real_links.txt")
    if os.path.exists(real_txt):
        download_videos(real_txt, "real")

    fake_txt = os.path.join(BASE_DIR, "..", "links", "fake_links.txt")
    if os.path.exists(fake_txt):
        download_videos(fake_txt, "fake")