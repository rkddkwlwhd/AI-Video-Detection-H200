import os
import sys
import yt_dlp

# [설정] 오디오 저장 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DATA_DIR = os.path.join(BASE_DIR, "..", "data_audio", "raw")

def download_audio(link_file, category):
    save_path = os.path.join(AUDIO_DATA_DIR, category)
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(link_file):
        print(f"❌ 파일 없음: {link_file}")
        return

    with open(link_file, 'r') as f:
        links = [line.strip() for line in f.readlines() if line.strip()]

    print(f"🎙️ {category.upper()} 오디오 수집 시작 (총 {len(links)}개)")

    for idx, link in enumerate(links, start=1):
        numbering = f"{idx:04d}"
        
        ydl_opts = {
            # 최상의 오디오 품질 선택
            'format': 'bestaudio/best',
            'outtmpl': f'{save_path}/{numbering}_%(title)s.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            # FFmpeg을 사용하여 오디오만 추출 및 변환
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',      # 비손실 압축인 wav 권장
                'preferredquality': '192',
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"🚀 [{numbering}] 다운로드 중: {link}")
                ydl.download([link])
        except Exception as e:
            print(f"❌ [{numbering}] 실패: {e}")

if __name__ == "__main__":
    # 링크 파일 경로 (기존 구조 활용)
    real_audio_txt = os.path.join(BASE_DIR, "..", "links", "real_audio_links.txt")
    fake_audio_txt = os.path.join(BASE_DIR, "..", "links", "fake_audio_links.txt")

    if os.path.exists(real_audio_txt):
        download_audio(real_audio_txt, "real")
    if os.path.exists(fake_audio_txt):
        download_audio(fake_audio_txt, "fake")