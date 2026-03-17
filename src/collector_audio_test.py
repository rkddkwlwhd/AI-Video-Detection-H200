import os, sys, yt_dlp

# [설정] 테스트 데이터 전용 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RAW_DIR = os.path.join(BASE_DIR, "..", "data_audio", "test_raw")

def download_test_audio(link_file, category):
    save_path = os.path.join(TEST_RAW_DIR, category)
    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(link_file):
        print(f"❌ 테스트 링크 파일 없음: {link_file}")
        return

    with open(link_file, 'r') as f:
        links = [line.strip() for line in f.readlines() if line.strip()]

    print(f"🎙️ [TEST DATA] {category.upper()} 수집 시작 (총 {len(links)}개)")

    for idx, link in enumerate(links, start=1):
        numbering = f"test_{idx:03d}"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{save_path}/{numbering}_%(title)s.%(ext)s',
            'noplaylist': True,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
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
    # 테스트용 링크 파일 경로
    test_real_txt = os.path.join(BASE_DIR, "..", "links", "test_real_audio.txt")
    test_fake_txt = os.path.join(BASE_DIR, "..", "links", "test_fake_audio.txt")

    # 실행
    if os.path.exists(test_real_txt):
        download_test_audio(test_real_txt, "real")
    if os.path.exists(test_fake_txt):
        download_test_audio(test_fake_txt, "fake")