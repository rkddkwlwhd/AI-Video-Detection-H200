import subprocess
import sys
import site

# 1. 사용자 패키지 경로를 최우선으로 설정
sys.path.insert(0, site.getusersitepackages())

def initialize():
    # 꼬인 OpenCV를 해결할 4.8 버전과 필수 라이브러리들
    packages = [
        "yt-dlp", 
        "opencv-python-headless==4.8.0.74", 
        "numpy<2"
    ]
    
    print("🛠️ H200 서버 환경 최적화 시작...")
    
    try:
        # --upgrade: 시스템의 고장난 버전을 무시하고 사용자 공간에 새로 설치
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user", "--upgrade"
        ] + packages)
        
        print("\n✅ 모든 라이브러리 설치 완료!")
        import cv2, yt_dlp, numpy as np
        print(f"📍 OpenCV: {cv2.__version__} (Headless)")
        print(f"📍 NumPy: {np.__version__}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    initialize()