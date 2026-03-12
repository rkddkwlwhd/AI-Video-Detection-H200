import os

# ==========================================
BASE_DIR = "../data/raw"  # 영상이 저장된 기본 경로
EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv') # 체크할 영상 확장자
# ==========================================

def get_dir_size(path):
    """폴더의 전체 용량을 계산 (GB 단위)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def check_dataset_status():
    print("\n" + "="*40)
    print("🚀 데이터 수집 현황 리포트")
    print("="*40)
    
    total_count = 0
    
    # Real과 Fake 폴더 각각 순회
    for category in ['real', 'fake']:
        path = os.path.join(BASE_DIR, category)
        
        if not os.path.exists(path):
            print(f"⚠️  경고: '{category}' 폴더가 존재하지 않습니다!")
            continue
            
        # 확장자에 맞는 영상 파일 리스트 추출
        video_files = [f for f in os.listdir(path) if f.lower().endswith(EXTENSIONS)]
        count = len(video_files)
        size_gb = get_dir_size(path)
        
        total_count += count
        
        print(f"📂 [{category}]")
        print(f"   - 영상 개수: {count}개")
        print(f"   - 폴더 용량: {size_gb:.2f} GB")
        print("-" * 20)

    print(f"✅ 총 수집 영상: {total_count}개")
    print("="*40 + "\n")

if __name__ == "__main__":
    check_dataset_status()