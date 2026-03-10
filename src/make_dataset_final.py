import sys
import os

# 1. 사용자 로컬 경로 정의 및 라이브러리 충돌 방지 (기존 로직 유지)
USER_SITE = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

if 'cv2' in sys.modules:
    del sys.modules['cv2']

try:
    import cv2
    import numpy as np
    import h5py
    import random
    from tqdm import tqdm
    print(f"🚀 성공! OpenCV 경로: {cv2.__file__}")
except Exception as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    sys.exit(1)

# HDF5 파일 잠금 에러 방지
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def create_balanced_dataset(raw_path, out_path, max_samples_per_class=300, frames_per_video=16, img_size=224):
    categories = ['real', 'fake']
    all_tasks = []
    
    # [Step 1] 파일 목록 수집 및 클래스 균형 맞추기
    for label_idx, cat in enumerate(categories):
        cat_path = os.path.join(raw_path, cat)
        if not os.path.exists(cat_path):
            print(f"⚠️ 폴더 없음: {cat_path}")
            continue
            
        all_files = [os.path.join(cat_path, f) for f in os.listdir(cat_path) if f.endswith('.mp4')]
        num_to_sample = min(len(all_files), max_samples_per_class)
        sampled_files = random.sample(all_files, num_to_sample)
        
        for f in sampled_files:
            all_tasks.append((f, label_idx))
        print(f"✅ {cat.upper()}: {num_to_sample}개 수집 완료")

    # [Step 2] 전체 작업 랜덤 셔플 (학습/검증 분리 시 라벨 편향 방지)
    random.shuffle(all_tasks)
    total_requested = len(all_tasks)

    # [Step 3] HDF5 파일 생성
    with h5py.File(out_path, 'w') as hf:
        # 가변 크기 대응을 위해 maxshape 설정
        x_ds = hf.create_dataset('x', 
                                 shape=(total_requested, frames_per_video, 3, img_size, img_size), 
                                 maxshape=(total_requested, frames_per_video, 3, img_size, img_size),
                                 dtype=np.float32)
        y_ds = hf.create_dataset('y', 
                                 shape=(total_requested,), 
                                 maxshape=(total_requested,),
                                 dtype=np.int64)
        
        saved_count = 0
        print(f"🚀 총 {total_requested}개 영상 변환 시작...")
        
        for v_path, label in tqdm(all_tasks):
            cap = cv2.VideoCapture(v_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 프레임이 부족한 영상은 과감히 스킵
            if total_f < frames_per_video:
                cap.release()
                continue

            frames = []
            interval = total_f // frames_per_video
            success = True
            
            for i in range(frames_per_video):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                ret, frame = cap.read()
                if not ret: 
                    success = False
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (img_size, img_size))
                # 0~1 스케일로 정규화
                frames.append(frame.astype(np.float32) / 255.0)
            cap.release()

            if success and len(frames) == frames_per_video:
                # (T, H, W, C) -> (T, C, H, W) 변환
                video_data = np.transpose(np.array(frames), (0, 3, 1, 2))
                x_ds[saved_count] = video_data
                y_ds[saved_count] = label
                saved_count += 1

        # [Step 4] 핵심: 실제 저장된 개수만큼 데이터셋 크기 축소 (빈 공간 제거)
        if saved_count < total_requested:
            x_ds.resize((saved_count, frames_per_video, 3, img_size, img_size))
            y_ds.resize((saved_count,))
            print(f"\n⚠️ {total_requested - saved_count}개 영상이 프레임 에러로 제외되었습니다.")

    print(f"✨ 최종 데이터셋 생성 완료: {out_path} (총 {saved_count}개 저장)")

if __name__ == "__main__":
    # 데이터셋 경로 및 크기 설정
    RAW_DATA_DIR = "../data/raw"
    OUTPUT_H5 = "../data/processed/dataset_balanced.h5"
    
    # [여기서 크기를 조절하세요]
    MAX_SAMPLES_PER_CLASS = 300 
    
    create_balanced_dataset(RAW_DATA_DIR, OUTPUT_H5, max_samples_per_class=MAX_SAMPLES_PER_CLASS)