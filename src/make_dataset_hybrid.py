import sys, os, cv2, h5py, random
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# [1] 경로 및 환경 설정 (OpenCV 충돌 방지 로직 유지)
USER_SITE = os.path.expanduser("~/.local/lib/python3.10/site-packages")
if USER_SITE not in sys.path: sys.path.insert(0, USER_SITE)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# [2] 개별 영상 처리 함수 (병렬/순차 공통 사용)
def extract_frames_worker(task):
    v_path, label = task
    frames_per_video = 16
    img_size = 224
    
    cap = cv2.VideoCapture(v_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_f < frames_per_video:
        cap.release()
        return None

    frames = []
    interval = total_f // frames_per_video
    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    if len(frames) == frames_per_video:
        return (np.transpose(np.array(frames), (0, 3, 1, 2)), label)
    return None

# [3] 메인 데이터셋 생성 함수
def create_dataset(raw_path, out_path, max_samples=300, parallel=True):
    # Step 1: 태스크 수집 및 셔플
    all_tasks = []
    for label, cat in enumerate(['real', 'fake']):
        cat_dir = os.path.join(raw_path, cat)
        if not os.path.exists(cat_dir): continue
        files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.mp4')]
        sampled = random.sample(files, min(len(files), max_samples))
        for f in sampled:
            all_tasks.append((f, label))
    
    random.shuffle(all_tasks)
    total_requested = len(all_tasks)
    processed_results = []

    # Step 2: 처리 모드 분기 (병렬 vs 순차)
    if parallel:
        print(f"🚀 [병렬 모드] H200의 멀티코어를 가동합니다. (대상: {total_requested}개)")
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(extract_frames_worker, all_tasks), total=total_requested))
            processed_results = [r for r in results if r is not None]
    else:
        print(f"🐢 [순차 모드] 한 장씩 꼼꼼하게 처리합니다. (대상: {total_requested}개)")
        for task in tqdm(all_tasks):
            res = extract_frames_worker(task)
            if res:
                processed_results.append(res)

    # Step 3: HDF5 저장 (Zero-padding 방지)
    saved_count = len(processed_results)
    if saved_count == 0:
        print("❌ 저장할 데이터가 없습니다. 원본 경로를 확인하세요.")
        return

    with h5py.File(out_path, 'w') as hf:
        x_ds = hf.create_dataset('x', shape=(saved_count, 16, 3, 224, 224), dtype=np.float32)
        y_ds = hf.create_dataset('y', shape=(saved_count,), dtype=np.int64)
        
        for i, (video, label) in enumerate(processed_results):
            x_ds[i] = video
            y_ds[i] = label

    print(f"\n✨ 완료: {out_path} (총 {saved_count}개 저장)")

# [4] 실행부
if __name__ == "__main__":
    # --- 설정 ---
    RAW_PATH = "../data/raw"
    OUT_PATH = "../data/processed/dataset_balanced.h5"
    MAX_SAMPLES = 300
    
    # ⭐ 병렬 처리를 켜려면 True, 끄려면 False로 설정하세요.
    USE_PARALLEL = True 
    # ----------
    
    create_dataset(RAW_PATH, OUT_PATH, max_samples=MAX_SAMPLES, parallel=USE_PARALLEL)