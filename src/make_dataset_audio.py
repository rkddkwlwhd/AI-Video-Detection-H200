import os, h5py, random, cv2
import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# [설정]
STZ_AUDIO_DIR = "../data_audio/standardized"
OUT_H5_PATH = "../data_audio/processed/audio_dataset.h5"
MAX_SAMPLES = 500
SR = 16000
SEGMENTS = 16

def audio_to_spectrogram_segments(v_path):
    try:
        # 1. 오디오 로드 (soundfile은 매우 가볍고 안정적입니다)
        y, sr = sf.read(v_path)
        
        # 모노 변환
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # 4초 분량 조절 (16000 * 4 = 64000 샘플)
        target_len = SR * 4
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        
        # 2. 16개 구간으로 분할하여 스펙트로그램 생성
        segment_len = len(y) // SEGMENTS
        spec_frames = []
        
        for i in range(SEGMENTS):
            chunk = y[i*segment_len : (i+1)*segment_len]
            
            # Scipy를 이용한 스펙트로그램 계산 (librosa 의존성 없음)
            # nperseg를 조절해 해상도 조정 가능
            f, t, Sxx = signal.spectrogram(chunk, fs=SR, nperseg=256, noverlap=128)
            
            # 로그 스케일(dB) 변환
            spec_db = 10 * np.log10(Sxx + 1e-10)
            
            # EfficientNet 입력 규격(224x224)으로 리사이즈 및 정규화
            spec_resized = cv2.resize(spec_db, (224, 224))
            spec_norm = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min() + 1e-6)
            spec_frames.append(spec_norm)
            
        return np.stack(spec_frames) # (16, 224, 224)
    except Exception as e:
        # 에러 발생 시 출력 (디버깅용)
        # print(f"❌ 에러: {e}")
        return None

def worker(task):
    path, label = task
    spec = audio_to_spectrogram_segments(path)
    if spec is not None:
        return spec, label
    return None

def create_audio_hdf5():
    os.makedirs(os.path.dirname(OUT_H5_PATH), exist_ok=True)
    tasks = []
    for label, cat in enumerate(['real', 'fake']):
        dir_path = os.path.join(STZ_AUDIO_DIR, cat)
        if not os.path.exists(dir_path): continue
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
        sampled = random.sample(files, min(len(files), MAX_SAMPLES))
        for f in sampled:
            tasks.append((f, label))
    
    random.shuffle(tasks)
    
    results = []
    print(f"🚀 [H200 Robust Mode] 스펙트로그램 가공 시작 (Librosa 없이 실행)...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as exe:
        for res in tqdm(exe.map(worker, tasks), total=len(tasks)):
            if res:
                results.append(res)

    if not results:
        print("❌ 저장할 데이터가 없습니다. 원본 wav 파일과 경로를 확인하세요.")
        return

    with h5py.File(OUT_H5_PATH, 'w') as hf:
        # (N, 16, 3, 224, 224) 형태로 저장하여 기존 EfficientNet 구조 유지
        x_ds = hf.create_dataset('x', shape=(len(results), SEGMENTS, 3, 224, 224), dtype=np.float32)
        y_ds = hf.create_dataset('y', shape=(len(results),), dtype=np.int64)
        
        for i, (spec, label) in enumerate(results):
            # 1채널 스펙트로그램을 3채널로 복제
            spec_3ch = np.repeat(spec[:, np.newaxis, :, :], 3, axis=1)
            x_ds[i] = spec_3ch
            y_ds[i] = label

    print(f"✨ 데이터셋 생성 성공: {OUT_H5_PATH} ({len(results)}개 저장)")

if __name__ == "__main__":
    create_audio_hdf5()