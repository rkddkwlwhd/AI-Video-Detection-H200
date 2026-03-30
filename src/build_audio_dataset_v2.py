import os
import h5py
import random
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm

# [설정]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(BASE_DIR, "data_audio", "real")
FAKE_DIR = os.path.join(BASE_DIR, "data_audio", "fake")
OUT_H5_PATH = os.path.join(BASE_DIR, "data_audio", "processed", "audio_dataset_v2.h5")

SR = 16000
SEGMENTS = 16

def audio_to_spec_torch(path):
    try:
        y, sr = sf.read(path)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        target_len = SR * 4
        y = np.pad(y, (0, max(0, target_len - len(y))))[:target_len]
        
        y_t = torch.from_numpy(y).float()
        segment_len = len(y) // SEGMENTS
        spec_frames = []
        window = torch.hann_window(256)
        
        for i in range(SEGMENTS):
            chunk = y_t[i*segment_len : (i+1)*segment_len]
            stft = torch.stft(chunk, n_fft=256, hop_length=128, win_length=256, window=window, return_complex=True)
            spec = stft.abs().pow(2)
            spec_db = 10 * torch.log10(spec + 1e-10)
            
            spec_db = spec_db.unsqueeze(0).unsqueeze(0)
            spec_res = F.interpolate(spec_db, size=(224, 224), mode='bilinear', align_corners=False)
            
            spec_res = spec_res.squeeze()
            s_min, s_max = spec_res.min(), spec_res.max()
            spec_norm = (spec_res - s_min) / (s_max - s_min + 1e-6)
            spec_frames.append(spec_norm)
            
        return torch.stack(spec_frames)
    except:
        return None

def main():
    os.makedirs(os.path.dirname(OUT_H5_PATH), exist_ok=True)
    real_files = [(os.path.join(REAL_DIR, f), 0) for f in os.listdir(REAL_DIR) if f.endswith('.wav')]
    fake_files = [(os.path.join(FAKE_DIR, f), 1) for f in os.listdir(FAKE_DIR) if f.endswith('.wav')]
    
    tasks = real_files + fake_files
    random.shuffle(tasks)
    
    with h5py.File(OUT_H5_PATH, 'w') as hf:
        x_ds = hf.create_dataset('x', shape=(len(tasks), SEGMENTS, 3, 224, 224), dtype=np.float32)
        y_ds = hf.create_dataset('y', shape=(len(tasks),), dtype=np.int64)
        
        for i, (path, label) in enumerate(tqdm(tasks, desc="Building HDF5 v2")):
            spec = audio_to_spec_torch(path)
            if spec is not None:
                # 3채널 복제 [16, 224, 224] -> [16, 3, 224, 224]
                spec_3ch = spec.unsqueeze(1).repeat(1, 3, 1, 1)
                x_ds[i] = spec_3ch.numpy()
                y_ds[i] = label

    print(f"✨ 데이터셋 v2 생성 완료: {OUT_H5_PATH}")

if __name__ == "__main__":
    main()