import librosa
import soundfile as sf
import numpy as np
import time

N_MFCC = 13
N_FFT = 512

# 26 MFCC stats + 10 other spectral/energy stats = 36 features
FEATURE_NAMES = (
    [f"mfcc{i+1}_mean" for i in range(N_MFCC)] + [f"mfcc{i+1}_std" for i in range(N_MFCC)] +
    ["zcr_mean", "zcr_std", "centroid_mean", "centroid_std",
     "rolloff_mean", "rolloff_std", "bandwidth_mean", "bandwidth_std",
     "rms_mean", "rms_std"]
)

def extract_features(path):
    """Extracts 36 features from a single audio clip."""
    y, sr = sf.read(path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [zcr.mean(), zcr.std()],
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [bandwidth.mean(), bandwidth.std()],
        [rms.mean(), rms.std()],
    ])
    return feats

def extract_all_features(paths):
    """Iterates through all paths and stacks the feature rows."""
    t0 = time.time()
    feature_rows = [extract_features(p) for p in paths]
    X_all = np.vstack(feature_rows)
    t1 = time.time()
    print(f"Extracted {X_all.shape[1]} features for {X_all.shape[0]} clips in {t1 - t0:.1f}s.")
    return X_all