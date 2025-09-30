import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

from src.segmentation import run_segmentation   # beat boundaries
from src.feature import extract_features        # 20 baseline features per beat


def load_wav(file_path, target_fs=2000):
    """
    Load and resample a .wav heart sound recording.
    """
    fs, audio = wav.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mono
    audio_resampled = signal.resample_poly(audio, target_fs, fs)
    return target_fs, audio_resampled


def read_label_from_header(header_file):
    """
    Parse .hea file to extract Normal/Abnormal label.
    """
    with open(header_file, "r") as f:
        content = f.read().lower()
    if "# normal" in content:
        return 0
    elif "# abnormal" in content:
        return 1
    else:
        return -1   # unsure


def process_recording(wav_path):
    """
    Process a single recording → return final 40-D feature vector and label.
    """
    hea_path = wav_path.replace(".wav", ".hea")
    fs, audio = load_wav(wav_path)

    # --- segmentation step (beat intervals)
    intervals = run_segmentation(audio, fs)   # list of tuples: (s1, sys, s2, dia, end)

    # --- per-beat features
    beat_features = []
    for iv in intervals:
        fvec = extract_features(audio, [iv], sr=fs)  # 20-D
        beat_features.append(fvec)
    beat_features = np.array(beat_features)

    if len(beat_features) == 0:
        return None, None

    # --- aggregate to recording-level (mean + std → 40-D)
    mean_feats = np.mean(beat_features, axis=0)
    std_feats  = np.std(beat_features, axis=0)
    final_feats = np.concatenate([mean_feats, std_feats])

    # --- label
    label = read_label_from_header(hea_path) if os.path.exists(hea_path) else -1

    return final_feats, label


def process_dataset(data_dirs):
    """
    Process all .wav files in one or multiple directories.
    Returns: Feature matrix X (N×40), Label vector y (N).
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    X, y = [], []
    print("=== Using PAPER-style process_dataset (segmentation + 20 features + aggregation) ===")

    for data_dir in data_dirs:
        for fname in os.listdir(data_dir):
            if fname.endswith(".wav"):
                wav_path = os.path.join(data_dir, fname)
                feats, label = process_recording(wav_path)
                if feats is not None:
                    X.append(feats)
                    y.append(label)

    return np.array(X), np.array(y)
