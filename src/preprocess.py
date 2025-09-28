import librosa
import numpy as np

def load_audio(filepath, target_sr=2000):
    """Load and resample wav file to 2000 Hz mono."""
    y, sr = librosa.load(filepath, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # normalize
    y = (y - np.mean(y)) / np.std(y)
    return y, target_sr
