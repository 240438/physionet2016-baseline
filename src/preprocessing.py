import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

def load_wav(file_path, target_fs=2000):
    """
    Load and resample a .wav heart sound recording
    """
    fs, audio = wav.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio_resampled = signal.resample_poly(audio, target_fs, fs)
    return target_fs, audio_resampled

def extract_features(audio, fs=2000):
    """
    Extract baseline features from a heart sound recording.
    """
    features = {}
    features["energy"] = np.mean(audio ** 2)
    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    features["zcr"] = len(zero_crossings) / len(audio)
    freqs, psd = signal.welch(audio, fs=fs)
    features["spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd)
    mean_freq = features["spectral_centroid"]
    features["spectral_bandwidth"] = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd) / np.sum(psd))
    return features

def get_label_from_header(header_file):
    """
    Parse .hea file to extract label (Normal=0, Abnormal=1)
    """
    with open(header_file, "r") as f:
        lines = f.readlines()
    label = -1
    for line in lines:
        if "normal" in line.lower():
            label = 0
            break
        elif "abnormal" in line.lower():
            label = 1
            break
    return label

def process_dataset(data_dir="data/training-b/"):
    print("=== Using UPDATED process_dataset ===")  # Debug print
    X, y = [], []
    for fname in os.listdir(data_dir):
        if fname.endswith(".wav"):
            file_path = os.path.join(data_dir, fname)
            fs, audio = load_wav(file_path)
            feats = extract_features(audio, fs)
            X.append(list(feats.values()))

            # Get label from matching .hea file
            hea_file = os.path.splitext(file_path)[0] + ".hea"
            if os.path.exists(hea_file):
                label = get_label_from_header(hea_file)
            else:
                label = -1
            y.append(label)
    return np.array(X), np.array(y)
