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
    features["spectral_bandwidth"] = np.sqrt(
        np.sum(((freqs - mean_freq) ** 2) * psd) / np.sum(psd)
    )
    return features


def read_label_from_header(header_file):
    """
    Parse .hea file to extract Normal/Abnormal label
    """
    with open(header_file, "r") as f:
        content = f.read().lower()
    if "# normal" in content:
        return 0
    elif "# abnormal" in content:
        return 1
    else:
        return -1


def process_dataset(data_dirs):
    """
    Process all wav files in one or multiple directories
    Returns: Feature matrix X, Label vector y
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    X, y = [], []
    print("=== Using FINAL process_dataset (with .hea labels) ===")

    for data_dir in data_dirs:
        for fname in os.listdir(data_dir):
            if fname.endswith(".wav"):
                wav_path = os.path.join(data_dir, fname)
                hea_path = wav_path.replace(".wav", ".hea")

                # Extract features
                fs, audio = load_wav(wav_path)
                feats = extract_features(audio, fs)
                X.append(list(feats.values()))

                # Extract label
                if os.path.exists(hea_path):
                    label = read_label_from_header(hea_path)
                else:
                    label = -1
                y.append(label)

    return np.array(X), np.array(y)