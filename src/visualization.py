import matplotlib.pyplot as plt
import numpy as np
import wave

def visualize_pcg_ecg(wav_path, hea_path, duration=5):
    """
    Visualize PCG (from wav) and ECG (from dat) using .hea file metadata.

    wav_path : str -> path to .wav file (PCG signal)
    hea_path : str -> path to .hea file (metadata, describes channels)
    duration : int -> seconds of signal to plot
    """
    # --- Step 1: Load PCG wav ---
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        signal = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
    
    # Normalize PCG
    pcg = signal / np.max(np.abs(signal))

    # --- Step 2: Parse .hea metadata ---
    with open(hea_path, "r") as f:
        header_lines = f.readlines()
    
    label = None
    ecg_file = None
    ecg_fs = None
    for line in header_lines:
        if line.startswith("#"):
            label = line.strip("# ").strip()
        elif ".dat" in line and "ECG" in line:
            ecg_file = line.split()[0]
            ecg_fs = int(line.split()[1])  # usually sampling freq is second field

    # --- Step 3: Select PCG time window ---
    n_samples_pcg = min(duration * sample_rate, len(pcg))
    time_pcg = np.arange(n_samples_pcg) / sample_rate

    # --- Step 4: Try ECG (if available) ---
    ecg = None
    time_ecg = None
    if ecg_file:
        try:
            ecg_path = hea_path.replace(hea_path.split("/")[-1], ecg_file)
            ecg = np.fromfile(ecg_path, dtype=np.int16)
            n_samples_ecg = min(duration * ecg_fs, len(ecg))
            ecg = ecg[:n_samples_ecg] / np.max(np.abs(ecg))
            time_ecg = np.arange(n_samples_ecg) / ecg_fs
        except Exception as e:
            print(f"ECG load error: {e}")

    # --- Step 5: Plot ---
    fig, axs = plt.subplots(2 if ecg is not None else 1, 1, figsize=(12, 6), sharex=False)
    
    if ecg is not None:
        axs[0].plot(time_pcg, pcg[:n_samples_pcg], color="blue")
        axs[0].set_title(f"PCG Signal ({wav_path}) | Label: {label}")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)

        axs[1].plot(time_ecg, ecg, color="red")
        axs[1].set_title(f"ECG Signal ({ecg_file})")
        axs[1].set_xlabel("Time (seconds)")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True)
    else:
        axs.plot(time_pcg, pcg[:n_samples_pcg], color="blue")
        axs.set_title(f"PCG Signal Only ({wav_path}) | Label: {label}")
        axs.set_xlabel("Time (seconds)")
        axs.set_ylabel("Amplitude")
        axs.grid(True)

    plt.tight_layout()
    plt.show()
