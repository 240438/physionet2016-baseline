import numpy as np

def extract_features(signal, intervals, sr=2000):
    """
    Extracts 20 baseline features from segmented heartbeats.
    intervals = list of (S1_start, Sys_start, S2_start, Dia_start, end)
    Returns 20-D numpy array.
    """
    RR = []
    IntS1, IntS2, IntSys, IntDia = [], [], [], []
    RatioSysRR, RatioDiaRR, RatioSysDia = [], [], []
    AmpSysS1, AmpDiaS2 = [], []

    for i in range(1, len(intervals)):
        prev = intervals[i-1]
        curr = intervals[i]
        RR.append((curr[0] - prev[0]) / sr)

    for (s1, sys, s2, dia, end) in intervals:
        IntS1.append((sys - s1)/sr)
        IntSys.append((s2 - sys)/sr)
        IntS2.append((dia - s2)/sr)
        IntDia.append((end - dia)/sr)

        beat_len = (end - s1)/sr
        if beat_len > 0:
            RatioSysRR.append((s2 - sys)/sr / beat_len)
            RatioDiaRR.append((end - dia)/sr / beat_len)
            RatioSysDia.append(((s2 - sys)/sr) / ((end - dia)/sr + 1e-6))

        # amplitude features (rough placeholder)
        seg_sys = signal[sys:s2]
        seg_s1 = signal[s1:sys]
        seg_dia = signal[dia:end]
        seg_s2 = signal[s2:dia]

        if len(seg_sys) > 0 and len(seg_s1) > 0:
            AmpSysS1.append(np.mean(np.abs(seg_sys)) / (np.mean(np.abs(seg_s1)) + 1e-6))
        if len(seg_dia) > 0 and len(seg_s2) > 0:
            AmpDiaS2.append(np.mean(np.abs(seg_dia)) / (np.mean(np.abs(seg_s2)) + 1e-6))

    def stats(x):
        return [np.mean(x) if len(x) > 0 else 0, np.std(x) if len(x) > 0 else 0]

    features = []
    for arr in [RR, IntS1, IntS2, IntSys, IntDia,
                RatioSysRR, RatioDiaRR, RatioSysDia,
                AmpSysS1, AmpDiaS2]:
        features.extend(stats(arr))

    return np.array(features[:20])  # enforce 20-dim
