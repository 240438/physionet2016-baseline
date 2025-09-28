import numpy as np

def segment_dummy(signal, sr=2000):
    """
    Dummy segmentation (for pipeline demo).
    Replace this with Springer HSMM segmentation.
    Returns fake beat intervals: list of (S1, systole, S2, diastole) boundaries.
    """
    n = len(signal)
    # fake segmentation into 1-sec beats
    step = sr
    intervals = []
    for start in range(0, n, step):
        end = min(start + step, n)
        intervals.append((start, start+step//4, start+step//2, start+3*step//4, end))
    return intervals
