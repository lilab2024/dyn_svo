"""Dominant stop-and-go period from real speed spectra and the detrend-window rule."""
import math

import numpy as np
from scipy.signal import welch


def experiment_spectrum(df, fs=10, step_frames=3, segment_s=120, overlap=0.5, nfft_factor=8):
    """Mean Welch PSD of vehicle speeds (rows with in_window, every step_frames frames)."""
    d = df[df["in_window"] & (df["k"] % step_frames == 0)]
    nperseg = int(segment_s * fs)
    psds, n_samples = [], []
    for _, g in d.groupby("veh_id"):
        v = g.sort_values("k")["v"].to_numpy(dtype=float)
        f, p = welch(v, fs=fs, nperseg=min(nperseg, len(v)), noverlap=min(int(nperseg * overlap), len(v) // 2),
                     nfft=nfft_factor * nperseg, detrend="constant")
        psds.append(p)
        n_samples.append(len(v))
    return f, np.mean(psds, axis=0), len(psds), int(min(n_samples))


def dominant_period(f, psd, period_range=(10.0, 120.0)):
    """Period (s) of the PSD maximum inside the range, peak / median ratio, and edge flag."""
    lo, hi = period_range
    band = (f >= 1.0 / hi) & (f <= 1.0 / lo)
    fb, pb = f[band], psd[band]
    i = int(np.argmax(pb))
    return float(1.0 / fb[i]), float(pb[i] / np.median(pb)), bool(i == 0 or i == len(fb) - 1)


def detrend_window(periods, lo=30, hi=120, step=10):
    """ rule: max dominant period, rounded up to `step` s, clamped to [lo, hi]."""
    if not periods:
        raise ValueError("No experiment with a clear peak - the rule requires a discussion")
    return int(min(hi, max(lo, math.ceil(max(periods) / step) * step)))
