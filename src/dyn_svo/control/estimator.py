"""Causal moving-average speed estimator shared by controllers."""
from collections import deque

from dyn_svo.config import load_config
from dyn_svo.sim import idm


class MeanSpeedEstimator:
    """Mean of the speeds seen so far within the last window_s seconds (shorter at the start)."""

    def __init__(self, window_s=None, dt=None):
        window_s = float(load_config("baselines")["mean_speed_window_s"]) if window_s is None else float(window_s)
        dt = idm.DT if dt is None else float(dt)
        self.n_max = max(1, int(round(window_s / dt)))
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.n_max)
        self.total = 0.0

    def update(self, v):
        """Add the current speed and return the mean over the window (only past and current values)."""
        if len(self.buf) == self.n_max:
            self.total -= self.buf[0]
        self.buf.append(float(v))
        self.total += float(v)
        return self.total / len(self.buf)
