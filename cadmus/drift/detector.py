"""
cadmus.drift.detector
----------------------
Syndrome Drift Detector.

Monitors syndrome measurement history using KL-divergence over a
sliding window. When the distribution of syndromes shifts significantly
from baseline, it signals that the hardware has changed and
recalibration is needed.

This is the feedback loop component of CADMUS — the piece that makes
the system truly adaptive rather than just pre-calibrated.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_EPSILON = 1e-10  # numerical stability for KL divergence


class SyndromeDriftDetector:
    """
    Detect hardware drift by monitoring syndrome distribution.

    Parameters
    ----------
    window : int
        Size of the sliding window for syndrome history.
    threshold : float
        KL-divergence threshold that triggers a drift alert.
    baseline_window : int
        Number of initial syndromes used to establish baseline.
        If None, uses first `window` samples.
    """

    def __init__(
        self,
        window: int = 50,
        threshold: float = 0.15,
        baseline_window: Optional[int] = None,
    ):
        self.window = window
        self.threshold = threshold
        self.baseline_window = baseline_window or window

        self._history: deque = deque(maxlen=window * 2)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_set = False
        self.last_kl: float = 0.0
        self._drift_count: int = 0

    def update(self, syndrome: list | int) -> None:
        """
        Add a new syndrome measurement to the history.

        Parameters
        ----------
        syndrome : list of int or int
            Syndrome bits from one execution cycle.
        """
        if isinstance(syndrome, int):
            syndrome = [int(b) for b in format(syndrome, "b")]
        self._history.append(list(syndrome))

        # Establish baseline from first N samples
        if not self._baseline_set and len(self._history) >= self.baseline_window:
            self._build_baseline()

    def drift_detected(self) -> bool:
        """
        Check if significant drift has been detected.

        Returns
        -------
        bool : True if KL-divergence exceeds threshold.
        """
        if not self._baseline_set:
            return False
        if len(self._history) < self.window:
            return False

        kl = self._compute_kl()
        self.last_kl = kl

        if kl > self.threshold:
            self._drift_count += 1
            logger.warning(
                "Syndrome drift detected! KL=%.4f > threshold=%.4f "
                "(count=%d)",
                kl, self.threshold, self._drift_count,
            )
            return True

        return False

    def reset_baseline(self) -> None:
        """Reset baseline to current window (after recalibration)."""
        self._baseline_set = False
        self._history.clear()
        self.last_kl = 0.0
        logger.info("Drift detector baseline reset.")

    @property
    def drift_count(self) -> int:
        """Total number of drift events detected."""
        return self._drift_count

    def summary(self) -> str:
        return (
            f"SyndromeDriftDetector | "
            f"window={self.window} | "
            f"threshold={self.threshold} | "
            f"last_kl={self.last_kl:.4f} | "
            f"drift_events={self._drift_count} | "
            f"baseline_set={self._baseline_set}"
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _build_baseline(self) -> None:
        """Build baseline distribution from first N syndrome samples.
        
        We encode the syndrome as a histogram over observed patterns
        (flattened bitstrings) rather than per-bit averages. This
        preserves correlation structure and makes JS divergence
        sensitive to pattern shifts, not just mean bit-flip rates.
        """
        baseline_samples = list(self._history)[: self.baseline_window]
        # Convert each syndrome to a tuple key for counting
        from collections import Counter
        counts = Counter(tuple(s) for s in baseline_samples)
        all_keys = list(counts.keys())
        self._baseline_keys = all_keys
        arr = np.array([counts[k] for k in all_keys], dtype=float)
        arr += _EPSILON
        self._baseline = arr / arr.sum()
        self._baseline_set = True
        logger.info(
            "Drift detector baseline established from %d samples (%d unique patterns).",
            self.baseline_window, len(all_keys),
        )

    def _compute_kl(self) -> float:
        """
        Compute Jensen-Shannon divergence between baseline pattern
        distribution and current window pattern distribution.
        """
        current_samples = list(self._history)[-self.window :]
        if not current_samples:
            return 0.0

        from collections import Counter
        counts = Counter(tuple(s) for s in current_samples)

        # Build Q over same keys as baseline, plus new keys
        all_keys = list(set(self._baseline_keys) | set(counts.keys()))
        n_baseline = self.baseline_window
        n_current = len(current_samples)

        # Baseline counts (unnormalized)
        p_counts = np.array(
            [self._baseline[self._baseline_keys.index(k)] * n_baseline
             if k in self._baseline_keys else 0.0
             for k in all_keys], dtype=float
        )
        # Current counts
        q_counts = np.array(
            [counts.get(k, 0) for k in all_keys], dtype=float
        )

        p = np.clip(p_counts, _EPSILON, None); p /= p.sum()
        q = np.clip(q_counts, _EPSILON, None); q /= q.sum()

        m = 0.5 * (p + q)
        js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        return float(max(0.0, js))
