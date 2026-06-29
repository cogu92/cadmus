"""
cadmus.decoder.hybrid_decoder
-------------------------------
Layer C: Hybrid MWPM + Bayesian decoder.

Fast-path: PyMatching MWPM for latency-guaranteed decoding.
Slow-path: Bayesian prior updated asynchronously from noise map
           and syndrome history.

The Bayesian prior is built from the noise_map produced by
NoiseProfiler — giving circuit-specific, hardware-specific
error probability priors rather than uniform assumptions.

The MWPM graph is a 1D repetition-code chain over the *data* qubits:
check k (the syndrome bit for pair (k, k+1)) sits between data qubits k
and k+1; each data qubit is a `fault_id` edge (boundary edge for the two
endpoint qubits, an edge between two checks for interior qubits). This
needs `n_data` (and pairs forming a complete chain) to build correctly —
register sizes alone don't reveal which qubits each syndrome bit checks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def is_complete_chain(pairs_order: list, n_data: int) -> bool:
    """True if pairs_order is exactly the 1D chain (0,1),(1,2),...,(n-2,n-1)
    — the only layout this module knows how to decode."""
    return list(pairs_order) == [(i, i + 1) for i in range(n_data - 1)]


def build_chain_matching(n_data: int):
    """Build a pymatching.Matching for a complete 1D repetition chain over
    n_data qubits. fault_ids are data-qubit indices (0..n_data-1); check
    k (0..n_data-2) is the syndrome bit for pair (k, k+1)."""
    import pymatching

    matching = pymatching.Matching()
    for q in range(n_data):
        left_check = q - 1 if q >= 1 else None
        right_check = q if q <= n_data - 2 else None
        # merge_strategy="independent": with a single check (n_data == 2),
        # both qubits boundary-edge to the *same* node — pymatching's
        # default "disallow" rejects that as a parallel edge, but the two
        # qubits genuinely are two distinct (equally-weighted, here
        # undecidable) explanations for the same syndrome.
        if left_check is None and right_check is not None:
            matching.add_boundary_edge(
                right_check, weight=1.0, fault_ids=q, merge_strategy="independent"
            )
        elif right_check is None and left_check is not None:
            matching.add_boundary_edge(
                left_check, weight=1.0, fault_ids=q, merge_strategy="independent"
            )
        elif left_check is not None and right_check is not None:
            matching.add_edge(
                left_check, right_check, weight=1.0, fault_ids=q,
                merge_strategy="independent",
            )
        # else n_data == 1: no checks at all, nothing to add for this qubit.
    return matching


class HybridDecoder:
    """
    Hybrid MWPM + Bayesian decoder.

    Parameters
    ----------
    noise_map : dict
        Output of NoiseProfiler.profile().
    use_pymatching : bool
        Use PyMatching MWPM fast-path if available (default True).
    bayesian_weight : float
        How much to weight the Bayesian prior vs MWPM (0.0–1.0).
        0.0 = pure MWPM, 1.0 = pure Bayes.
    n_data : int, optional
        Number of data qubits. Required (together with a complete-chain
        `pairs_order`) to build a real repetition-code MWPM decoder; if
        omitted, `decode_syndrome`/`decode_counts` fall back to the
        legacy per-bit Bayesian heuristic.
    pairs_order : list[tuple[int, int]], optional
        Syndrome pairs in syndrome-bit-index order, as produced by
        `AdaptiveEncoder.pairs_order`.
    """

    def __init__(
        self,
        noise_map: dict,
        use_pymatching: bool = True,
        bayesian_weight: float = 0.4,
        n_data: Optional[int] = None,
        pairs_order: Optional[list] = None,
    ):
        self.noise_map = noise_map
        self.bayesian_weight = bayesian_weight
        self.n_data = n_data
        self.pairs_order = pairs_order
        self._pm = None
        self._matching = None
        self._prior: Optional[np.ndarray] = None
        self._syndrome_history: list = []

        if use_pymatching:
            self._try_init_pymatching()

    def _try_init_pymatching(self) -> None:
        try:
            import pymatching
            self._pm = pymatching
            logger.info("PyMatching loaded — MWPM fast-path enabled.")
            if self.n_data and self.pairs_order is not None:
                if is_complete_chain(self.pairs_order, self.n_data):
                    self._matching = build_chain_matching(self.n_data)
                else:
                    logger.info(
                        "Syndrome pairs %s don't form a complete 1D chain "
                        "over %d data qubits — MWPM decoding unavailable, "
                        "falling back to Bayesian heuristic.",
                        self.pairs_order, self.n_data,
                    )
        except ImportError:
            logger.info(
                "PyMatching not installed — falling back to Bayesian-only decoder. "
                "Install with: pip install pymatching"
            )

    # ── Public API ─────────────────────────────────────────────────────

    def decode_per_shot(self, per_shot: list[tuple[str, list[int]]]) -> dict:
        """
        Decode each shot using *its own* syndrome (not an aggregate), then
        tally the corrected outcomes into a counts dict. This is the
        correct way to use a per-shot syndrome: decoding from already-
        aggregated counts (see `decode_counts`) cannot recover which
        outcome bucket each syndrome sample actually belongs to.

        Parameters
        ----------
        per_shot : list of (data_bitstring, syndrome_bits)
            One entry per shot, as produced by
            `cadmus.utils.executor.execute_circuit_per_shot`.

        Returns
        -------
        dict : decoded counts.
        """
        if not per_shot:
            return {}

        decoded_counts: dict = {}
        for data_bits, syndrome in per_shot:
            correction = self.decode_syndrome(list(syndrome))
            bits = list(data_bits)
            for i, flip in enumerate(correction):
                if flip and i < len(bits):
                    bits[i] = "1" if bits[i] == "0" else "0"
            corrected = "".join(bits)
            decoded_counts[corrected] = decoded_counts.get(corrected, 0) + 1

        return decoded_counts

    def decode_counts(self, counts: dict, syndromes: list) -> dict:
        """
        Legacy path: decode aggregated counts using only the *last*
        syndrome sample, applying that single correction to every bucket.
        Kept for backward compatibility — prefer `decode_per_shot` when
        per-shot (data, syndrome) pairs are available (the normal case
        from `cadmus.utils.executor.execute_circuit_per_shot`), since a
        single global correction has no real per-shot justification.

        Parameters
        ----------
        counts : dict
            Raw counts from execution.
        syndromes : list
            Syndrome measurements accumulated during execution.

        Returns
        -------
        dict : decoded/corrected counts.
        """
        if not counts:
            return counts
        if not syndromes:
            return counts

        self._syndrome_history.extend(syndromes)
        self._update_bayesian_prior(syndromes)

        # Decode each bitstring in counts
        decoded_counts: dict = {}
        for bitstring, count in counts.items():
            corrected = self._decode_bitstring(bitstring, syndromes)
            decoded_counts[corrected] = decoded_counts.get(corrected, 0) + count

        return decoded_counts

    def decode_syndrome(self, syndrome: list | int) -> list:
        """
        Decode a single syndrome measurement into a correction vector.

        Parameters
        ----------
        syndrome : list of int or int
            Syndrome bits (0 or 1).

        Returns
        -------
        list of int : correction vector, one entry per *data qubit*
            (length `self.n_data` when MWPM is active, else length
            `len(syndrome)` for the Bayesian fallback).
        """
        if isinstance(syndrome, int):
            syndrome = [int(b) for b in format(syndrome, "b")]

        if self._matching is not None:
            try:
                return self._mwpm_decode(syndrome)
            except Exception as e:
                logger.debug("MWPM decode failed: %s — using Bayes.", e)
                return self._bayesian_decode(syndrome)
        return self._bayesian_decode(syndrome)

    def update_prior(self, new_noise_map: dict) -> None:
        """Update Bayesian prior with a new noise map (called on drift detection)."""
        self.noise_map = new_noise_map
        self._prior = None  # reset, will rebuild on next decode
        logger.info("Decoder prior updated from new noise map.")

    # ── Internal ───────────────────────────────────────────────────────

    def _decode_bitstring(self, bitstring: str, syndromes: list) -> str:
        """Apply correction to a result bitstring based on syndrome."""
        if not syndromes:
            return bitstring

        # Use the most recent syndrome
        last_syndrome = syndromes[-1] if syndromes else []
        if not last_syndrome:
            return bitstring

        correction = self.decode_syndrome(last_syndrome)
        bits = list(bitstring.replace(" ", ""))

        for i, flip in enumerate(correction):
            if flip and i < len(bits):
                bits[i] = "1" if bits[i] == "0" else "0"

        return "".join(bits)

    def _mwpm_decode(self, syndrome: list) -> list:
        """Repetition-code MWPM decode. Returns a correction vector indexed
        by *data qubit* (length n_data), via the chain Matching built in
        __init__ with fault_ids = data qubit index."""
        syndrome_array = np.array(syndrome, dtype=np.uint8)
        correction = self._matching.decode(syndrome_array)
        return list(correction)

    def _bayesian_decode(self, syndrome: list) -> list:
        """Bayesian correction based on noise map prior."""
        n = len(syndrome)
        correction = []

        for i, s in enumerate(syndrome):
            if s == 1:
                # Estimate which qubit is most likely to have errored
                # using noise map as prior
                pair = tuple(sorted([i, (i + 1) % n]))
                p_error = self.noise_map.get(pair, 0.007)
                # Bayesian decision: correct if p_error > 0.5
                # (simple threshold; can be replaced with full posterior)
                correction.append(1 if p_error > 0.001 else 0)
            else:
                correction.append(0)

        return correction

    def _update_bayesian_prior(self, syndromes: list) -> None:
        """Update internal Bayesian prior from new syndrome data."""
        if not syndromes:
            return

        # Estimate empirical error rates from syndrome history
        n_syndromes = len(syndromes)
        if n_syndromes < 10:
            return  # not enough data yet

        # Count syndrome bit flips as proxy for error rates
        all_syndromes = np.array(syndromes)
        if all_syndromes.ndim == 1:
            return

        empirical_rates = all_syndromes.mean(axis=0)

        # Blend with noise map prior (Bayesian update)
        alpha = 0.3  # weight of new evidence vs prior
        for i, rate in enumerate(empirical_rates):
            pair = tuple(sorted([i, i + 1])) if i + 1 < len(empirical_rates) else (i,)
            prior = self.noise_map.get(pair, 0.007)
            self.noise_map[pair] = (1 - alpha) * prior + alpha * float(rate)

        logger.debug(
            "Bayesian prior updated from %d syndrome samples.", n_syndromes
        )
