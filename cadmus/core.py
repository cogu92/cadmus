"""
cadmus.core
-----------
Main CADMUS orchestrator. Coordinates all layers: profiling,
encoding, correction (A/B/C), drift detection, and feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


@dataclass
class CADMUSResult:
    """Output of a CADMUS.run() call."""
    counts: dict
    viability_score: float
    noise_report: dict
    syndrome_log: list
    layers_applied: list[str]
    raw_counts: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"CADMUS Result",
            f"  Viability score : {self.viability_score:.3f}",
            f"  Layers applied  : {', '.join(self.layers_applied)}",
            f"  Syndrome entries: {len(self.syndrome_log)}",
            f"  Top counts      : {dict(list(self.counts.items())[:5])}",
        ]
        return "\n".join(lines)


class CADMUS:
    """
    Circuit-Aware Dynamic Multilayer Update System.

    Full-stack adaptive quantum error correction pipeline.

    Parameters
    ----------
    backend : str or Backend
        'aer_simulator' for local simulation, or an IBM backend name/object.
    layers : list[str]
        Which correction layers to apply. Default: ['A', 'B', 'C']
        A = readout correction
        B = mid-circuit correction
        C = adaptive hybrid decoder
    drift_window : int
        Sliding window size for syndrome drift detection.
    drift_threshold : float
        KL-divergence threshold that triggers recalibration.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        backend="aer_simulator",
        layers: Optional[list[str]] = None,
        drift_window: int = 50,
        drift_threshold: float = 0.15,
        verbose: bool = False,
    ):
        self.layers = layers or ["A", "B", "C"]
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.verbose = verbose

        # Resolve backend
        if isinstance(backend, str) and backend == "aer_simulator":
            self.backend = AerSimulator()
            self._backend_name = "aer_simulator"
        else:
            self.backend = backend
            self._backend_name = str(backend)

        if verbose:
            logging.basicConfig(level=logging.INFO)

        logger.info("CADMUS initialized. Backend=%s, Layers=%s",
                    self._backend_name, self.layers)

    def run(self, circuit: QuantumCircuit, shots: int = 1024) -> CADMUSResult:
        """
        Execute circuit with full CADMUS error correction pipeline.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to execute.
        shots : int
            Number of shots.

        Returns
        -------
        CADMUSResult
        """
        from cadmus.profiler.health_check import CircuitHealthCheck
        from cadmus.profiler.noise_profiler import NoiseProfiler
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        from cadmus.correction.readout import ReadoutCorrector
        from cadmus.correction.mid_circuit import MidCircuitCorrector
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        from cadmus.drift.detector import SyndromeDriftDetector
        from cadmus.utils.executor import execute_circuit_per_shot

        syndrome_log = []
        layers_applied = []

        # Ensure data qubits are measured *before* encoding. AdaptiveEncoder
        # adds its own syndrome classical register, so a circuit with no
        # measurements (e.g. the quickstart example) would already have
        # clbits by the time it reached the executor's measure_all()
        # fallback — leaving the actual data qubits unmeasured.
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        # ── Pre-execution ──────────────────────────────────────────────
        logger.info("Step 1/6: Circuit Health Check")
        health = CircuitHealthCheck(self.backend)
        score_report = health.score(circuit)
        viability = score_report["viability"]

        if viability < 0.3:
            logger.warning(
                "Circuit viability is %.2f — consider optimizing before running.",
                viability
            )

        logger.info("Step 2/6: Noise Profiling")
        profiler = NoiseProfiler(self.backend)
        noise_map = profiler.profile(circuit)

        # ── Encoding ───────────────────────────────────────────────────
        # encode() defers the circuit's own final measurement so syndrome
        # extraction (and Layer B's correction) runs while data qubits are
        # still live — a correction applied after measurement can't change
        # the reported result. finalize_measurement() re-attaches it once
        # Layer B has had its chance.
        logger.info("Step 3/6: Adaptive Encoding")
        encoder = AdaptiveEncoder(noise_map)
        encoded = encoder.encode(circuit)

        # ── Layer B: Mid-circuit correction ────────────────────────────
        if "B" in self.layers:
            logger.info("Step 4/6: Applying Layer B (mid-circuit correction)")
            mid_corrector = MidCircuitCorrector()
            encoded = mid_corrector.apply(encoded, encoder)
            layers_applied.append("B")

        encoded = encoder.finalize_measurement(encoded)

        # ── Execute ────────────────────────────────────────────────────
        logger.info("Step 5/6: Executing circuit (%d shots)", shots)
        raw_counts, syndromes, per_shot = execute_circuit_per_shot(
            encoded, self.backend, shots
        )
        syndrome_log.extend(syndromes)

        # ── Layer C: Hybrid decoder ────────────────────────────────────
        # Layer A only ever cares about the *data* qubits — calibrating
        # over the combined data+ancilla space (measured benefit analysis,
        # FINDINGS.md) adds calibration overhead with no benefit, so both
        # the Layer-C-skipped fallback and Layer C's own output are kept
        # data-only here.
        n_data = encoder.n_data
        counts = _marginalize_to_data(raw_counts, n_data)
        if "C" in self.layers:
            logger.info("Applying Layer C (hybrid MWPM + Bayes decoder)")
            decoder = HybridDecoder(
                noise_map, n_data=n_data, pairs_order=encoder.pairs_order
            )
            if per_shot:
                counts = decoder.decode_per_shot(per_shot)
            layers_applied.append("C")

        # ── Layer A: Readout correction ────────────────────────────────
        if "A" in self.layers:
            logger.info("Applying Layer A (readout correction)")
            readout = ReadoutCorrector(self.backend)
            counts = readout.correct(counts)
            layers_applied.append("A")

        # ── Drift detection ────────────────────────────────────────────
        detector = SyndromeDriftDetector(
            window=self.drift_window,
            threshold=self.drift_threshold,
        )
        for s in syndromes:
            detector.update(s)
        if detector.drift_detected():
            logger.warning(
                "Syndrome drift detected (KL=%.4f). "
                "Consider recalibrating before next run.",
                detector.last_kl
            )

        return CADMUSResult(
            counts=counts,
            viability_score=viability,
            noise_report={**score_report, "noise_map": noise_map},
            syndrome_log=syndrome_log,
            layers_applied=layers_applied,
            raw_counts=_marginalize_to_data(raw_counts, n_data),
        )


def _marginalize_to_data(counts: dict, n_data: int) -> dict:
    """Collapse a (possibly data+syndrome combined) counts dict down to
    just the data-qubit bits — a no-op when no ancilla was added (the
    combined key is already data-only)."""
    out: dict = {}
    for key, count in counts.items():
        data_bits = key.replace(" ", "")[:n_data]
        out[data_bits] = out.get(data_bits, 0) + count
    return out
