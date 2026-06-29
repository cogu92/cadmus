"""
cadmus.profiler.health_check
-----------------------------
Pre-execution circuit viability scoring.

Scores a circuit 0.0 (very noisy, likely to fail) to 1.0 (clean circuit).
Uses four sub-metrics:
  - Depth penalty   : penalizes deep circuits relative to T2
  - Gate error budget: accumulated gate error probability
  - Two-qubit ratio : fraction of expensive 2Q gates
  - TVD estimate    : estimated Total Variation Distance via Aer noise model

Does NOT require hardware access for Aer mode — uses the backend's
reported noise model to simulate.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

logger = logging.getLogger(__name__)

# Typical T2 for ibm_torino heavy-hex qubits (µs → gate units)
_DEFAULT_T2_US = 150.0
_TYPICAL_2Q_GATE_TIME_US = 0.4  # CNOT/ECR on Eagle r3


class CircuitHealthCheck:
    """
    Compute a viability score for a quantum circuit on a given backend.

    Parameters
    ----------
    backend : AerSimulator or IBM backend
        Backend used to extract noise parameters.
    t2_us : float
        Average T2 time in microseconds (used if backend has no calibration).
    """

    def __init__(self, backend=None, t2_us: float = _DEFAULT_T2_US):
        self.backend = backend or AerSimulator()
        self.t2_us = t2_us
        self._noise_model: NoiseModel | None = None
        self._try_load_noise_model()

    def _try_load_noise_model(self) -> None:
        try:
            if hasattr(self.backend, "properties") and self.backend.properties():
                self._noise_model = NoiseModel.from_backend(self.backend)
                logger.debug("Noise model loaded from backend properties.")
        except Exception:
            logger.debug("Could not load noise model — using defaults.")

    # ── Public API ─────────────────────────────────────────────────────

    def score(self, circuit: QuantumCircuit) -> dict:
        """
        Score the circuit and return a detailed report.

        Returns
        -------
        dict with keys:
          viability         : float 0–1 (composite score)
          depth_penalty     : float 0–1 (0 = no penalty)
          gate_error_budget : float 0–1 (accumulated error)
          two_qubit_ratio   : float 0–1 (fraction of 2Q gates)
          tvd_estimate      : float 0–1 (estimated TVD)
          recommendation    : str
        """
        tc = transpile(circuit, self.backend, optimization_level=1)

        depth_penalty = self._depth_penalty(tc)
        gate_error = self._gate_error_budget(tc)
        tq_ratio = self._two_qubit_ratio(tc)
        tvd = self._tvd_estimate(circuit)

        # Weighted composite (tunable)
        viability = (
            0.35 * (1.0 - depth_penalty)
            + 0.35 * (1.0 - gate_error)
            + 0.15 * (1.0 - tq_ratio)
            + 0.15 * (1.0 - tvd)
        )
        viability = float(np.clip(viability, 0.0, 1.0))

        return {
            "viability": round(viability, 4),
            "depth_penalty": round(depth_penalty, 4),
            "gate_error_budget": round(gate_error, 4),
            "two_qubit_ratio": round(tq_ratio, 4),
            "tvd_estimate": round(tvd, 4),
            "recommendation": self._recommendation(viability),
        }

    # ── Sub-metrics ────────────────────────────────────────────────────

    def _depth_penalty(self, tc: QuantumCircuit) -> float:
        """Penalize based on circuit depth relative to T2."""
        depth = tc.depth()
        # Approximate time per layer in µs
        time_per_layer_us = _TYPICAL_2Q_GATE_TIME_US
        total_time_us = depth * time_per_layer_us
        # penalty grows as time approaches T2
        penalty = 1.0 - np.exp(-total_time_us / self.t2_us)
        return float(np.clip(penalty, 0.0, 1.0))

    def _gate_error_budget(self, tc: QuantumCircuit) -> float:
        """Estimate cumulative error from individual gate error rates."""
        if self._noise_model is None:
            # Fallback: typical IBM gate errors
            n_1q = sum(1 for inst in tc.data if len(inst.qubits) == 1
                       and inst.operation.name not in ("barrier", "measure"))
            n_2q = sum(1 for inst in tc.data if len(inst.qubits) == 2)
            p_success = (0.9997 ** n_1q) * (0.993 ** n_2q)
        else:
            # Use noise model gate errors
            p_success = 1.0
            basis_gates = self._noise_model.basis_gates
            for inst in tc.data:
                name = inst.operation.name
                if name in ("barrier", "measure", "reset"):
                    continue
                n_q = len(inst.qubits)
                # Typical fallback rates if gate not in noise model
                if n_q == 1:
                    p_success *= 0.9997
                elif n_q == 2:
                    p_success *= 0.993

        error = float(np.clip(1.0 - p_success, 0.0, 1.0))
        return error

    def _two_qubit_ratio(self, tc: QuantumCircuit) -> float:
        """Fraction of 2Q gates out of all gates (excluding barriers/measures)."""
        ops = [inst for inst in tc.data
               if inst.operation.name not in ("barrier", "measure", "reset")]
        if not ops:
            return 0.0
        n_2q = sum(1 for inst in ops if len(inst.qubits) == 2)
        return float(n_2q / len(ops))

    def _tvd_estimate(self, circuit: QuantumCircuit) -> float:
        """
        Estimate TVD by simulating with and without noise.
        Returns normalized TVD in [0, 1].
        Only runs if circuit has classical bits; otherwise returns depth-based estimate.
        """
        if circuit.num_clbits == 0:
            # No measurements — use depth as proxy
            return min(circuit.depth() / 200.0, 1.0)
        try:
            from qiskit_aer import AerSimulator
            shots = 512

            # Keep the circuit at its own (small) qubit count for this local
            # simulation. transpile(circuit, self.backend, ...) (used for the
            # other sub-metrics) expands the register to the real device's
            # full width (e.g. 156 qubits) — once a per-qubit noise model is
            # attached, Aer can no longer auto-truncate idle qubits and tries
            # to allocate state for the whole device, which OOMs for any
            # real backend beyond ~25-30 qubits.
            # Pseudo-ops (measure/reset/delay/if_else/measure_2) aren't real
            # transpile targets and aren't accepted in `basis_gates`.
            basis = (
                [g for g in self._noise_model.basis_gates
                 if g not in ("measure", "measure_2", "reset", "delay", "if_else")]
                if self._noise_model else None
            )
            sim_circuit = (
                transpile(circuit, basis_gates=basis, optimization_level=1)
                if basis else circuit
            )

            # Ideal simulation
            ideal_sim = AerSimulator()
            ideal_job = ideal_sim.run(sim_circuit, shots=shots)
            ideal_counts = ideal_job.result().get_counts()

            # Noisy simulation
            if self._noise_model:
                noisy_sim = AerSimulator(noise_model=self._noise_model)
            else:
                from qiskit_aer.noise import depolarizing_error
                nm = NoiseModel()
                err_1q = depolarizing_error(0.001, 1)
                err_2q = depolarizing_error(0.01, 2)
                nm.add_all_qubit_quantum_error(err_1q, ["u", "u2", "u3"])
                nm.add_all_qubit_quantum_error(err_2q, ["cx", "ecr"])
                noisy_sim = AerSimulator(noise_model=nm)

            noisy_job = noisy_sim.run(sim_circuit, shots=shots)
            noisy_counts = noisy_job.result().get_counts()

            # Compute TVD
            all_keys = set(ideal_counts) | set(noisy_counts)
            tvd = 0.5 * sum(
                abs(ideal_counts.get(k, 0) / shots - noisy_counts.get(k, 0) / shots)
                for k in all_keys
            )
            return float(np.clip(tvd, 0.0, 1.0))

        except Exception as e:
            logger.debug("TVD simulation failed: %s", e)
            return 0.05  # conservative default

    def _recommendation(self, viability: float) -> str:
        if viability >= 0.85:
            return "Excellent — run as-is."
        elif viability >= 0.70:
            return "Good — minor optimization may help."
        elif viability >= 0.50:
            return "Moderate — consider reducing circuit depth before running."
        elif viability >= 0.30:
            return "Poor — significant optimization or shorter circuit recommended."
        else:
            return "Critical — circuit likely to produce unreliable results."
