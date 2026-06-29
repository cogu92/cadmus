"""
cadmus.correction.readout
--------------------------
Layer A: Readout error correction via confusion matrix inversion.

Builds on QVoting's readout mitigation approach. Calibrates a
confusion matrix A where A[i][j] = P(measure i | prepare j),
then applies A^{-1} to raw counts to get mitigated counts.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from cadmus.utils.executor import execute_circuit

logger = logging.getLogger(__name__)


class ReadoutCorrector:
    """
    Readout error corrector using confusion matrix inversion.

    Parameters
    ----------
    backend : AerSimulator or IBM backend
    shots_cal : int
        Shots used for calibration circuits.
    """

    def __init__(self, backend=None, shots_cal: int = 1024):
        self.backend = backend or AerSimulator()
        self.shots_cal = shots_cal
        self._cal_matrix: Optional[np.ndarray] = None
        self._n_qubits: Optional[int] = None

    def calibrate(self, n_qubits: int) -> np.ndarray:
        """
        Build and cache the confusion matrix for n_qubits.

        Returns
        -------
        np.ndarray of shape (2^n, 2^n)
        """
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        n = n_qubits
        dim = 2 ** n
        cal_matrix = np.zeros((dim, dim))

        for j in range(dim):
            # Prepare basis state |j⟩
            qc = QuantumCircuit(n, n)
            for bit_idx in range(n):
                if (j >> bit_idx) & 1:
                    qc.x(bit_idx)
            qc.measure(range(n), range(n))

            # execute_circuit() dispatches to backend.run() for Aer or to
            # SamplerV2 for real IBM backends (backend.run() was removed
            # from qiskit-ibm-runtime).
            counts, _ = execute_circuit(qc, self.backend, shots=self.shots_cal)

            for bitstring, count in counts.items():
                i = int(bitstring.replace(" ", ""), 2)
                cal_matrix[i][j] = count / self.shots_cal

        self._cal_matrix = cal_matrix
        self._n_qubits = n_qubits
        logger.info(
            "Readout calibration complete for %d qubits. "
            "Matrix condition number: %.2f",
            n, np.linalg.cond(cal_matrix)
        )
        return cal_matrix

    def correct(self, counts: dict) -> dict:
        """
        Apply readout correction to raw counts.

        Parameters
        ----------
        counts : dict
            Raw measurement counts from execution.

        Returns
        -------
        dict : mitigated counts (may contain small negative values
               clipped to 0 due to matrix inversion noise).
        """
        if not counts:
            return counts

        # Infer n_qubits from counts
        sample_key = next(iter(counts))
        n_qubits = len(sample_key.replace(" ", ""))

        if self._cal_matrix is None or self._n_qubits != n_qubits:
            logger.info(
                "Calibration matrix not found for %d qubits — calibrating now.",
                n_qubits,
            )
            self.calibrate(n_qubits)

        dim = 2 ** n_qubits
        total_shots = sum(counts.values())

        # Build probability vector
        p_raw = np.zeros(dim)
        for bitstring, count in counts.items():
            idx = int(bitstring.replace(" ", ""), 2)
            p_raw[idx] = count / total_shots

        # Invert: p_ideal = A^{-1} @ p_raw
        try:
            cal_inv = np.linalg.pinv(self._cal_matrix)
            p_corrected = cal_inv @ p_raw
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed — returning raw counts.")
            return counts

        # Clip negatives (artifact of inversion) and renormalize
        p_corrected = np.clip(p_corrected, 0.0, None)
        total = p_corrected.sum()
        if total > 0:
            p_corrected /= total

        # Convert back to counts
        corrected_counts = {}
        for idx in range(dim):
            if p_corrected[idx] > 1e-6:
                bitstring = format(idx, f"0{n_qubits}b")
                corrected_counts[bitstring] = int(
                    round(p_corrected[idx] * total_shots)
                )

        logger.debug(
            "Readout correction applied. Raw keys: %d, Corrected keys: %d",
            len(counts), len(corrected_counts)
        )
        return corrected_counts
