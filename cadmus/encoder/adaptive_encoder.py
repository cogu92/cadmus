"""
cadmus.encoder.adaptive_encoder
---------------------------------
Adaptive QEC Encoder.

Encodes a circuit using a repetition code with heterogeneous ancilla
distribution: noisier qubit pairs get more ancilla qubits for syndrome
measurement. This is what distinguishes CADMUS from uniform-distance
surface codes.

Currently implements repetition code (1D) for compatibility with
IBM heavy-hex topology. Future versions will support 2D surface code
with adaptive distance per region.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

logger = logging.getLogger(__name__)

_DEFAULT_ANCILLA_PER_PAIR = 1
_HIGH_NOISE_THRESHOLD = 0.01   # pairs above this get extra ancilla
_CRITICAL_NOISE_THRESHOLD = 0.05  # pairs above this get 3 ancilla


class AdaptiveEncoder:
    """
    Encode a quantum circuit with heterogeneous ancilla allocation.

    Parameters
    ----------
    noise_map : dict
        Output of NoiseProfiler.profile(). {(q_i, q_j): error_rate}
    max_ancilla_per_pair : int
        Maximum ancilla qubits per noisy pair (default 2).
    """

    def __init__(
        self,
        noise_map: dict,
        max_ancilla_per_pair: int = 2,
    ):
        self.noise_map = noise_map
        self.max_ancilla_per_pair = max_ancilla_per_pair

    def encode(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Encode the circuit with adaptive ancilla for syndrome measurement.

        Returns the encoded circuit with ancilla registers appended, with
        syndrome extraction inserted *before* the circuit's own final
        measurement (if any) — a correction conditioned on the syndrome can
        only change the reported outcome if it runs before the data qubits
        collapse. If the circuit has its own measurements, they are deferred
        and must be re-attached with `finalize_measurement()` after any
        mid-circuit correction (Layer B) has been applied; call it
        immediately if no Layer B step is used.

        Sets `self.pairs_order` (the syndrome pairs in syndrome-bit-index
        order) and `self.n_data` so `MidCircuitCorrector`/`HybridDecoder`
        can build a real repetition-code decoder — register sizes alone
        don't reveal which qubits each syndrome bit checks.

        Parameters
        ----------
        circuit : QuantumCircuit

        Returns
        -------
        QuantumCircuit : encoded circuit (call `finalize_measurement()` to
            attach the original circuit's final measurement, if deferred).
        """
        n_data = circuit.num_qubits
        self.n_data = n_data
        ancilla_plan = self._plan_ancilla(circuit)
        total_ancilla = sum(ancilla_plan.values())
        self.pairs_order = sorted(ancilla_plan.keys())
        self.pair_first_syndrome_bit = {}
        self.n_syndrome_bits = total_ancilla

        if total_ancilla == 0:
            logger.info("No ancilla needed for this circuit (low noise).")
            self._pending_measurements = None
            return circuit.copy()

        logger.info(
            "Encoding: %d data qubits + %d ancilla qubits (%d syndrome pairs)",
            n_data, total_ancilla, len(ancilla_plan),
        )

        # Build encoded circuit
        data_reg = QuantumRegister(n_data, "d")
        anc_reg = QuantumRegister(total_ancilla, "a")
        syndrome_reg = ClassicalRegister(total_ancilla, "s")

        # Preserve original classical registers
        original_cregs = circuit.cregs

        encoded = QuantumCircuit(data_reg, anc_reg, syndrome_reg)
        for creg in original_cregs:
            if creg.name != "s":
                encoded.add_register(creg)

        # Map each original classical bit to its same-named register/position
        # in the encoded circuit. Matching by global circuit-wide index against
        # registers of different sizes (the previous approach) silently
        # misroutes bits whenever a coincidentally-valid index exists in the
        # wrong register — e.g. measure(q1, c[1]) landing in a 1-bit syndrome
        # register because index 1 doesn't fit there... except when it does
        # for index 0, which corrupts c[0] by leaving it permanently unwritten.
        clbit_map = {}
        for creg in original_cregs:
            if creg.name == "s":
                continue
            new_creg = next(c for c in encoded.cregs if c.name == creg.name)
            for i, bit in enumerate(creg):
                clbit_map[bit] = new_creg[i]

        # Final measurements are deferred until after syndrome extraction
        # (and any Layer B correction) — assumes any `measure` instruction
        # is part of the circuit's final readout, which holds for the
        # "build circuit, then measure" usage this package targets.
        unitary_instructions = [
            inst for inst in circuit.data if inst.operation.name != "measure"
        ]
        measurement_instructions = [
            inst for inst in circuit.data if inst.operation.name == "measure"
        ]

        for inst in unitary_instructions:
            data_qubits = [data_reg[circuit.find_bit(q).index] for q in inst.qubits]
            clbits = [clbit_map[cb] for cb in inst.clbits]
            encoded.append(inst.operation, data_qubits, clbits)

        # Add syndrome extraction for each planned pair, BEFORE final
        # measurement — data qubits are still live, so a correction based
        # on this syndrome can still affect what gets measured.
        anc_idx = 0
        for pair in self.pairs_order:
            qi, qj = pair
            n_anc = ancilla_plan[pair]
            self.pair_first_syndrome_bit[pair] = anc_idx
            for k in range(n_anc):
                # CNOT-based parity check
                encoded.cx(data_reg[qi], anc_reg[anc_idx])
                encoded.cx(data_reg[qj], anc_reg[anc_idx])
                encoded.measure(anc_reg[anc_idx], syndrome_reg[anc_idx])
                encoded.reset(anc_reg[anc_idx])
                anc_idx += 1

        self._pending_measurements = (measurement_instructions, data_reg, clbit_map, circuit)
        logger.info("Encoding complete (measurement deferred). Total qubits: %d",
                    encoded.num_qubits)
        return encoded

    def finalize_measurement(self, encoded_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Attach the original circuit's final measurement, deferred by
        `encode()`. Call this after any Layer B correction has been
        inserted — measurement must come last so the correction can still
        affect the reported result. No-op if `encode()` didn't defer
        anything (e.g. no ancilla was needed).
        """
        if not getattr(self, "_pending_measurements", None):
            return encoded_circuit
        measurement_instructions, data_reg, clbit_map, original_circuit = self._pending_measurements
        for inst in measurement_instructions:
            data_qubits = [
                data_reg[original_circuit.find_bit(q).index] for q in inst.qubits
            ]
            clbits = [clbit_map[cb] for cb in inst.clbits]
            encoded_circuit.append(inst.operation, data_qubits, clbits)
        return encoded_circuit

    def _plan_ancilla(self, circuit: QuantumCircuit) -> dict:
        """
        Decide how many ancilla qubits to allocate per 2Q pair.

        Returns
        -------
        dict: {(q_i, q_j): n_ancilla}
        """
        n_data = circuit.num_qubits
        plan = {}

        for key, err in self.noise_map.items():
            if len(key) != 2:
                continue
            qi, qj = key
            # Only plan for pairs within the circuit's qubit range
            if qi >= n_data or qj >= n_data:
                continue

            if err >= _CRITICAL_NOISE_THRESHOLD:
                n = min(3, self.max_ancilla_per_pair)
            elif err >= _HIGH_NOISE_THRESHOLD:
                n = min(2, self.max_ancilla_per_pair)
            elif err > 0.002:
                n = _DEFAULT_ANCILLA_PER_PAIR
            else:
                n = 0  # low noise pair — skip

            if n > 0:
                plan[key] = n

        return plan

    def ancilla_summary(self, circuit: QuantumCircuit) -> str:
        """Describe the ancilla allocation plan."""
        plan = self._plan_ancilla(circuit)
        if not plan:
            return "No ancilla allocated (circuit noise is below threshold)."
        lines = ["Ancilla allocation plan:"]
        for pair, n in sorted(plan.items(), key=lambda x: -x[1]):
            err = self.noise_map.get(pair, 0.0)
            lines.append(f"  pair {pair}: {n} ancilla  (err={err:.4f})")
        lines.append(f"  Total ancilla: {sum(plan.values())}")
        return "\n".join(lines)
