"""
cadmus.correction.mid_circuit
------------------------------
Layer B: Mid-circuit active correction.

Inserts conditional Pauli feedback based on the *full* syndrome pattern,
decoded once per pattern with the same repetition-code MWPM graph used by
`cadmus.decoder.hybrid_decoder` (fault_ids = data qubit index). Each of
the 2**n_checks possible syndrome patterns is decoded once at circuit-
build time; patterns whose correction is non-trivial get one `if_test`
branch comparing the syndrome register against that exact pattern.

Requires Qiskit Runtime dynamic circuits support, and requires the
syndrome to be measured *before* the data qubits are finally read out —
see `AdaptiveEncoder.encode()`/`finalize_measurement()` — a correction
applied after the data register is already measured cannot change the
reported result.

Only works when the encoder's syndrome pairs form a complete 1D chain
over the data qubits (one check between every adjacent pair); otherwise
there isn't enough information to localize a single-qubit error and
Layer B is skipped.
"""

from __future__ import annotations

import itertools
import logging

from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr

from cadmus.decoder.hybrid_decoder import build_chain_matching, is_complete_chain

logger = logging.getLogger(__name__)

_MAX_CHECKS_FOR_LOOKUP_TABLE = 12  # 2**12 = 4096 patterns; keeps build time bounded


class MidCircuitCorrector:
    """
    Apply conditional Pauli corrections based on the decoded syndrome
    pattern, inserted before the data qubits are measured.
    """

    def apply(self, encoded_circuit: QuantumCircuit, encoder=None) -> QuantumCircuit:
        """
        Insert mid-circuit conditional corrections into an encoded circuit.

        Parameters
        ----------
        encoded_circuit : QuantumCircuit
            Output of `AdaptiveEncoder.encode()`, *before*
            `finalize_measurement()` — data qubits must still be live.
        encoder : AdaptiveEncoder
            The encoder instance that produced `encoded_circuit`, used to
            read `pairs_order`/`n_data` (register sizes alone don't reveal
            which qubits each syndrome bit checks).

        Returns
        -------
        QuantumCircuit with conditional corrections inserted.
        """
        syndrome_reg = next(
            (c for c in encoded_circuit.cregs if c.name == "s"), None
        )
        if syndrome_reg is None:
            logger.info(
                "No syndrome register found — mid-circuit correction skipped."
            )
            return encoded_circuit

        anc_reg = next((q for q in encoded_circuit.qregs if q.name == "a"), None)
        data_reg = next((q for q in encoded_circuit.qregs if q.name == "d"), None)
        if anc_reg is None or data_reg is None:
            logger.info("No ancilla/data registers — skipping Layer B.")
            return encoded_circuit

        if encoder is None or not getattr(encoder, "pairs_order", None):
            logger.info(
                "No pair/chain info from the encoder — cannot localize "
                "errors from register layout alone. Skipping Layer B."
            )
            return encoded_circuit

        n_data = encoder.n_data
        pairs_order = encoder.pairs_order
        if not is_complete_chain(pairs_order, n_data):
            logger.info(
                "Syndrome pairs %s don't form a complete 1D chain over "
                "%d data qubits — cannot build a repetition-code decoder. "
                "Skipping Layer B.", pairs_order, n_data,
            )
            return encoded_circuit

        n_checks = len(pairs_order)
        if n_checks == 0:
            return encoded_circuit
        if n_checks > _MAX_CHECKS_FOR_LOOKUP_TABLE:
            logger.info(
                "%d syndrome checks exceeds the lookup-table limit (%d) — "
                "skipping Layer B.", n_checks, _MAX_CHECKS_FOR_LOOKUP_TABLE,
            )
            return encoded_circuit

        try:
            matching = build_chain_matching(n_data)
        except ImportError:
            logger.info("PyMatching not installed — skipping Layer B.")
            return encoded_circuit

        # Only the first ancilla of each pair feeds the decoder; extra
        # redundant ancilla (allocated for high/critical-noise pairs) are
        # measured but not yet used for majority-vote syndrome cleanup.
        bit_indices = [encoder.pair_first_syndrome_bit[pair] for pair in pairs_order]

        corrected = encoded_circuit.copy()
        branches = 0
        import numpy as np

        for pattern in itertools.product((0, 1), repeat=n_checks):
            correction = list(matching.decode(np.array(pattern, dtype="uint8")))
            if not any(correction):
                continue
            condition = _pattern_condition(syndrome_reg, bit_indices, pattern)
            with corrected.if_test(condition):
                for q, flip in enumerate(correction):
                    if flip:
                        corrected.x(data_reg[q])
            branches += 1

        logger.info(
            "Layer B: %d/%d syndrome patterns produce a correction "
            "(%d data qubits, %d checks).",
            branches, 2 ** n_checks, n_data, n_checks,
        )
        return corrected


def _pattern_condition(syndrome_reg, bit_indices, pattern):
    """Build a classical expr matching syndrome_reg[bit_indices[i]] ==
    pattern[i] for all i, ANDed together."""
    conditions = [
        expr.equal(syndrome_reg[bit_indices[i]], bool(bit))
        for i, bit in enumerate(pattern)
    ]
    condition = conditions[0]
    for extra in conditions[1:]:
        condition = expr.logic_and(condition, extra)
    return condition
