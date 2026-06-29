"""
cadmus.utils.executor
----------------------
Unified circuit executor for Aer and IBM Quantum backends.
Returns both counts and syndrome measurements.
"""

from __future__ import annotations

import logging
from typing import Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


def execute_circuit(
    circuit: QuantumCircuit,
    backend=None,
    shots: int = 1024,
) -> Tuple[dict, list]:
    """
    Execute a circuit and extract counts + syndrome measurements.

    Parameters
    ----------
    circuit : QuantumCircuit
    backend : AerSimulator or IBM backend
    shots : int

    Returns
    -------
    tuple: (counts dict, syndrome_list)
        counts: measurement outcomes
        syndrome_list: list of syndrome bit arrays, one per shot
    """
    counts, _memory = _run(circuit, backend, shots)
    syndrome_list = _extract_syndromes(counts, circuit)
    logger.debug(
        "Execution complete. Unique outcomes: %d, Syndrome entries: %d",
        len(counts), len(syndrome_list)
    )
    return counts, syndrome_list


def execute_circuit_per_shot(
    circuit: QuantumCircuit,
    backend=None,
    shots: int = 1024,
) -> Tuple[dict, list, list]:
    """
    Like `execute_circuit`, but also returns the raw per-shot (data
    bitstring, syndrome bits) pairs needed for real per-shot decoding
    (Layer C) — aggregated counts lose the pairing between a specific
    syndrome sample and the outcome it actually came from.

    Returns
    -------
    tuple: (counts, syndrome_list, per_shot)
        counts, syndrome_list : same as `execute_circuit`.
        per_shot : list of (data_bitstring, syndrome_bits), one per shot.
            Empty if the circuit has no syndrome register.
    """
    counts, memory = _run(circuit, backend, shots)
    syndrome_list = _extract_syndromes(counts, circuit)
    per_shot = _build_per_shot_pairs(memory, circuit)
    return counts, syndrome_list, per_shot


def _run(circuit: QuantumCircuit, backend, shots: int) -> Tuple[dict, list]:
    """Execute and return (aggregated counts, per-shot memory bitstrings).
    `memory` is empty for IBM backends only if the job genuinely returns
    no shots; for Aer it always matches `shots` in length."""
    if backend is None:
        backend = AerSimulator()

    # Ensure circuit has measurements
    if circuit.num_clbits == 0:
        circuit = circuit.copy()
        circuit.measure_all()

    backend_type = type(backend).__name__
    is_aer = "Aer" in backend_type or "Simulator" in backend_type

    try:
        if is_aer:
            # Transpile first: a plain AerSimulator() accepts arbitrary gates,
            # but AerSimulator.from_backend(real_device) restricts itself to
            # that device's basis gates and rejects circuits still expressed
            # in terms of e.g. 'h'/'cx'.
            tc = transpile(circuit, backend, optimization_level=1)
            job = backend.run(tc, shots=shots, memory=True)
            result = job.result()
            counts = result.get_counts()
            memory = result.get_memory()
        else:
            counts, memory = _run_on_ibm_backend(circuit, backend, shots)
    except Exception as e:
        logger.error("Circuit execution failed: %s", e)
        raise
    return counts, memory


def _run_on_ibm_backend(circuit: QuantumCircuit, backend, shots: int) -> Tuple[dict, list]:
    """
    Execute on a real IBM Quantum backend via the SamplerV2 primitive.

    `BackendV2.run()` was removed from qiskit-ibm-runtime — primitives are
    now the only execution path. Per-shot bitstrings from every classical
    register are re-joined (in declaration order, matching Aer's
    space-separated convention) so multi-register circuits (data + syndrome)
    keep the same joint, shot-correlated counts format on both backends.

    Returns
    -------
    tuple: (counts, memory) — memory is the list of per-shot joint
        bitstrings, same convention as Aer's `result.get_memory()`.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    sampler = Sampler(backend)
    job = sampler.run([isa_circuit], shots=shots)
    result = job.result()
    pub_result = result[0]

    cregs = isa_circuit.cregs
    if len(cregs) == 1:
        bit_array = getattr(pub_result.data, cregs[0].name)
        memory = bit_array.get_bitstrings()
    else:
        # Multiple classical registers: zip per-shot bitstrings to preserve
        # data/syndrome correlation, then join most-recently-declared-first
        # (Aer's convention, e.g. cregs=['s','c'] -> keys "c_bits s_bits").
        per_register_bits = [
            getattr(pub_result.data, creg.name).get_bitstrings() for creg in cregs
        ]
        memory = [
            " ".join(reversed(shot_bits)) for shot_bits in zip(*per_register_bits)
        ]

    counts: dict = {}
    for key in memory:
        counts[key] = counts.get(key, 0) + 1
    return counts, memory


def _syndrome_bits(bitstring: str, n_syndrome_bits: int) -> list[int] | None:
    """Parse the syndrome bits out of a single combined bitstring."""
    parts = bitstring.split(" ")
    if len(parts) >= 2:
        # Last part is typically the first-added register
        # Syndrome register 's' is typically the first classical register
        syndrome_part = parts[-1] if len(parts[-1]) == n_syndrome_bits else parts[0]
    else:
        syndrome_part = bitstring[-n_syndrome_bits:]
    try:
        return [int(b) for b in syndrome_part]
    except (ValueError, IndexError):
        return None


def _extract_syndromes(counts: dict, circuit: QuantumCircuit) -> list:
    """
    Extract syndrome bits from measurement results.

    In Qiskit, when multiple classical registers exist, the result
    bitstring contains all registers concatenated. The syndrome register
    's' is identified and extracted, weighted by shot count so the
    returned list reflects the actual per-shot syndrome distribution
    (needed for drift detection and decoder priors).
    """
    syndrome_register = next(
        (creg for creg in circuit.cregs if creg.name == "s"), None
    )
    if syndrome_register is None:
        return []

    n_syndrome_bits = len(syndrome_register)
    syndromes = []
    for bitstring, count in counts.items():
        syndrome_bits = _syndrome_bits(bitstring, n_syndrome_bits)
        if syndrome_bits is None:
            continue
        # Weight by shot count so the syndrome history reflects the true
        # per-shot distribution, not just the set of unique outcomes.
        syndromes.extend([syndrome_bits] * count)

    return syndromes


def _build_per_shot_pairs(memory: list, circuit: QuantumCircuit) -> list:
    """Build (data_bitstring, syndrome_bits) pairs, one per shot, from the
    raw per-shot memory. data_bitstring is the flat (no separators) data
    register content; syndrome_bits are kept as a list for the decoder."""
    syndrome_register = next(
        (creg for creg in circuit.cregs if creg.name == "s"), None
    )
    data_qreg = next((q for q in circuit.qregs if q.name == "d"), None)
    if syndrome_register is None or data_qreg is None:
        return []

    n_syndrome_bits = len(syndrome_register)
    n_data = len(data_qreg)

    pairs = []
    for bitstring in memory:
        syndrome_bits = _syndrome_bits(bitstring, n_syndrome_bits)
        if syndrome_bits is None:
            continue
        flat = bitstring.replace(" ", "")
        data_bits = flat[:n_data]
        pairs.append((data_bits, syndrome_bits))
    return pairs
