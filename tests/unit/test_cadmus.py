"""
Tests unitarios de CADMUS.
Ejecutar con: pytest tests/ -v
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def backend():
    return AerSimulator()


@pytest.fixture
def bell_circuit():
    """Circuito Bell simple para tests."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def ghz_circuit():
    """Circuito GHZ de 3 qubits."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


@pytest.fixture
def simple_noise_map():
    return {
        (0, 1): 0.012,
        (1, 2): 0.008,
        (0,): 0.0003,
        (1,): 0.0004,
        (2,): 0.0002,
    }


# ── Health Check Tests ─────────────────────────────────────────────────

class TestCircuitHealthCheck:

    def test_score_returns_dict(self, backend, bell_circuit):
        from cadmus.profiler.health_check import CircuitHealthCheck
        chk = CircuitHealthCheck(backend)
        result = chk.score(bell_circuit)
        assert isinstance(result, dict)

    def test_viability_in_range(self, backend, bell_circuit):
        from cadmus.profiler.health_check import CircuitHealthCheck
        chk = CircuitHealthCheck(backend)
        result = chk.score(bell_circuit)
        assert 0.0 <= result["viability"] <= 1.0

    def test_all_keys_present(self, backend, bell_circuit):
        from cadmus.profiler.health_check import CircuitHealthCheck
        chk = CircuitHealthCheck(backend)
        result = chk.score(bell_circuit)
        expected_keys = {
            "viability", "depth_penalty", "gate_error_budget",
            "two_qubit_ratio", "tvd_estimate", "recommendation"
        }
        assert expected_keys.issubset(result.keys())

    def test_deeper_circuit_lower_viability(self, backend):
        from cadmus.profiler.health_check import CircuitHealthCheck
        chk = CircuitHealthCheck(backend)

        shallow = QuantumCircuit(2, 2)
        shallow.h(0)
        shallow.measure_all()

        deep = QuantumCircuit(4, 4)
        for _ in range(20):
            deep.cx(0, 1)
            deep.cx(1, 2)
            deep.cx(2, 3)
        deep.measure_all()

        score_shallow = chk.score(shallow)["viability"]
        score_deep = chk.score(deep)["viability"]
        assert score_shallow > score_deep

    def test_recommendation_is_string(self, backend, bell_circuit):
        from cadmus.profiler.health_check import CircuitHealthCheck
        chk = CircuitHealthCheck(backend)
        result = chk.score(bell_circuit)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0


# ── Noise Profiler Tests ───────────────────────────────────────────────

class TestNoiseProfiler:

    def test_profile_returns_dict(self, backend, bell_circuit):
        from cadmus.profiler.noise_profiler import NoiseProfiler
        profiler = NoiseProfiler(backend)
        noise_map = profiler.profile(bell_circuit)
        assert isinstance(noise_map, dict)

    def test_profile_contains_active_pairs(self, backend, bell_circuit):
        from cadmus.profiler.noise_profiler import NoiseProfiler
        profiler = NoiseProfiler(backend)
        noise_map = profiler.profile(bell_circuit)
        # Bell circuit has a CNOT between 0 and 1
        assert (0, 1) in noise_map

    def test_error_rates_in_range(self, backend, ghz_circuit):
        from cadmus.profiler.noise_profiler import NoiseProfiler
        profiler = NoiseProfiler(backend)
        noise_map = profiler.profile(ghz_circuit)
        for rate in noise_map.values():
            assert 0.0 <= rate <= 1.0

    def test_summary_returns_string(self, backend, bell_circuit):
        from cadmus.profiler.noise_profiler import NoiseProfiler
        profiler = NoiseProfiler(backend)
        noise_map = profiler.profile(bell_circuit)
        summary = profiler.summary(noise_map)
        assert isinstance(summary, str)


# ── Adaptive Encoder Tests ─────────────────────────────────────────────

class TestAdaptiveEncoder:

    def test_encode_returns_circuit(self, bell_circuit, simple_noise_map):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        encoder = AdaptiveEncoder(simple_noise_map)
        encoded = encoder.encode(bell_circuit)
        assert isinstance(encoded, QuantumCircuit)

    def test_encoded_has_more_or_equal_qubits(self, bell_circuit, simple_noise_map):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        encoder = AdaptiveEncoder(simple_noise_map)
        encoded = encoder.encode(bell_circuit)
        assert encoded.num_qubits >= bell_circuit.num_qubits

    def test_low_noise_map_no_ancilla(self, bell_circuit):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        low_noise = {(0, 1): 0.0001}
        encoder = AdaptiveEncoder(low_noise)
        encoded = encoder.encode(bell_circuit)
        # Low noise should result in no ancilla
        assert encoded.num_qubits == bell_circuit.num_qubits

    def test_ancilla_summary_string(self, bell_circuit, simple_noise_map):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        encoder = AdaptiveEncoder(simple_noise_map)
        summary = encoder.ancilla_summary(bell_circuit)
        assert isinstance(summary, str)

    def test_encode_defers_measurement(self, simple_noise_map):
        """encode() must extract the syndrome *before* the data qubits are
        measured -- otherwise a Layer B correction can't affect the result
        (see FINDINGS.md). The measured circuit's own measure instructions
        should not appear until finalize_measurement() is called."""
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        encoder = AdaptiveEncoder(simple_noise_map)
        encoded = encoder.encode(qc)

        measure_ops = [inst for inst in encoded.data if inst.operation.name == "measure"]
        # Only the syndrome ancilla measurement(s) should be present yet.
        assert len(measure_ops) == encoder.n_syndrome_bits
        assert encoder._pending_measurements is not None

        finalized = encoder.finalize_measurement(encoded)
        measure_ops_final = [
            inst for inst in finalized.data if inst.operation.name == "measure"
        ]
        assert len(measure_ops_final) == encoder.n_syndrome_bits + 2

    def test_finalize_measurement_noop_without_ancilla(self, bell_circuit):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        low_noise = {(0, 1): 0.0001}
        encoder = AdaptiveEncoder(low_noise)
        encoded = encoder.encode(bell_circuit)
        finalized = encoder.finalize_measurement(encoded)
        assert finalized is encoded

    def test_pairs_order_and_n_data_set(self, bell_circuit, simple_noise_map):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        encoder = AdaptiveEncoder(simple_noise_map)
        encoder.encode(bell_circuit)
        assert encoder.n_data == 2
        assert encoder.pairs_order == [(0, 1)]


# ── Mid-Circuit Corrector Tests ─────────────────────────────────────────

class TestMidCircuitCorrector:

    def test_skips_without_encoder(self, bell_circuit, simple_noise_map):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        from cadmus.correction.mid_circuit import MidCircuitCorrector
        qc = bell_circuit.copy()
        encoder = AdaptiveEncoder(simple_noise_map)
        encoded = encoder.encode(qc)
        corrected = MidCircuitCorrector().apply(encoded, encoder=None)
        assert corrected is encoded

    def test_inserts_branches_for_complete_chain(self):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        from cadmus.correction.mid_circuit import MidCircuitCorrector
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        noise_map = {(0, 1): 0.02, (1, 2): 0.02, (0,): 0.0003, (1,): 0.0003, (2,): 0.0003}
        encoder = AdaptiveEncoder(noise_map)
        encoded = encoder.encode(qc)
        assert encoder.pairs_order == [(0, 1), (1, 2)]

        corrected = MidCircuitCorrector().apply(encoded, encoder)
        n_ops_before = len(encoded.data)
        n_ops_after = len(corrected.data)
        assert n_ops_after > n_ops_before

    def test_skips_incomplete_chain(self):
        from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
        from cadmus.correction.mid_circuit import MidCircuitCorrector
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 2)  # not adjacent -- pair (0,2), not a complete chain
        qc.measure([0, 1, 2], [0, 1, 2])

        noise_map = {(0, 2): 0.02, (0,): 0.0003, (2,): 0.0003}
        encoder = AdaptiveEncoder(noise_map)
        encoded = encoder.encode(qc)
        corrected = MidCircuitCorrector().apply(encoded, encoder)
        assert corrected is encoded


# ── Readout Corrector Tests ────────────────────────────────────────────

class TestReadoutCorrector:

    def test_correct_preserves_total_shots(self, backend):
        from cadmus.correction.readout import ReadoutCorrector
        corrector = ReadoutCorrector(backend, shots_cal=256)
        raw_counts = {"00": 500, "11": 500}
        corrected = corrector.correct(raw_counts)
        total_raw = sum(raw_counts.values())
        total_corrected = sum(corrected.values())
        # Allow small rounding difference
        assert abs(total_corrected - total_raw) <= 5

    def test_correct_returns_dict(self, backend):
        from cadmus.correction.readout import ReadoutCorrector
        corrector = ReadoutCorrector(backend, shots_cal=256)
        raw_counts = {"00": 480, "01": 10, "10": 10, "11": 500}
        corrected = corrector.correct(raw_counts)
        assert isinstance(corrected, dict)

    def test_correct_empty_counts(self, backend):
        from cadmus.correction.readout import ReadoutCorrector
        corrector = ReadoutCorrector(backend, shots_cal=256)
        assert corrector.correct({}) == {}


# ── Hybrid Decoder Tests ───────────────────────────────────────────────

class TestHybridDecoder:

    def test_decode_syndrome_no_error(self, simple_noise_map):
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(simple_noise_map, use_pymatching=False)
        correction = decoder.decode_syndrome([0, 0, 0])
        assert correction == [0, 0, 0]

    def test_decode_counts_returns_dict(self, simple_noise_map):
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(simple_noise_map, use_pymatching=False)
        counts = {"00": 500, "11": 500}
        syndromes = [[0, 0], [0, 0]]
        result = decoder.decode_counts(counts, syndromes)
        assert isinstance(result, dict)

    def test_decode_empty_syndromes(self, simple_noise_map):
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(simple_noise_map, use_pymatching=False)
        counts = {"00": 1000}
        result = decoder.decode_counts(counts, [])
        assert result == counts

    def test_update_prior(self, simple_noise_map):
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(simple_noise_map, use_pymatching=False)
        new_map = {(0, 1): 0.02}
        decoder.update_prior(new_map)
        assert decoder.noise_map == new_map

    def test_is_complete_chain(self):
        from cadmus.decoder.hybrid_decoder import is_complete_chain
        assert is_complete_chain([(0, 1), (1, 2)], 3)
        assert not is_complete_chain([(0, 2)], 3)
        assert not is_complete_chain([(0, 1)], 3)

    def test_build_chain_matching_localizes_single_error(self):
        """3 data qubits / 2 checks: a flip on the *middle* qubit should
        fire both checks and be localized to qubit 1, not 0 or 2."""
        from cadmus.decoder.hybrid_decoder import build_chain_matching
        import numpy as np
        matching = build_chain_matching(3)
        assert list(matching.decode(np.array([0, 0]))) == [0, 0, 0]
        assert list(matching.decode(np.array([1, 1]))) == [0, 1, 0]
        assert list(matching.decode(np.array([1, 0]))) == [1, 0, 0]
        assert list(matching.decode(np.array([0, 1]))) == [0, 0, 1]

    def test_decode_per_shot_corrects_known_flip(self, simple_noise_map):
        """decode_per_shot must flip exactly the qubit decode_syndrome
        localizes, in the reported data bitstring."""
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(
            simple_noise_map, use_pymatching=True, n_data=2, pairs_order=[(0, 1)],
        )
        correction = decoder.decode_syndrome([1])
        assert sum(correction) == 1  # localizes to exactly one qubit

        per_shot = [("10", [1])]
        result = decoder.decode_per_shot(per_shot)
        assert sum(result.values()) == 1
        expected_bits = list("10")
        flip_idx = correction.index(1)
        expected_bits[flip_idx] = "0" if expected_bits[flip_idx] == "1" else "1"
        assert "".join(expected_bits) in result

    def test_decode_per_shot_empty(self, simple_noise_map):
        from cadmus.decoder.hybrid_decoder import HybridDecoder
        decoder = HybridDecoder(simple_noise_map, use_pymatching=False)
        assert decoder.decode_per_shot([]) == {}


# ── Drift Detector Tests ───────────────────────────────────────────────

class TestSyndromeDriftDetector:

    def test_no_drift_with_stable_syndromes(self):
        from cadmus.drift.detector import SyndromeDriftDetector
        detector = SyndromeDriftDetector(window=20, threshold=0.5)
        # Feed identical syndromes — no drift
        for _ in range(60):
            detector.update([0, 0, 0, 0])
        assert not detector.drift_detected()

    def test_drift_detected_on_change(self):
        from cadmus.drift.detector import SyndromeDriftDetector
        detector = SyndromeDriftDetector(window=20, threshold=0.01)
        # Baseline: all zeros
        for _ in range(30):
            detector.update([0, 0, 0, 0])
        # Sudden change: all ones
        for _ in range(30):
            detector.update([1, 1, 1, 1])
        assert detector.drift_detected()

    def test_kl_is_float(self):
        from cadmus.drift.detector import SyndromeDriftDetector
        detector = SyndromeDriftDetector(window=10, threshold=0.1)
        for _ in range(30):
            detector.update([0, 1, 0, 1])
        detector.drift_detected()
        assert isinstance(detector.last_kl, float)

    def test_reset_clears_baseline(self):
        from cadmus.drift.detector import SyndromeDriftDetector
        detector = SyndromeDriftDetector(window=10, threshold=0.1)
        for _ in range(20):
            detector.update([0, 0])
        detector.reset_baseline()
        assert not detector._baseline_set

    def test_summary_string(self):
        from cadmus.drift.detector import SyndromeDriftDetector
        detector = SyndromeDriftDetector()
        summary = detector.summary()
        assert isinstance(summary, str)


# ── Integration Test ───────────────────────────────────────────────────

class TestCADMUSIntegration:

    def test_full_pipeline_bell(self, backend, bell_circuit):
        from cadmus import CADMUS
        cadmus = CADMUS(backend=backend, layers=["A"], verbose=False)
        result = cadmus.run(bell_circuit, shots=512)

        assert isinstance(result.counts, dict)
        assert 0.0 <= result.viability_score <= 1.0
        assert isinstance(result.noise_report, dict)
        assert isinstance(result.syndrome_log, list)
        assert "A" in result.layers_applied

    def test_result_summary(self, backend, bell_circuit):
        from cadmus import CADMUS
        cadmus = CADMUS(backend=backend, layers=["A"])
        result = cadmus.run(bell_circuit, shots=256)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "CADMUS" in summary

    def test_full_pipeline_abc_ghz(self, backend, ghz_circuit):
        """Layers A+B+C end-to-end on a complete-chain (n_checks=2)
        circuit -- the case the repetition-code decoder can actually
        localize errors for. Must run without error and report
        n_data-qubit-wide counts (no leftover syndrome/ancilla bits)."""
        from cadmus import CADMUS
        cadmus = CADMUS(backend=backend, layers=["A", "B", "C"], verbose=False)
        result = cadmus.run(ghz_circuit, shots=512)

        assert set(result.layers_applied) == {"A", "B", "C"}
        assert all(len(k) == 3 for k in result.counts)
        assert all(len(k) == 3 for k in result.raw_counts)
        # Layer A's matrix-inversion rounding can drift the total slightly.
        assert abs(sum(result.counts.values()) - 512) <= 5
