"""
measure_real_benefit.py
------------------------
Does CADMUS's Layer A actually improve fidelity, or does it just "run
without crashing"? FINDINGS.md confirmed the latter for Layers B/C
(mechanically functional, physical benefit unverified). This script
settles Layer A with real TVD numbers, and gives a definitive answer
for Layer B's no-op hypothesis.

Three-way comparison (fair, apples-to-apples):
  (a) RAW       — bare circuit, no CADMUS at all.
  (b) DIRECT-A  — CADMUS's own ReadoutCorrector applied directly to the
                  bare circuit's data qubits (no AdaptiveEncoder, no
                  irrelevant ancilla in the calibration). Isolates
                  "does Layer A's correction algorithm help at all".
  (c) FULL-A    — the full CADMUS(layers=["A"]) pipeline: AdaptiveEncoder
                  adds an ancilla/syndrome qubit, then Layer A calibrates
                  over data+ancilla combined and we marginalize back to
                  data bits. Isolates "what do you actually get from the
                  shipped package, ancilla overhead included".

All three measured as TVD against the noiseless-ideal distribution,
under AerSimulator(noise_model=NoiseModel.from_backend(ibm_kingston))
(real noise model, no hardware queue — uses the already-fetched backend
calibration data, which is a free API call, not a job).

Layer B no-op check: build the encoded circuit, apply MidCircuitCorrector,
and diff the data-register counts before/after — since data qubits are
measured *before* the syndrome-based correction runs (see FINDINGS.md
bug #2/#3 discussion), the correction cannot retroactively change an
already-written classical bit. This is logically certain; this script
demonstrates it empirically with a fixed seed for an exact match.
"""
import json
import datetime

# Run from the repo root with the package installed: pip install -e .

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
from cadmus.correction.readout import ReadoutCorrector
from cadmus.correction.mid_circuit import MidCircuitCorrector
from cadmus.decoder.hybrid_decoder import HybridDecoder
from cadmus.profiler.noise_profiler import NoiseProfiler
from cadmus.utils.executor import execute_circuit

BACKEND_NAME = "ibm_kingston"
SHOTS = 8192
SHOTS_CAL = 1024


def tvd(counts: dict, ideal: dict, shots: int) -> float:
    keys = set(counts) | set(ideal)
    return 0.5 * sum(
        abs(counts.get(k, 0) / shots - ideal.get(k, 0)) for k in keys
    )


def marginalize_data(counts: dict, n_data: int) -> dict:
    """Keep only the first n_data characters (data bits) of each key,
    after stripping register separators."""
    out: dict = {}
    for key, count in counts.items():
        flat = key.replace(" ", "")
        data_bits = flat[:n_data]
        out[data_bits] = out.get(data_bits, 0) + count
    return out


def ideal_distribution(circuit: QuantumCircuit, n_data: int) -> dict:
    sim = AerSimulator()
    qc = circuit.copy()
    qc.measure_all()
    counts = sim.run(qc, shots=20000).result().get_counts()
    total = sum(counts.values())
    return {k.replace(" ", "")[:n_data]: v / total for k, v in counts.items()}


def run_case(name: str, circuits: dict, results: dict, backend, noisy_sim):
    print("\n" + "=" * 60)
    print(f"Circuito: {name}")
    qc = circuits[name]
    n_data = qc.num_qubits

    ideal = ideal_distribution(qc, n_data)
    print(f"  Ideal: {ideal}")

    # (a) RAW
    tc = transpile(qc, noisy_sim, optimization_level=1)
    tc.measure_all()
    raw_counts = noisy_sim.run(tc, shots=SHOTS).result().get_counts()
    raw_counts = {k.replace(" ", ""): v for k, v in raw_counts.items()}
    tvd_raw = tvd(raw_counts, ideal, SHOTS)
    print(f"  (a) RAW          TVD={tvd_raw:.4f}  counts={raw_counts}")

    # (b) DIRECT-A: ReadoutCorrector straight on the bare circuit
    readout_direct = ReadoutCorrector(noisy_sim, shots_cal=SHOTS_CAL)
    corrected_direct = readout_direct.correct(dict(raw_counts))
    tvd_direct = tvd(corrected_direct, ideal, SHOTS)
    print(f"  (b) DIRECT-A     TVD={tvd_direct:.4f}  counts={corrected_direct}")

    # (c) FULL-A: AdaptiveEncoder + Layer A on combined data+ancilla
    # NoiseProfiler needs backend.properties() for real calibration data —
    # pass the real IBM backend (a free metadata call), not the local sim.
    profiler = NoiseProfiler(backend)
    noise_map = profiler.profile(qc)
    encoder = AdaptiveEncoder(noise_map)
    qc_meas = qc.copy()
    qc_meas.measure_all()
    encoded = encoder.encode(qc_meas)
    n_ancilla = encoded.num_qubits - n_data
    full_raw_counts, _ = execute_circuit(encoded, noisy_sim, shots=SHOTS)
    readout_full = ReadoutCorrector(noisy_sim, shots_cal=SHOTS_CAL)
    full_corrected = readout_full.correct(full_raw_counts)
    full_corrected_marg = marginalize_data(full_corrected, n_data)
    tvd_full = tvd(full_corrected_marg, ideal, SHOTS)
    print(f"  (c) FULL-A       TVD={tvd_full:.4f}  ancilla={n_ancilla}  "
          f"counts={full_corrected_marg}")

    results[name] = {
        "n_data_qubits": n_data,
        "ideal": ideal,
        "tvd_raw": tvd_raw,
        "tvd_direct_layerA": tvd_direct,
        "tvd_full_cadmus_layerA": tvd_full,
        "direct_vs_raw_improvement_pct": (tvd_raw - tvd_direct) / tvd_raw * 100 if tvd_raw else None,
        "full_vs_direct_delta": tvd_full - tvd_direct,
        "n_ancilla_added": n_ancilla,
    }


print("=" * 60)
print("CADMUS — Does Layer A actually help? (TVD vs ideal, real noise model)")
print("=" * 60)

print("\nConectando a IBM Quantum (solo metadata, sin jobs)...")
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)

# AerSimulator.from_backend(backend) inherits the device's full qubit count
# (e.g. 156). transpile() against it expands small circuits to that width,
# and Aer can't auto-truncate idle qubits once a per-qubit noise model is
# attached -> OOM. A plain simulator with just the noise model attached
# keeps circuits at their own (small) qubit count.
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(noise_model=noise_model)
print(f"  Backend: {BACKEND_NAME}")

bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)

ghz3 = QuantumCircuit(3)
ghz3.h(0)
ghz3.cx(0, 1)
ghz3.cx(1, 2)

circuits = {"bell_2q": bell, "ghz_3q": ghz3}
results: dict = {}

for name in circuits:
    run_case(name, circuits, results, backend, noisy_sim)

# ── Layer B no-op check ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Layer B no-op check (mid-circuit correction tras medir los datos)")
noise_map_b = {(0, 1): 0.01, (0,): 0.0003, (1,): 0.0003}
encoder_b = AdaptiveEncoder(noise_map_b)
qc_b = bell.copy()
qc_b.measure_all()
encoded_b = encoder_b.encode(qc_b)
corrected_b = MidCircuitCorrector().apply(encoded_b)

sim_seeded = AerSimulator(seed_simulator=42)
counts_before = sim_seeded.run(encoded_b, shots=4096, seed_simulator=42).result().get_counts()
counts_after = sim_seeded.run(corrected_b, shots=4096, seed_simulator=42).result().get_counts()
n_data_b = bell.num_qubits
marg_before = marginalize_data({k.replace(" ", ""): v for k, v in counts_before.items()}, n_data_b)
marg_after = marginalize_data({k.replace(" ", ""): v for k, v in counts_after.items()}, n_data_b)
identical = marg_before == marg_after
print(f"  Data marginal SIN Layer B: {marg_before}")
print(f"  Data marginal CON Layer B: {marg_after}")
print(f"  Identicas (mismo seed): {identical}")

results["layer_b_noop_check"] = {
    "data_marginal_before": marg_before,
    "data_marginal_after": marg_after,
    "identical": identical,
}

# ── Layer C check: does the decoder help, hurt, or no-op? ───────────────
print("\n" + "=" * 60)
print("Layer C check (HybridDecoder) — TVD antes/despues, 5 repeticiones")
qc_c = bell.copy()
qc_c.measure_all()
profiler_c = NoiseProfiler(backend)
noise_map_c = profiler_c.profile(bell)
ideal_bell = ideal_distribution(bell, 2)

layer_c_trials = []
for t in range(5):
    encoded_c = AdaptiveEncoder(noise_map_c).encode(qc_c)
    raw_c, syndromes_c = execute_circuit(encoded_c, noisy_sim, shots=SHOTS)
    decoder = HybridDecoder(noise_map_c, use_pymatching=True)
    decoded_c = decoder.decode_counts(dict(raw_c), syndromes_c)

    tvd_before = tvd(marginalize_data(raw_c, 2), ideal_bell, SHOTS)
    tvd_after = tvd(marginalize_data(decoded_c, 2), ideal_bell, SHOTS)
    print(f"  trial {t}: TVD raw={tvd_before:.4f}  TVD post-LayerC={tvd_after:.4f}"
          f"  last_syndrome={syndromes_c[-1]}")
    layer_c_trials.append({"tvd_before": tvd_before, "tvd_after": tvd_after,
                            "last_syndrome": syndromes_c[-1]})

results["layer_c_check"] = {
    "trials": layer_c_trials,
    "mean_tvd_before": sum(t["tvd_before"] for t in layer_c_trials) / len(layer_c_trials),
    "mean_tvd_after": sum(t["tvd_after"] for t in layer_c_trials) / len(layer_c_trials),
}

# ── Resumen ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
for name in circuits:
    r = results[name]
    print(f"  {name}: RAW={r['tvd_raw']:.4f}  DIRECT-A={r['tvd_direct_layerA']:.4f}"
          f"  FULL-A={r['tvd_full_cadmus_layerA']:.4f}"
          f"  (DIRECT-A mejora raw {r['direct_vs_raw_improvement_pct']:.1f}%,"
          f" FULL-A vs DIRECT-A delta={r['full_vs_direct_delta']:+.4f})")
print(f"  Layer B no-op confirmado: {results['layer_b_noop_check']['identical']}")
print(f"  Layer C: TVD media antes={results['layer_c_check']['mean_tvd_before']:.4f}"
      f"  despues={results['layer_c_check']['mean_tvd_after']:.4f}"
      f"  (empeora si > antes)")

results["summary_meta"] = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "backend_noise_model": BACKEND_NAME,
    "shots": SHOTS,
}

out = "resultados_cadmus_real_benefit.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado en {out}")
