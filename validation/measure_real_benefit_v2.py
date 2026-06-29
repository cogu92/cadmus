"""
measure_real_benefit_v2.py
---------------------------
Re-measures CADMUS's benefit after the Layer B/C redesign documented in
FINDINGS.md ("Redesign — does it actually work now?"):

  - AdaptiveEncoder now extracts the syndrome *before* the data qubits'
    final measurement (encode() + finalize_measurement()), so a
    mid-circuit correction can actually change the reported result.
  - MidCircuitCorrector (Layer B) now decodes the *full* syndrome pattern
    with a proper repetition-code MWPM graph (fault_ids = data qubit),
    instead of the old "syndrome[k] -> flip data[k % n_data]" heuristic,
    and inserts one if_test branch per non-trivial pattern.
  - HybridDecoder (Layer C) now has `decode_per_shot`, which decodes each
    shot with *its own* syndrome (via `execute_circuit_per_shot`), instead
    of applying one global correction (derived from the *last* syndrome
    sample) to the entire aggregated counts histogram.

measure_real_benefit.py (v1) found: Layer A helps (-57% to -97% TVD),
Layer B is a no-op, Layer C increases TVD ~60x. This script checks
whether the redesign fixes B and C, using the same TVD-vs-ideal
methodology, real noise model, no hardware queue.

Two circuits, deliberately chosen for the information-theoretic split
identified in FINDINGS.md:
  - bell_2q (1 syndrome check): a single ancilla parity bit cannot
    *localize* which of 2 qubits flipped — at best it can push minority
    (zero-ideal-weight) outcomes back into a valid bucket.
  - ghz_3q (2 syndrome checks, complete chain): enough information to
    truly localize a single bit-flip among 3 qubits, the case the
    repetition-code MWPM decoder is designed for.

For each, compares (via real CADMUS.run(), multiple trials for CI):
  RAW < A < A+B < A+B+C
under AerSimulator(noise_model=NoiseModel.from_backend(ibm_kingston)).
"""
import json
import datetime
import statistics

# Run from the repo root with the package installed: pip install -e .

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from cadmus import CADMUS

BACKEND_NAME = "ibm_kingston"
SHOTS = 8192
N_TRIALS = 5


def tvd(counts: dict, ideal: dict, shots: int) -> float:
    keys = set(counts) | set(ideal)
    return 0.5 * sum(
        abs(counts.get(k, 0) / shots - ideal.get(k, 0)) for k in keys
    )


def ideal_distribution(circuit: QuantumCircuit, n_data: int) -> dict:
    sim = AerSimulator()
    qc = circuit.copy()
    qc.measure_all()
    counts = sim.run(qc, shots=20000).result().get_counts()
    total = sum(counts.values())
    return {k.replace(" ", "")[:n_data]: v / total for k, v in counts.items()}


def raw_tvd(qc: QuantumCircuit, noisy_sim, ideal: dict) -> float:
    tc = transpile(qc, noisy_sim, optimization_level=1)
    tc.measure_all()
    counts = noisy_sim.run(tc, shots=SHOTS).result().get_counts()
    counts = {k.replace(" ", ""): v for k, v in counts.items()}
    return tvd(counts, ideal, SHOTS)


def layered_tvd(qc: QuantumCircuit, noisy_sim, ideal: dict, layers: list) -> float:
    cadmus = CADMUS(backend=noisy_sim, layers=layers, verbose=False)
    result = cadmus.run(qc, shots=SHOTS)
    return tvd(result.counts, ideal, SHOTS)


def run_case(name: str, qc: QuantumCircuit, noisy_sim, results: dict):
    print("\n" + "=" * 60)
    print(f"Circuito: {name}  ({qc.num_qubits} qubits)")
    n_data = qc.num_qubits
    ideal = ideal_distribution(qc, n_data)
    print(f"  Ideal: {ideal}")

    configs = {
        "RAW": None,
        "A": ["A"],
        "A+B": ["A", "B"],
        "A+B+C": ["A", "B", "C"],
    }
    trial_results = {k: [] for k in configs}

    for t in range(N_TRIALS):
        row = []
        for label, layers in configs.items():
            if layers is None:
                v = raw_tvd(qc, noisy_sim, ideal)
            else:
                v = layered_tvd(qc, noisy_sim, ideal, layers)
            trial_results[label].append(v)
            row.append(f"{label}={v:.4f}")
        print(f"  trial {t}: " + "  ".join(row))

    means = {k: statistics.mean(v) for k, v in trial_results.items()}
    stdevs = {k: statistics.stdev(v) if len(v) > 1 else 0.0 for k, v in trial_results.items()}
    print(f"  MEANS: " + "  ".join(f"{k}={means[k]:.4f}(+-{stdevs[k]:.4f})" for k in configs))

    results[name] = {
        "n_data": n_data,
        "ideal": ideal,
        "trials": trial_results,
        "means": means,
        "stdevs": stdevs,
        "b_helps_vs_a": means["A"] - means["A+B"],
        "c_helps_vs_b": means["A+B"] - means["A+B+C"],
    }


print("=" * 60)
print("CADMUS v2 — Does the Layer B/C redesign actually help?")
print("=" * 60)

print("\nConectando a IBM Quantum (solo metadata, sin jobs)...")
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
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

results: dict = {}
run_case("bell_2q", bell, noisy_sim, results)
run_case("ghz_3q", ghz3, noisy_sim, results)

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
for name, r in results.items():
    m = r["means"]
    print(f"  {name}: RAW={m['RAW']:.4f}  A={m['A']:.4f}  A+B={m['A+B']:.4f}"
          f"  A+B+C={m['A+B+C']:.4f}")
    print(f"    Layer B vs A: {'AYUDA' if r['b_helps_vs_a'] > 0 else 'EMPEORA'}"
          f" (delta={r['b_helps_vs_a']:+.4f})")
    print(f"    Layer C vs A+B: {'AYUDA' if r['c_helps_vs_b'] > 0 else 'EMPEORA'}"
          f" (delta={r['c_helps_vs_b']:+.4f})")

results["summary_meta"] = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "backend_noise_model": BACKEND_NAME,
    "shots": SHOTS,
    "n_trials": N_TRIALS,
}

out = "resultados_cadmus_real_benefit_v2.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado en {out}")
