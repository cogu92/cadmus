"""
sweep_qubit_count.py
----------------------
Does the A+B+C-beats-A advantage (confirmed for ghz_3q in
measure_real_benefit_v2.py / confirm_redesign_hardware.py) grow, shrink,
or plateau as the qubit count grows?

Two competing effects as GHZ-n grows:
  - More checks (n-1) = more localization power for the repetition-code
    decoder (Layer C) -- in principle, more error to find AND more
    ability to find it.
  - More checks also means more Layer-B syndrome-extraction gates (2
    extra CX per check) -- real overhead that scales with n, and a 1D
    repetition decoder only reliably corrects *single* isolated flips
    per shot; multiple simultaneous flips (likelier as the circuit gets
    deeper) aren't its design point.

Sweeps GHZ-n for n in N_RANGE, same TVD-vs-ideal methodology as
measure_real_benefit_v2.py, noise model from ibm_kingston, no hardware
queue. Real-hardware confirmation for one representative larger n is a
separate, explicit follow-up (job count grows as 2**n for Layer A's
calibration alone).
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
SHOTS = 4096
N_TRIALS = 3
N_RANGE = [2, 3, 4, 5, 6]


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


def ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


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


print("=" * 60)
print("CADMUS — A+B+C vs A as qubit count grows (GHZ-n, noise model real)")
print("=" * 60)

print("\nConectando a IBM Quantum (solo metadata, sin jobs)...")
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(noise_model=noise_model)
print(f"  Backend: {BACKEND_NAME}")

results: dict = {}

for n in N_RANGE:
    print("\n" + "=" * 60)
    print(f"GHZ-{n} ({n} qubits, {n - 1} checks)")
    qc = ghz(n)
    ideal = ideal_distribution(qc, n)

    configs = {"RAW": None, "A": ["A"], "A+B": ["A", "B"], "A+B+C": ["A", "B", "C"]}
    trial_results = {k: [] for k in configs}

    for t in range(N_TRIALS):
        row = []
        for label, layers in configs.items():
            v = raw_tvd(qc, noisy_sim, ideal) if layers is None else layered_tvd(qc, noisy_sim, ideal, layers)
            trial_results[label].append(v)
            row.append(f"{label}={v:.4f}")
        print(f"  trial {t}: " + "  ".join(row))

    means = {k: statistics.mean(v) for k, v in trial_results.items()}
    stdevs = {k: statistics.stdev(v) if len(v) > 1 else 0.0 for k, v in trial_results.items()}
    abc_vs_a = means["A"] - means["A+B+C"]
    print(f"  MEANS: " + "  ".join(f"{k}={means[k]:.4f}" for k in configs))
    print(f"  A+B+C vs A: {'MEJOR' if abc_vs_a > 0 else 'PEOR'} (delta={abc_vs_a:+.4f}, "
          f"{abc_vs_a / means['A'] * 100:+.1f}%)")

    results[f"ghz_{n}"] = {
        "n_data": n, "n_checks": n - 1, "trials": trial_results,
        "means": means, "stdevs": stdevs,
        "abc_vs_a_delta": abc_vs_a,
        "abc_vs_a_pct": abc_vs_a / means["A"] * 100 if means["A"] else None,
    }

print("\n" + "=" * 60)
print("RESUMEN — A+B+C vs A en funcion de n")
print("=" * 60)
for n in N_RANGE:
    r = results[f"ghz_{n}"]
    print(f"  n={n} ({r['n_checks']} checks): A={r['means']['A']:.4f}  "
          f"A+B+C={r['means']['A+B+C']:.4f}  delta={r['abc_vs_a_delta']:+.4f} "
          f"({r['abc_vs_a_pct']:+.1f}%)")

results["summary_meta"] = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "backend_noise_model": BACKEND_NAME,
    "shots": SHOTS,
    "n_trials": N_TRIALS,
    "n_range": N_RANGE,
}

out = "resultados_sweep_qubit_count.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado en {out}")
