"""
confirm_redesign_hardware.py
------------------------------
Real-hardware confirmation of the Layer B/C redesign (see FINDINGS.md
"Redesign"). measure_real_benefit_v2.py already showed, under a real
noise model (no hardware queue), that for ghz_3q (a complete 2-check
chain — enough syndrome information to localize a single bit-flip):

    A=0.0079  A+B=0.0201  A+B+C=0.0069   (TVD vs ideal, mean of 5 trials)

i.e. the full A+B+C pipeline slightly beats Layer A alone, after Layer C
undoes the overhead Layer B's extra gates introduce. This script checks
whether that direction holds on real iron, not just the noise model.

Scope kept tight (job count): one GHZ-3 circuit, two configurations
(A vs A+B+C), each needing its own Layer-A calibration (2**3 = 8 jobs)
plus one execution job — about 18 real-hardware jobs total.
"""
import json
import datetime

# Run from the repo root with the package installed: pip install -e .

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from qvoting.nisq_selector import NISQSelector
from cadmus import CADMUS

CANDIDATE_BACKENDS = ["ibm_fez", "ibm_marrakesh", "ibm_kingston"]
SHOTS = 2048

results: dict = {}

print("=" * 60)
print("CADMUS redesign — real hardware confirmation (ghz_3q)")
print("=" * 60)

print("\nConectando a IBM Quantum...")
service = QiskitRuntimeService()
selector = NISQSelector(service, CANDIDATE_BACKENDS)
try:
    ranked = selector.ranked()
    print(selector.report())
    backend_name = ranked[0][0]
except Exception as e:
    print(f"  Q(t) ranking failed ({e}) -- fallback {CANDIDATE_BACKENDS[0]}")
    backend_name = CANDIDATE_BACKENDS[0]

backend = service.backend(backend_name)
print(f"\n  Backend: {backend_name} ({backend.num_qubits}q)")
results["backend"] = backend_name

ghz3 = QuantumCircuit(3)
ghz3.h(0)
ghz3.cx(0, 1)
ghz3.cx(1, 2)

ideal_sim = AerSimulator()
ideal_qc = ghz3.copy()
ideal_qc.measure_all()
ideal_counts = ideal_sim.run(ideal_qc, shots=20000).result().get_counts()
total = sum(ideal_counts.values())
ideal = {k.replace(" ", ""): v / total for k, v in ideal_counts.items()}
print(f"  Ideal: {ideal}")


def tvd(counts, ideal, shots):
    keys = set(counts) | set(ideal)
    return 0.5 * sum(abs(counts.get(k, 0) / shots - ideal.get(k, 0)) for k in keys)


for label, layers in [("A", ["A"]), ("A+B+C", ["A", "B", "C"])]:
    print("\n" + "=" * 60)
    print(f"Configuracion: layers={layers}")
    cadmus = CADMUS(backend=backend, layers=layers, verbose=False)
    result = cadmus.run(ghz3, shots=SHOTS)
    t = tvd(result.counts, ideal, SHOTS)
    print(f"  layers_applied={result.layers_applied}  TVD={t:.4f}")
    print(f"  counts={result.counts}")
    results[label] = {
        "layers_applied": result.layers_applied,
        "counts": result.counts,
        "tvd": t,
        "viability_score": result.viability_score,
    }

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  A      TVD={results['A']['tvd']:.4f}")
print(f"  A+B+C  TVD={results['A+B+C']['tvd']:.4f}")
better = results["A+B+C"]["tvd"] < results["A"]["tvd"]
print(f"  A+B+C bate a A en hardware real: {'SI' if better else 'NO'}")

results["summary"] = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "backend": backend_name,
    "shots": SHOTS,
    "abc_beats_a": better,
}

out = "resultados_redesign_hardware.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado en {out}")
