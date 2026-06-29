"""
run_cadmus_hardware_validation.py
----------------------------------
Real-hardware integration test for CADMUS (cadmus_framework/), the
adaptive QEC package vendored from cadmus-qec-v0.1.0.

Why this script exists
-----------------------
The upstream README claims "Validated on real IBM hardware" with a
detailed benchmark table (Bell fidelity 85.3% -> 97.3%, TVD 0.147 -> 0.031,
etc.), but `benchmarks/`, `scripts/` and `tests/integration/` ship empty
in the upstream tarball -- there is no code or data behind those numbers.
Separately, `backend.run()` was removed from qiskit-ibm-runtime
(IBMBackendError: "Support for backend.run() has been removed"), which
cadmus's executor.py and readout.py called directly -- so the package
could not execute on real IBM hardware at all before being patched here
(see cadmus_framework/cadmus/utils/executor.py and
cadmus_framework/cadmus/correction/readout.py).

This script re-derives real numbers, scoped to keep hardware job count
small:

  Step 1-3 (local, no hardware job): CircuitHealthCheck, NoiseProfiler,
      AdaptiveEncoder against the real backend's calibration data.
  Step 4  (1 hardware job): submit the adaptively-encoded circuit via
      the patched executor -- proves the SamplerV2 migration works.
  Step 5  (2^n hardware jobs, n = data+ancilla qubits): Layer A
      (ReadoutCorrector) calibration + correction against real counts.
  Step 6  (local): HybridDecoder + SyndromeDriftDetector fed with the
      REAL syndrome history collected in step 4 (not synthetic fixtures).
  Step 7  (Aer, real backend noise model, no hardware job): full
      CADMUS.run(layers=["A","B","C"]) end-to-end, since Layer B
      (mid-circuit) and Layer C (decoder) are marked "in progress" /
      not done in upstream's own roadmap -- noise-model simulation is
      the honest validation scope for unfinished phases.
  Step 8  (1 optional hardware job): full CADMUS.run(layers=["A","B","C"])
      directly on real hardware as a smoke test. Wrapped in try/except;
      failure here is reported, not treated as a script error.
"""
import json
import datetime

# Run from the repo root with the package installed: pip install -e .

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from qvoting.nisq_selector import NISQSelector

from cadmus import CADMUS
from cadmus.profiler.health_check import CircuitHealthCheck
from cadmus.profiler.noise_profiler import NoiseProfiler
from cadmus.encoder.adaptive_encoder import AdaptiveEncoder
from cadmus.correction.readout import ReadoutCorrector
from cadmus.decoder.hybrid_decoder import HybridDecoder
from cadmus.drift.detector import SyndromeDriftDetector
from cadmus.utils.executor import execute_circuit

CANDIDATE_BACKENDS = ["ibm_fez", "ibm_marrakesh", "ibm_kingston"]
SHOTS = 1024
SHOTS_CAL = 256

results: dict = {}

print("=" * 60)
print("CADMUS — Real IBM Hardware Validation")
print("=" * 60)

# ── Connect + pick backend ──────────────────────────────────────────────
print("\nConectando a IBM Quantum...")
service = QiskitRuntimeService()
selector = NISQSelector(service, CANDIDATE_BACKENDS)
try:
    ranked = selector.ranked()
    print(selector.report())
    backend_name = ranked[0][0]
except Exception as e:
    print(f"  Q(t) ranking failed ({e}) -- falling back to {CANDIDATE_BACKENDS[0]}")
    backend_name = CANDIDATE_BACKENDS[0]

backend = service.backend(backend_name)
print(f"\n  Backend seleccionado: {backend_name} ({backend.num_qubits}q)")
results["backend"] = backend_name

# ── Step 1-3: local profiling + encoding ────────────────────────────────
print("\n" + "=" * 60)
print("PASO 1-3: Health check, noise profiling, adaptive encoding (local)")

bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])

health = CircuitHealthCheck(backend)
score_report = health.score(bell)
print(f"  Viability: {score_report['viability']:.4f} -- {score_report['recommendation']}")

profiler = NoiseProfiler(backend)
noise_map = profiler.profile(bell)
print(profiler.summary(noise_map))

encoder = AdaptiveEncoder(noise_map)
encoded = encoder.encode(bell)
print(encoder.ancilla_summary(bell))
print(f"  Encoded circuit: {encoded.num_qubits}q (data=2 + ancilla="
      f"{encoded.num_qubits - 2}), cregs={[c.name for c in encoded.cregs]}")

results["step1_3"] = {
    "viability": score_report,
    "noise_map": {str(k): v for k, v in noise_map.items()},
    "encoded_qubits": encoded.num_qubits,
}

# ── Step 4: submit encoded circuit to real hardware ─────────────────────
print("\n" + "=" * 60)
print("PASO 4: Ejecucion en hardware real (vía SamplerV2 parchado)")
raw_counts, syndromes = execute_circuit(encoded, backend, shots=SHOTS)
print(f"  Outcomes unicos: {len(raw_counts)}  Syndrome entries: {len(syndromes)}")
print(f"  Top counts: {dict(sorted(raw_counts.items(), key=lambda x: -x[1])[:5])}")

results["step4"] = {
    "raw_counts": raw_counts,
    "n_syndrome_entries": len(syndromes),
    "shots": SHOTS,
}

# ── Step 5: Layer A readout correction on real hardware ─────────────────
print("\n" + "=" * 60)
print("PASO 5: Layer A — readout correction (calibracion en hardware real)")
readout = ReadoutCorrector(backend, shots_cal=SHOTS_CAL)
corrected_counts = readout.correct(raw_counts)
n_qubits_cal = readout._n_qubits
print(f"  Qubits calibrados: {n_qubits_cal} ({2**n_qubits_cal} jobs de calibracion)")
print(f"  Corrected top counts: {dict(sorted(corrected_counts.items(), key=lambda x: -x[1])[:5])}")

results["step5"] = {
    "n_qubits_calibrated": n_qubits_cal,
    "n_calibration_jobs": 2 ** n_qubits_cal,
    "corrected_counts": corrected_counts,
}

# ── Step 6: decoder + drift detector on REAL syndrome history ───────────
print("\n" + "=" * 60)
print("PASO 6: HybridDecoder + SyndromeDriftDetector sobre syndromes reales")
if syndromes:
    decoder = HybridDecoder(noise_map, use_pymatching=True)
    decoded_counts = decoder.decode_counts(raw_counts, syndromes)
    print(f"  PyMatching activo: {decoder._pm is not None}")
    print(f"  Decoded outcomes: {len(decoded_counts)}")

    detector = SyndromeDriftDetector(window=min(50, len(syndromes) // 2 or 1), threshold=0.15)
    for s in syndromes:
        detector.update(s)
    drift = detector.drift_detected()
    print(f"  {detector.summary()}")

    results["step6"] = {
        "pymatching_active": decoder._pm is not None,
        "decoded_counts": decoded_counts,
        "drift_detected": drift,
        "last_kl": detector.last_kl,
    }
else:
    print("  Sin syndromes (encoding no agrego ancilla) -- pasos B/C locales omitidos.")
    results["step6"] = {"skipped": "no syndrome register in encoded circuit"}

# ── Step 7: Layers A+B+C end-to-end on Aer with REAL noise model ────────
print("\n" + "=" * 60)
print("PASO 7: CADMUS full pipeline (A+B+C) — Aer con noise model real")
noisy_sim = AerSimulator.from_backend(backend)
cadmus_sim = CADMUS(backend=noisy_sim, layers=["A", "B", "C"], verbose=False)
try:
    sim_result = cadmus_sim.run(bell, shots=SHOTS)
    print(sim_result.summary())
    results["step7"] = {
        "ok": True,
        "layers_applied": sim_result.layers_applied,
        "counts": sim_result.counts,
        "viability_score": sim_result.viability_score,
    }
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    results["step7"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ── Step 8: optional — full pipeline directly on real hardware ──────────
print("\n" + "=" * 60)
print("PASO 8 (opcional): CADMUS full pipeline (A+B+C) directo en hardware real")
cadmus_hw = CADMUS(backend=backend, layers=["A", "B", "C"], verbose=False)
try:
    hw_result = cadmus_hw.run(bell, shots=512)
    print(hw_result.summary())
    results["step8"] = {
        "ok": True,
        "layers_applied": hw_result.layers_applied,
        "counts": hw_result.counts,
        "viability_score": hw_result.viability_score,
    }
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    results["step8"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ── Summary ───────────────────────────────────────────────────────────────
results["summary"] = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    "backend": backend_name,
    "step4_hardware_execution_ok": len(raw_counts) > 0,
    "step5_readout_correction_ok": len(corrected_counts) > 0,
    "step7_full_pipeline_aer_noisemodel_ok": results["step7"]["ok"],
    "step8_full_pipeline_real_hardware_ok": results["step8"]["ok"],
}

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
for k, v in results["summary"].items():
    print(f"  {k}: {v}")

out = "resultados_cadmus_hardware.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nGuardado en {out}")
