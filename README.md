# CADMUS — Circuit-Aware Dynamic Multilayer Update System

> Adaptive quantum error correction for NISQ hardware: profiles a circuit's
> real noise, encodes it with heterogeneous ancillas, and applies up to
> three correction layers (readout, mid-circuit, decoder).

[![Tests](https://github.com/cogu92/cadmus/actions/workflows/tests.yml/badge.svg)](https://github.com/cogu92/cadmus/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.0+](https://img.shields.io/badge/Qiskit-2.0%2B-6929c4.svg)](https://github.com/Qiskit/qiskit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

---

## Status: experimental

Per the [Qiskit Ecosystem maturity definitions](https://qisk.it/ecosystem-maturity):
this project is **under active development, with APIs subject to breaking
changes**. Not recommended for production use unless you're prepared to
re-validate for your own circuits.

That's not a hedge — it's the literal, measured finding. The mechanism is
real (see [`validation/FINDINGS.md`](validation/FINDINGS.md) for full data,
including real IBM hardware runs), but:

- **Layer A (readout correction) reliably helps** — 57-97% TVD reduction
  vs. raw, confirmed in simulation and on real hardware.
- **The full A+B+C pipeline sometimes beats Layer A alone, sometimes
  doesn't** — confirmed ~2.4x better on real IBM hardware for one 3-qubit
  case, but a qubit-count sweep (2 to 6 qubits) found the effect is **not
  monotonic**: it wins at 4-5 qubits and loses at 2, 3, and 6 in the same
  sweep. Two measurements of the *same* 3-qubit case a few days apart even
  disagreed with each other, just from shot noise and day-to-day hardware
  calibration drift.

Recommendation: use Layer A by default. Only enable B+C if you've measured
a benefit for your specific circuit, backend, and qubit count — don't
assume it generalizes.

---

## What CADMUS does

```
Input Circuit
     │
     ▼
┌─────────────────────────────────────┐  PRE-EXECUTION
│  Health Check  │  Noise Profiler    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐  ENCODING
│  Adaptive QEC Encoder               │
│  (heterogeneous ancillas; syndrome  │
│   extracted before final measure)   │
└─────────────────────────────────────┘
     │
     ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐  ACTIVE CORRECTION
│ Layer B      │ │ (finalize    │ │ Layer C      │
│ Mid-circuit  │ │  measurement)│ │ Decoder      │
└──────────────┘ └──────────────┘ └──────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Layer A — Readout correction       │
└─────────────────────────────────────┘
     │
     ▼
Corrected counts + viability score + noise report + syndrome log
```

Circuit-specific noise profiling (not a global, pre-calibrated noise
model), a repetition-code-style adaptive encoder, and three independent
correction layers you can mix:

- **Layer A — Readout correction**: confusion-matrix inversion calibrated
  on just the circuit's own data qubits.
- **Layer B — Mid-circuit correction**: syndrome extracted *before* the
  final measurement, decoded via a real repetition-code MWPM graph
  (`pymatching`), corrections inserted as `if_test` branches — requires a
  backend with dynamic-circuits support.
- **Layer C — Decoder**: decodes each shot with *its own* syndrome
  (not an aggregate), via the same MWPM graph.

Layer B and C only activate when the circuit's syndrome-check pairs form
a complete 1D chain over the data qubits (one check between every
adjacent pair) — that's the only topology the repetition-code decoder
knows how to localize a single-qubit error in. Otherwise they're skipped
(logged, not guessed).

---

## Quickstart

```bash
git clone https://github.com/cogu92/cadmus.git
pip install -e ./cadmus
```

```python
from qiskit import QuantumCircuit
from cadmus import CADMUS

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

cadmus = CADMUS(backend="aer_simulator", layers=["A"])   # recommended default
result = cadmus.run(qc, shots=1024)

print(result.counts)           # corrected counts (data qubits only)
print(result.viability_score)  # 0.0-1.0 circuit health
print(result.noise_report)     # per-qubit noise map
print(result.syndrome_log)     # syndrome history from encoding (populated
                                # whenever the noise profile warrants an
                                # ancilla, independent of which layers run)
```

Swap `backend="aer_simulator"` for a real `qiskit_ibm_runtime` backend
object to run on IBM Quantum hardware (`backend.run()` was removed from
`qiskit-ibm-runtime` — CADMUS routes through the `SamplerV2` primitive
instead; see `cadmus/utils/executor.py`).

To try the full pipeline (only worth it if you've measured a benefit for
your circuit — see Status above):

```python
cadmus = CADMUS(backend=backend, layers=["A", "B", "C"])
result = cadmus.run(qc, shots=1024)
```

---

## Modules

### `cadmus.profiler` — Circuit Health Check + Noise Profiler

```python
from cadmus.profiler import CircuitHealthCheck, NoiseProfiler

score = CircuitHealthCheck(backend).score(circuit)
# {"viability": 0.73, "depth_penalty": 0.12, "gate_error_budget": 0.09,
#  "two_qubit_ratio": ..., "tvd_estimate": ..., "recommendation": ...}

noise_map = NoiseProfiler(backend).profile(circuit)
# {(0, 1): 0.007, (1, 2): 0.005, (0,): 0.0003, ...}
```

### `cadmus.encoder` — Adaptive QEC Encoder

`encode()` defers the circuit's own final measurement so Layer B's
correction (if used) can still affect the result; call
`finalize_measurement()` once Layer B has had its turn (or immediately if
skipping Layer B). `CADMUS.run()` already does this for you — only call
these directly if you're building a custom pipeline.

```python
from cadmus.encoder import AdaptiveEncoder

encoder = AdaptiveEncoder(noise_map)
encoded = encoder.encode(circuit)              # syndrome extraction, data not yet measured
# ... optionally apply Layer B here ...
encoded = encoder.finalize_measurement(encoded)  # now measure the data qubits
```

### `cadmus.correction` — Layer A + Layer B

```python
from cadmus.correction import ReadoutCorrector, MidCircuitCorrector

corrector = ReadoutCorrector(backend)
corrected_counts = corrector.correct(raw_counts)

# Layer B needs the encoder instance (for pairs_order/n_data — register
# sizes alone don't say which qubits each syndrome bit checks):
corrected_circuit = MidCircuitCorrector().apply(encoded, encoder)
```

### `cadmus.decoder` — Layer C

```python
from cadmus.decoder import HybridDecoder
from cadmus.utils.executor import execute_circuit_per_shot

decoder = HybridDecoder(noise_map, n_data=encoder.n_data, pairs_order=encoder.pairs_order)
counts, syndromes, per_shot = execute_circuit_per_shot(encoded, backend, shots=1024)
decoded_counts = decoder.decode_per_shot(per_shot)   # decodes each shot with its own syndrome
```

### `cadmus.drift` — Syndrome Drift Detector

```python
from cadmus.drift import SyndromeDriftDetector

detector = SyndromeDriftDetector(window=50, threshold=0.15)
detector.update(syndrome)
if detector.drift_detected():
    ...  # trigger recalibration
```

---

## Dependency on QVoting

CADMUS builds on [QVoting](https://github.com/cogu92/qvoting) for the
`SamplerV2`-based IBM execution pattern (`backend.run()` was removed from
`qiskit-ibm-runtime`) and ZNE gate folding.

---

## Validation

All claims above are backed by reproducible data in
[`validation/`](validation/) — scripts, raw logs (before/after each fix),
and JSON results, including real IBM Quantum hardware runs (`ibm_kingston`).
See [`validation/FINDINGS.md`](validation/FINDINGS.md) for the full story:
6 bugs found and fixed (the package could not run on real IBM hardware at
all before this), a Layer B/C redesign, and the qubit-count sweep that
found the non-monotonic benefit described in Status above.

38/38 unit tests pass (`pytest tests/`).

---

## Roadmap

- [x] Phase 1: Circuit Health Check + Noise Profiler
- [x] Phase 2: Adaptive QEC Encoder (syndrome extracted before measurement)
- [x] Phase 3: Layer A — Readout correction (data-qubit-only calibration)
- [x] Phase 4: Layer B — Mid-circuit correction (real repetition-code MWPM)
- [x] Phase 5: Layer C — Per-shot decoder (real repetition-code MWPM)
- [x] Phase 6: Syndrome Drift Detector
- [ ] Characterize *why* the A+B+C benefit is non-monotonic in qubit count
      (overhead-vs-localization tradeoff is a hypothesis, not confirmed)
- [ ] Majority-vote readout cleanup for the redundant ancillas allocated
      to high/critical-noise pairs (currently measured but unused by the
      decoder)
- [ ] PyPI release

---

## Citation

```bibtex
@software{cadmus2026,
  author  = {Corredor Guasca, Nicolas},
  title   = {CADMUS: Circuit-Aware Dynamic Multilayer Update System},
  year    = {2026},
  url     = {https://github.com/cogu92/cadmus},
  version = {0.2.0}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
