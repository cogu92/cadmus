# Changelog

## 0.2.0

Six bugs fixed and a Layer B/C redesign — see [`validation/FINDINGS.md`](validation/FINDINGS.md)
for the full data behind every claim below.

### Fixed

- `backend.run()` was removed from `qiskit-ibm-runtime` — the package could
  not execute on real IBM hardware at all. `cadmus/utils/executor.py` now
  routes IBM backends through `SamplerV2`.
- `AdaptiveEncoder.encode()` silently misrouted classical bits when a
  smaller register (the syndrome) coincidentally had a valid index that
  collided with a data creg's global bit index — corrupted measurement
  results (one data bit permanently stuck at 0).
- `CADMUS.run()` never measured data qubits for circuits without explicit
  `.measure()` calls (e.g. the original quickstart example) — the
  auto-`measure_all()` fallback ran too late, after encoding had already
  added classical bits.
- The Aer execution path didn't transpile before `backend.run()` — broke
  against `AerSimulator.from_backend(real_device)` (restricted basis
  gates).
- Syndrome history was recorded once per unique counts outcome instead of
  weighted by shot count, undermining the drift detector and decoder.
- `CircuitHealthCheck._tvd_estimate()` ran its local noisy simulation
  against the circuit already transpiled to the real backend's full
  width (e.g. 156 qubits) — Aer can't truncate idle qubits once a
  per-qubit noise model is attached, so this silently OOM'd and the
  surrounding exception handler returned a hardcoded `0.05` for any real
  backend above ~25-30 qubits.

### Changed

- **Breaking**: `AdaptiveEncoder.encode()` now defers the circuit's own
  final measurement (extracts the syndrome first); call the new
  `finalize_measurement()` to re-attach it. `CADMUS.run()` already does
  this internally.
- **Breaking**: `MidCircuitCorrector.apply()` now takes the `encoder`
  instance as a second argument (register sizes alone don't reveal which
  qubits each syndrome bit checks).
- **Breaking**: `HybridDecoder` gained `n_data`/`pairs_order` constructor
  arguments and a `decode_per_shot()` method, which decodes each shot
  with its own syndrome via a corrected repetition-code MWPM graph
  (`fault_ids` per data qubit). The old `decode_counts()` (one global
  correction derived from the *last* syndrome sample, applied to every
  histogram bucket) is kept for backward compatibility but is no longer
  used internally — it has no real per-shot justification.
- `MidCircuitCorrector` replaced the `syndrome[k] -> flip data[k % n_data]`
  heuristic with a lookup table: every possible syndrome pattern is
  decoded once via the matching graph, and one `if_test` branch is
  emitted per non-trivial pattern. Only runs when the syndrome pairs form
  a complete 1D chain; skips (logged) otherwise.
- `CADMUS.run()`'s Layer A now calibrates and corrects only the data
  qubits, not the combined data+ancilla space — measured to be strictly
  better (the ancilla added calibration overhead with no benefit to data
  fidelity).
- `requires` floor raised to `qiskit>=2.0.0`, `qiskit-aer>=0.15.0`.

### Added

- 11 new unit tests for the encoder/decoder/mid-circuit redesign (38/38
  total).
- `validation/` — real-hardware logs (`ibm_kingston`), before/after data
  for every fix, and the qubit-count sweep showing the redesigned
  Layer B/C's benefit is real but **not monotonic** in qubit count.

## 0.1.0

Initial release (Circuit Health Check, Noise Profiler, Adaptive Encoder,
Layer A, mid-circuit correction stub, decoder stub, drift detector). Not
independently validated against real IBM hardware prior to 0.2.0.
