# CADMUS QEC — validation findings

Everything below is backed by the scripts, raw logs, and JSON results in
this folder. Real IBM hardware runs used `ibm_kingston`.

## Estado al recibir el repo (v0.1.0)

El README original afirmaba "Validated on real IBM hardware: ✅" con una
tabla de benchmarks (Bell fidelity 85.3%→97.3%, TVD 0.147→0.031, logical
error 2.1e-3→8.4e-4). **No había nada detrás de esos números**:
`benchmarks/`, `scripts/` y `tests/integration/` venían vacíos en el
tarball original — solo `tests/integration/__init__.py`. El roadmap
propio del README marcaba las Fases 4-7 (Layer B, Layer C, drift
detector, integración completa) como no terminadas, pero el código ya
las implementaba (sin validar).

## Bug crítico #1 — `backend.run()` fue removido

`qiskit-ibm-runtime` eliminó `IBMBackend.run()` (mensaje real al
invocarlo: *"Support for backend.run() has been removed. ... migrate to
the primitives interface"*). `cadmus/utils/executor.py` y
`cadmus/correction/readout.py` llamaban a `self.backend.run(...)`
directamente — **CADMUS no podía ejecutar en hardware real de IBM en
absoluto**, con ninguna versión reciente de `qiskit-ibm-runtime`, antes
de este fix.

**Fix:** `execute_circuit()` ahora despacha a `SamplerV2` para backends
IBM, y `ReadoutCorrector.calibrate()` reusa `execute_circuit()` en vez de
llamar a `backend.run()` directo. Para circuitos con múltiples
registros clásicos (datos + síndrome), se reconstruyen los bitstrings
conjuntos por shot (`get_bitstrings()` por registro, zip, join en orden
de declaración inverso) para preservar la correlación dato/síndrome.

## Bug #2 — `AdaptiveEncoder` corrompía el registro de datos

`encode()` mapeaba cada clbit original probando, en orden, cada `creg`
del circuito codificado y usando el **índice global** del bit en el
circuito original como índice dentro de ese creg, descartando
`IndexError` con un `try/except` silencioso. Para un Bell circuit (creg
`c` de 2 bits) más 1 ancilla (`syndrome_reg` de 1 bit), `measure(d0,
c[0])` probaba primero `syndrome_reg[0]` — que **sí existe** (tamaño 1,
índice 0 válido) — y quedaba mal enrutado ahí en vez de a `c[0]`. El
resultado: `c[0]` nunca se escribía y quedaba fijo en 0 para todos los
shots, mientras el síndrome real lo sobreescribía después.

Confirmado con la primera corrida en hardware real (`ibm_kingston`,
`run_log_before_fixes.txt`): `raw_counts = {'10 0': 501, '00 0': 487,
'10 1': 26, '00 1': 10}` — el bit derecho de `c` nunca es `1`, rompiendo
por completo la correlación de Bell esperada.

**Fix:** se construye un mapeo explícito clbit→clbit por posición
dentro del mismo registro nombrado, sin probar registros de tamaño
incorrecto. Confirmado en simulador (`'11 00'`/`'00 00'` ~50/50, sin
`'01'`/`'10'`, correlación de Bell correcta) y re-confirmado en
`ibm_kingston` (`run_log_after_fixes.txt`): `{'00 0': 503, '11 0': 483,
'11 1': 19, '00 1': 7, '10 1': 7}` — ahora domina `00`/`11` como debe
ser, con la dispersión minoritaria explicada por ruido real de hardware.

## Bug #3 — el circuito de quickstart del propio README nunca medía

El ejemplo de quickstart no llamaba `.measure()` (`qc.h(0); qc.cx(0,1);
qc.cx(1,2)`). `execute_circuit()` solo llamaba `measure_all()` si
`circuit.num_clbits == 0` — pero para entonces el circuito ya había
pasado por `AdaptiveEncoder`, que le agrega su propio registro de
síndrome, así que `num_clbits` ya no era 0 y el chequeo nunca disparaba.
Los qubits de datos quedaban sin medir; `result.counts` solo contenía
bits de síndrome.

**Fix:** el auto-`measure_all()` se movió al inicio de `CADMUS.run()`,
antes de profiling/encoding. Confirmado en simulador: el circuito de
quickstart (GHZ de 3 qubits sin measure) da `{'000': 513, '111': 487}` —
la correlación GHZ esperada.

## Bug #4 — el path de Aer no transpilaba

`execute_circuit()` llamaba `backend.run(circuit, ...)` sin transpilar
primero. Funciona con `AerSimulator()` plano (acepta cualquier puerta),
pero falla con `AerSimulator.from_backend(backend_real)` (restringe el
basis gate set al del chip real) — exactamente el patrón recomendado
para testear con modelo de ruido realista. Error real observado:
`AerError: 'unknown instruction: h'`.

**Fix:** se agregó `transpile(circuit, backend, optimization_level=1)`
antes de `backend.run()` en el path de Aer.

## Bug #5 (menor) — síndromes no pesados por shot

`_extract_syndromes()` agregaba un síndrome por **outcome único**, no
por shot — con 1024 shots y 4 outcomes únicos, el detector de drift y el
decoder veían solo 4 muestras en vez de 1024. Fix: se repite cada
síndrome `count` veces.

## Validación en hardware real — fixes 1-5

Backend: `ibm_kingston` (156q, Heron r2, Q(t)=1.00 en el ranking del
NISQ Health Monitor). Circuito: Bell de 2 qubits. Script:
[`run_cadmus_hardware_validation.py`](run_cadmus_hardware_validation.py).

| Paso | Qué prueba | Resultado |
|---|---|---|
| 1-3 | Health check + noise profiler + encoder (local, usa `backend.properties()` real) | Viability 0.971 "Excellent"; 1 ancilla asignada al par (0,1), err=0.70% |
| 4 | Ejecución del circuito codificado en hardware real vía SamplerV2 | OK — `{'00 0':503,'11 0':483,...}`, correlación de Bell correcta, 1024 entradas de síndrome (=shots) |
| 5 | Layer A: calibración (8 jobs reales, 2³ estados base) + corrección | OK — `{'000':519,'110':495,...}` |
| 6 | HybridDecoder + SyndromeDriftDetector sobre síndromes reales | OK — PyMatching activo, decode sin errores |
| 7 | Pipeline completo A+B+C en Aer con noise model real de `ibm_kingston` | OK (tras fix #4; antes fallaba) |
| 8 | Pipeline completo A+B+C directo en hardware real (incluye Layer B con `if_test` dinámico) | **OK** — corrió sin error en hardware real |

Detalle completo: `resultados_cadmus_hardware_after_fixes.json`,
`run_log_after_fixes.txt` (después de los 5 fixes) vs
`run_log_before_fixes.txt` (antes, muestra el bug #2 en crudo).

## Bug #6 — `CircuitHealthCheck` falseaba el TVD contra backends reales

`_tvd_estimate()` reutilizaba `tc` (el circuito ya transpilado contra el
backend real completo, ej. 156 qubits) para una simulación local en Aer
con noise model adjunto. Una vez el noise model está adjunto, Aer ya no
puede truncar qubits ociosos y trata de reservar estado para el chip
entero → `OOM` ("a circuit requires more memory than max_memory_mb"). El
`except Exception` envolvente lo atrapaba en silencio y devolvía **un
valor constante `0.05`** — cualquier "Viability: 0.97 Excellent" contra
un backend real razonablemente grande tenía ese sub-componente (15% del
score) inventado, no medido.

**Fix:** la simulación local usa el circuito original (sin expandir a
las qubits del dispositivo completo), transpilado solo a los gates
nativos del noise model — confirmado: ahora da `tvd_estimate=0.0215`
real en vez del `0.05` fijo, sin romper nada.

## ¿CADMUS realmente mejora la fidelidad? (`measure_real_benefit.py`)

Comparación de tres formas, contra distribución ideal (TVD), bajo
`AerSimulator(noise_model=NoiseModel.from_backend(ibm_kingston))`
(modelo de ruido real, sin cola de hardware — 8192 shots, Bell-2q y
GHZ-3q, sin promediar entre corridas):

| | Bell-2q | GHZ-3q |
|---|---|---|
| RAW (sin CADMUS) | 0.0183 | 0.0195 |
| DIRECT-A (Layer A solo en los qubits de datos, sin ancilla) | **0.0005** (−97%) | **0.0084** (−57%) |
| FULL-A (paquete shipped: encoder agrega 1-2 ancillas + Layer A sobre el combinado) | 0.0031 | 0.0103 |

**Layer A funciona — el algoritmo de corrección de readout reduce TVD
de verdad (57-97%).** Pero la versión que de verdad enviaba el paquete
(`FULL-A`, con el ancilla de síndrome metido en la calibración) quedaba
sistemáticamente un poco *peor* que aplicar la misma corrección
directamente sobre los 2-3 qubits de datos (`DIRECT-A`) — el ancilla
extra no aporta nada a la fidelidad de los datos y solo añade ruido de
calibración (matriz más grande, mismo `shots_cal`). Sigue siendo mucho
mejor que no corregir nada.

**Layer B confirmado matemáticamente como no-op** (test con seed fija,
`AerSimulator(seed_simulator=42)`): los qubits de datos se medían *antes*
de que el síndrome decidiera la corrección, así que el `if_test` + `x`
posterior no podía alterar retroactivamente el bit clásico ya escrito.
Resultado antes/después de Layer B: **bit por bit idéntico**.

**Layer C (`HybridDecoder`) no era un no-op — empeoraba el resultado
drásticamente.** 5 repeticiones, Bell-2q, 8192 shots: TVD pasaba de
~0.016-0.025 (ya bueno) a **~0.98** después del decoder — prácticamente
el peor TVD posible. Causa: `decode_counts()` recibía `counts` ya
agregados (un histograma, no shots individuales) pero solo conocía
`syndromes[-1]` (la última muestra de la lista) — aplicaba **una única
corrección global, derivada de esa muestra, a TODAS las claves del
histograma por igual**, sin ninguna correspondencia real entre síndrome
y resultado por shot. Para Bell (correlación `q0==q1`), voltear
globalmente un bit movía toda la masa de `'00'/'11'` a `'01'/'10'` —
exactamente las claves que la distribución ideal asigna ~0 probabilidad.
Esto era estructural, no un caso límite: por diseño, `decode_counts()`
no puede funcionar correctamente sobre counts ya agregados.

## Veredicto (primera iteración, antes del rediseño)

| Capa | Roadmap original | Corre en hardware real | Mejora la fidelidad |
|---|---|---|---|
| A — Readout correction | "done" | Sí | **Sí** (pero el ancilla del encoder no ayuda; corrección directa es mejor) |
| B — Mid-circuit | "in progress" | Sí (sin error) | **No — no-op matemático**, nunca puede ayudar tal como estaba diseñado |
| C — Hybrid decoder | "in progress" | Sí (sin error) | **No — empeora ~60x el TVD**, bug estructural (usaba solo el último síndrome para todo el histograma) |

A este punto, B y C necesitaban un rediseño de arquitectura (síndrome
extraído *antes* del colapso de datos, decodificación por shot en vez de
por histograma agregado) para ser algo más que cosmético. Eso es el
rediseño de la v0.2.0.

## Rediseño (v0.2.0) — Layer B y C arreglados de verdad

**`encoder/adaptive_encoder.py`** — `encode()` ahora **difiere** la
medición final del circuito original: copia la parte unitaria, extrae el
síndrome (los qubits de datos siguen vivos), y solo deja la medición
final pendiente. `finalize_measurement()` la reaplica después de que
Layer B tuvo su oportunidad de corregir. Sin esto, ninguna corrección
mid-circuit puede afectar el resultado. El encoder también expone
`pairs_order`/`n_data`/`pair_first_syndrome_bit` — sin esto, ni el
corrector ni el decoder pueden saber qué qubits checa cada bit de
síndrome (los tamaños de registro no lo dicen).

**`decoder/hybrid_decoder.py`** — el grafo de MWPM estaba mal construido
(trataba los bits de síndrome como nodos de una cadena, en vez de los
qubits de datos como aristas entre nodos-síndrome). Nuevo
`build_chain_matching(n_data)`: cada qubit de dato es una arista con
`fault_ids=qubit_index` (arista de frontera para los 2 qubits extremos,
arista interna para el resto) — verificado con pymatching que localiza
correctamente un flip en un qubit interior dado el patrón de síndrome de
sus 2 checks vecinos. Nuevo `decode_per_shot()`: decodifica cada shot
con *su propio* síndrome (no el último de la lista), usando los pares
(dato, síndrome) reales por shot.

**`utils/executor.py`** — nueva `execute_circuit_per_shot()`: usa
`memory=True` (Aer) / `get_bitstrings()` por registro (IBM SamplerV2)
para devolver el par (bitstring de datos, síndrome) de cada shot
individual — la información que se perdía al agregar todo en `counts`
antes de decodificar.

**`correction/mid_circuit.py`** — ya no usa la heurística
`syndrome[k] → flip data[k % n_data]`. Ahora precalcula, con el mismo
grafo de matching, la corrección para cada uno de los `2**n_checks`
patrones de síndrome posibles, y emite un `if_test` por patrón con
corrección no trivial — antes de la medición final (gracias al cambio en
el encoder). Solo actúa si los pares de síndrome forman una cadena 1D
completa (`(0,1),(1,2),...`); si no, se salta Layer B explícitamente en
vez de adivinar.

**`core.py`** — Layer A ahora calibra y corrige *solo* los qubits de
datos (no el combinado datos+ancilla) — el hallazgo de la primera
iteración (el ancilla extra en la calibración solo añade ruido) ya está
incorporado por defecto.

27 tests existentes siguen pasando; se agregaron 11 tests nuevos
(`encode`/`finalize_measurement` diferidos, MWPM localiza el qubit
correcto, `decode_per_shot` corrige el bit correcto, Layer B se salta
cadenas incompletas, pipeline A+B+C de punta a punta) — 38/38 pasan.

### ¿Ahora sí ayuda? (`measure_real_benefit_v2.py`, noise model real, 5 trials)

| | RAW | A | A+B | A+B+C |
|---|---|---|---|---|
| bell_2q (1 check, no localiza) | 0.0180 | 0.0063 | 0.0107 (peor) | 0.0070 |
| ghz_3q (2 checks, localiza) | 0.0238 | 0.0079 | 0.0201 (peor) | **0.0069 (mejor que A)** |

Layer B solo (el circuito extra de síndrome) **siempre empeora** — las
compuertas de 2 qubits adicionales para extraer el síndrome introducen
más ruido real del que la corrección (limitada o nula, según el caso)
alcanza a compensar. Layer C (puro post-procesamiento, sin compuertas
extra) **siempre ayuda** sobre A+B, y para `ghz_3q` (donde con 2 checks
sí se puede localizar un flip de un solo qubit) el paquete completo
A+B+C termina **ganándole a Layer A solo**.

### Confirmación en hardware real (`confirm_redesign_hardware.py`, `ibm_kingston`)

| Config | TVD |
|---|---|
| A | 0.0691 |
| A+B+C | **0.0284** |

**A+B+C es ~2.4x mejor que A solo, en hardware real** — confirma la
dirección de la simulación con modelo de ruido. `ibm_kingston`, GHZ-3q,
2048 shots, Q(t)=1.00 ese día.

## ¿Y para más qubits? (`sweep_qubit_count.py`, GHZ-2 a GHZ-6, 3 trials c/u)

| n (qubits) | checks | A | A+B+C | A+B+C vs A |
|---|---|---|---|---|
| 2 | 1 | 0.0036 | 0.0068 | PEOR (-88%) |
| 3 | 2 | 0.0057 | 0.0079 | PEOR (-40%) |
| 4 | 3 | 0.0146 | 0.0093 | MEJOR (+36%) |
| 5 | 4 | 0.0155 | 0.0105 | MEJOR (+32%) |
| 6 | 5 | 0.0162 | 0.0205 | PEOR (-27%) |

**No es una tendencia limpia.** La fila n=3 de esta corrida (PEOR) además
**contradice** la de `measure_real_benefit_v2.py` un par de días antes
(MEJOR, +12.7%) — con solo 3-5 trials por punto, el ruido estadístico de
shots es del mismo orden que el efecto que se está midiendo, y el
calibrado real de `ibm_kingston` (T1/T2/error de compuerta) deriva de un
día a otro, así que "noise model real" no es una constante: dos
mediciones en días distintos están comparando contra ruido de fondo
ligeramente distinto. No alcanza para afirmar "n=4-5 es el punto dulce" —
alcanza para decir que el beneficio de A+B+C **no crece monótonamente**
con n, y que para conclusiones robustas por tamaño de circuito harían
falta muchos más trials (y probablemente promediar sobre varios días) de
los que se corrieron aquí.

Mecanismo plausible detrás de la no-monotonía: Layer B agrega `2*(n-1)`
compuertas de 2 qubits reales (overhead que crece linealmente con n, y
cada compuerta de 2 qubits en hardware real es la fuente de error
dominante), mientras que el decoder de cadena 1D solo corrige bien
errores de **un solo qubit por shot** — a medida que el circuito crece
(más profundidad, más tiempo, más probabilidad de 2+ errores
simultáneos en el mismo shot), el supuesto de "un solo error" del
decoder se vuelve menos válido. Ninguno de estos efectos se midió por
separado — es una hipótesis consistente con los datos, no una conclusión
confirmada.

## Veredicto final

CADMUS, después de 6 bugs corregidos y el rediseño de Layer B/C, **corre
en hardware real de IBM y puede mejorar la fidelidad de verdad** — no
solo "sin crashear". Layer A solo (corrección directa de datos) es una
mejora sólida, barata, y consistente en todos los tamaños probados (2-6
qubits). El paquete completo A+B+C *a veces* la mejora más todavía
(confirmado en hardware real para GHZ-3q un día específico, y en
simulación para n=4,5), pero el efecto no es monótono ni garantizado por
tamaño de circuito. Para producción, la recomendación honesta es:
**Layer A solo por defecto**, y activar B+C solo si se mide (con
suficientes trials, en el circuito y backend específicos que se van a
usar) que efectivamente ayuda ahí.

## Files in this folder

- `run_cadmus_hardware_validation.py` + `run_log_before_fixes.txt` /
  `run_log_after_fixes.txt` + `resultados_cadmus_hardware_after_fixes.json`
  — validation of fixes 1-5 on real hardware.
- `measure_real_benefit.py` + `resultados_cadmus_real_benefit.json` —
  pre-redesign benefit measurement: A helps, B no-op, C destroys.
- `measure_real_benefit_v2.py` + `resultados_cadmus_real_benefit_v2.json`
  — post-redesign measurement with real noise model: A+B+C beats A on
  `ghz_3q`.
- `confirm_redesign_hardware.py` + `confirm_redesign_log.txt` +
  `resultados_redesign_hardware.json` — real `ibm_kingston` confirmation:
  A+B+C ~2.4x better than A alone.
- `sweep_qubit_count.py` + `sweep_log.txt` +
  `resultados_sweep_qubit_count.json` — the qubit-count sweep showing the
  non-monotonic benefit.
