# Performance

Measured with [`examples/benchmark_lightning_vs_maestro.py`](../examples/benchmark_lightning_vs_maestro.md) on an Apple M4 Pro (8 performance and 4 efficiency cores, 48 GB) with macOS 26.6, Python 3.12, PennyLane 0.45.1, pennylane-lightning 0.45.0, quimb 1.15.0 and qoro-maestro 0.3.14, default thread settings. Each time is the mean of 3 runs after a warm-up; DNF means the warm-up and the runs took over 60 s.

## Statevector

`StronglyEntanglingLayers` (3 layers), `⟨Z₀⟩`, analytic:

| Qubits | `default.qubit` | `lightning.qubit` | **`maestro.qubit`** | vs `default.qubit` | vs `lightning.qubit` |
|---|---|---|---|---|---|
| 20 | 1.61 s | 112 ms | **22.8 ms** | 70× | 4.9× |
| 22 | 7.07 s | 534 ms | **95.0 ms** | 74× | 5.6× |
| 24 | DNF | 2.33 s | **391 ms** | — | 6.0× |
| 26 | DNF | 10.1 s | **1.68 s** | — | 6.0× |

## MPS, up to 1,000 qubits

RY + CNOT ladder (3 layers), bond dimension 128. Expectation value `⟨Z₀Z₁⟩`, and 10,000 shots of every qubit. `default.qubit` runs out of memory at these sizes, and `default.tensor` doesn't support finite shots:

| Qubits | `default.tensor` (quimb) | **`maestro.qubit`** | Speedup | **`maestro.qubit`**, 10,000 shots |
|---|---|---|---|---|
| 100 | 80.9 ms | **8.1 ms** | 10× | **216 ms** |
| 200 | 183 ms | **15.7 ms** | 12× | **607 ms** |
| 500 | 570 ms | **49.0 ms** | 12× | **1.55 s** |
| 1000 | 1.60 s | **91.2 ms** | 18× | **3.27 s** |

## GPU

Fermi-Hubbard MPS (48 qubits, χ = 256), from an earlier GPU run (not re-measured here):

| Simulator | Relative runtime |
|---|---|
| **Maestro GPU** | **1×** |
| Maestro CPU | 5× |
| Qibo GPU | 7.5× |
| Qiskit CPU | 14× |
| PennyLane GPU | 64× |
