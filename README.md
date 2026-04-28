# pennylane-maestro

[![PyPI version](https://badge.fury.io/py/pennylane-maestro.svg)](https://badge.fury.io/py/pennylane-maestro)
[![Tests](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A [PennyLane](https://pennylane.ai/) plugin for the [Maestro quantum simulator](https://github.com/QoroQuantum/maestro) by Qoro Quantum.

Drop-in replacement for `default.qubit` — one line change, same code, faster results.

```python
# Before
dev = qp.device("default.qubit", wires=20)

# After 
dev = qp.device("maestro.qubit", wires=20)
```

**Why Maestro?**

* **Iterate Faster** — Up to 20× faster statevector simulations for VQE/QAOA loops.
* **Scale Up** — Simulate 1000+ qubits using Maestro's optimized MPS backend.
* **Sample from MPS** — The only PennyLane MPS backend that supports shot-based sampling.
* **GPU Ready** — [Fast GPU simlation](https://maestro.qoroquantum.net) on MPS workloads via cuQuantum.

## Installation

```bash
pip install pennylane-maestro
```

That's it. This installs `pennylane` (≥0.38) and `qoro-maestro` (≥0.2.11) automatically.

## Quick Start

```python
import pennylane as qp
import numpy as np

dev = qp.device("maestro.qubit", wires=2)

@qp.qnode(dev)
def circuit(theta):
    qp.RX(theta, wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.PauliZ(1))

print(circuit(np.pi / 4))
```

## Performance

> Benchmarked on PennyLane 0.44.1. Run `examples/benchmark_lightning_vs_maestro.py` to reproduce.

### Statevector

| Qubits | `default.qubit` | `lightning.qubit` | **`maestro.qubit`** | vs dq | vs lq |
|--------|----------------|-------------------|---------------------|-------|-------|
| 20     | 977 ms         | 115 ms            | **45 ms**           | 22×   | 2.6×  |
| 22     | 4.31 s         | 543 ms            | **184 ms**          | 23×   | 3.0×  |
| 24     | 10.5 s         | 2.36 s            | **820 ms**          | 13×   | 2.9×  |
| 26     | DNF            | 10.1 s            | **3.56 s**          | ∞     | 2.8×  |

### MPS — Scaling to 1000+ Qubits

| Qubits | `default.qubit` | `default.tensor` (quimb) | **`maestro.qubit` (MPS)** | vs quimb |
|--------|----------------|--------------------------|--------------------------|----------|
| 100    | OOM            | 90 ms                    | **11 ms**                | 8×       |
| 500    | OOM            | 689 ms                   | **90 ms**                | 7.7×     |
| 1000   | OOM            | 1.96 s                   | **207 ms**               | 9.5×     |

### MPS Shot Sampling

> **🔥 Only Maestro supports this.** Neither `default.tensor` nor Qiskit Aer MPS offer shot-based sampling through PennyLane.

| Qubits | **`maestro.qubit` (MPS)** | Unique Samples |
|--------|--------------------------|----------------|
| 100    | 259 ms                   | 10,000         |
| 500    | 1.22 s                   | 10,000         |
| 1000   | 2.43 s                   | 10,000         |

## Backends

All backends are selected via keyword arguments — no code changes needed:

```python
# CPU Statevector (default)
dev = qp.device("maestro.qubit", wires=20)

# MPS for 100+ qubits
dev = qp.device("maestro.qubit", wires=100,
                simulation_type="MatrixProductState",
                max_bond_dimension=256)

# Stabilizer for Clifford circuits
dev = qp.device("maestro.qubit", wires=1000,
                simulation_type="Stabilizer")

# Finite shots
dev = qp.device("maestro.qubit", wires=10, shots=10_000)

# GPU (requires separate license — see below)
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu")
```

<details>
<summary><strong>All available options</strong></summary>

| `simulator_type` | Description |
|---|---|
| `"QCSim"` | Qoro's optimized CPU simulator **(default)** |
| `"Gpu"` | CUDA-accelerated GPU simulator |
| `"CompositeQCSim"` | p-block simulation |

| `simulation_type` | Description |
|---|---|
| `"Statevector"` | Full statevector **(default)** |
| `"MatrixProductState"` | MPS / tensor-train for large qubit counts |
| `"Stabilizer"` | Clifford-only stabilizer |
| `"TensorNetwork"` | General tensor network |
| `"PauliPropagator"` | Pauli propagation |
| `"ExtendedStabilizer"` | Extended stabilizer |

</details>

## Examples

### Hamiltonian VQE

Hamiltonians are evaluated natively via Maestro's batched `estimate()` — all Pauli terms in a single C++ call:

```python
import pennylane as qp
import numpy as np

n_qubits = 20
H = qp.Hamiltonian(
    [0.5] * (n_qubits - 1) + [0.3] * n_qubits,
    [qp.PauliZ(i) @ qp.PauliZ(i+1) for i in range(n_qubits - 1)] +
    [qp.PauliX(i) for i in range(n_qubits)]
)

dev = qp.device("maestro.qubit", wires=n_qubits)

@qp.qnode(dev, diff_method="parameter-shift")
def vqe_circuit(params):
    for i in range(n_qubits):
        qp.RY(params[i], wires=i)
    for i in range(n_qubits - 1):
        qp.CNOT(wires=[i, i + 1])
    return qp.expval(H)

params = np.random.random(n_qubits)
energy = vqe_circuit(params)
gradient = qp.grad(vqe_circuit)(params)
```

### 100-Qubit Ising Quench with MPS Shot Sampling

See [`examples/ising_phase_transition.py`](examples/ising_phase_transition.py) for a full demo that simulates a 100-qubit transverse-field Ising model and uses Maestro's exclusive MPS shot sampling to extract magnetization distributions and spatial correlations. Runs in ~30 seconds.

### Supported Operations

PennyLane automatically decomposes any unsupported gate (e.g. `Rot`) into Maestro's native gate set. No manual intervention needed.

---

## 🚀 GPU Acceleration

For large-scale workloads, Maestro supports CUDA-accelerated simulation (statevector, MPS, tensor network) via NVIDIA cuQuantum.

```python
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu")
```

#### GPU Benchmark — Fermi-Hubbard MPS (48 qubits, χ=256)

| Simulator | Relative Runtime |
|---|---|
| **Maestro GPU** | **1×** |
| Maestro CPU | 5× |
| Qibo GPU | 7.5× |
| Qiskit CPU | 14× |
| PennyLane GPU | 64× |

> Larger instances failed to run on some platforms, limiting comparison.

**→ [Request GPU access & free trial at maestro.qoroquantum.net](https://maestro.qoroquantum.net)**

---

## Citation

If you use `pennylane-maestro` in your research, please cite:

```bibtex
@misc{pennylane_maestro,
  author       = {{Qoro Quantum}},
  title        = {PennyLane-Maestro: High-Performance C++ Backend for PennyLane},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/QoroQuantum/pennylane-maestro}}
}
```

## Authors & License

Maintained by **Qoro Quantum** ([team@qoroquantum.de](mailto:team@qoroquantum.de)).

Licensed under **GPL-3.0**. See the `LICENSE` file for details.
