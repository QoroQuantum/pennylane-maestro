# pennylane-maestro

[![PyPI version](https://badge.fury.io/py/pennylane-maestro.svg)](https://badge.fury.io/py/pennylane-maestro)
[![Tests](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A [PennyLane](https://pennylane.ai/) plugin for the [Maestro quantum simulator](https://github.com/QoroQuantum/maestro) by Qoro Quantum.

Run your PennyLane circuits on Maestro's high-performance C++ backend — drop-in, one-line device change, no code rewrite.

## Performance Highlights

> Benchmarked on PennyLane 0.44.1.  Run `examples/benchmark_lightning_vs_maestro.py` to reproduce.

### Statevector — Variational Circuit (analytic, `⟨Z₀⟩`)

| Qubits | `default.qubit` | `lightning.qubit` | **`maestro.qubit`** | vs dq | vs lq |
|--------|----------------|-------------------|---------------------|-------|-------|
| 20     | 977 ms         | 115 ms            | **45 ms**           | 22×   | 2.6×  |
| 22     | 4.31 s         | 543 ms            | **184 ms**          | 23×   | 3.0×  |
| 24     | 10.5 s         | 2.36 s            | **820 ms**          | 13×   | 2.9×  |
| 26     | DNF            | 10.1 s            | **3.56 s**          | ∞     | 2.8×  |

### MPS — Scaling to 1000 Qubits (analytic, bond dim = 128)

| Qubits | `default.qubit` | `default.tensor` (quimb) | **`maestro.qubit` (MPS)** | vs quimb |
|--------|----------------|--------------------------|--------------------------|----------|
| 100    | OOM            | 90 ms                    | **11 ms**                | 8×       |
| 500    | OOM            | 689 ms                   | **90 ms**                | 7.7×     |
| 1000   | OOM            | 1.96 s                   | **207 ms**               | 9.5×     |

### MPS Shot Sampling — 10,000 shots (only Maestro supports this)

| Qubits | **`maestro.qubit` (MPS)** | Unique Samples |
|--------|--------------------------|----------------|
| 100    | 259 ms                   | 10,000         |
| 500    | 1.22 s                   | 10,000         |
| 1000   | 2.43 s                   | 10,000         |

> Neither `default.tensor` nor Qiskit Aer MPS support shot-based sampling through PennyLane.

## Installation

```bash
pip install pennylane-maestro
```

This automatically installs `pennylane` (≥0.38) and `qoro-maestro` (≥0.2.8).

## Quick Start

```python
import pennylane as qml
import numpy as np

dev = qml.device("maestro.qubit", wires=2)

@qml.qnode(dev)
def circuit(theta):
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

result = circuit(np.pi / 4)
print(result)
```

## Switching Backends

All backends are selected via keyword arguments to `qml.device`:

```python
# ── CPU Statevector (default) ──
dev = qml.device("maestro.qubit", wires=20)

# ── GPU Statevector ──
dev = qml.device("maestro.qubit", wires=20, simulator_type="Gpu")

# ── GPU with Double Precision ──
dev = qml.device("maestro.qubit", wires=20,
                 simulator_type="Gpu", use_double_precision=True)

# ── MPS for 100+ qubits ──
dev = qml.device("maestro.qubit", wires=100,
                 simulation_type="MatrixProductState",
                 max_bond_dimension=256)

# ── Stabilizer for Clifford circuits ──
dev = qml.device("maestro.qubit", wires=1000,
                 simulation_type="Stabilizer")

# ── Finite shots ──
dev = qml.device("maestro.qubit", wires=10, shots=10_000)
```

### Available Options

| `simulator_type` | Description |
|---|---|
| `"QCSim"` | Qoro's optimised CPU simulator **(default)** |
| `"Gpu"` | CUDA-accelerated GPU simulator |
| `"CompositeQCSim"` | p-block Simulation |

| `simulation_type` | Description |
|---|---|
| `"Statevector"` | Full statevector **(default)** |
| `"MatrixProductState"` | MPS / tensor-train for large qubit counts |
| `"Stabilizer"` | Clifford-only stabilizer |
| `"TensorNetwork"` | General tensor network |
| `"PauliPropagator"` | Pauli propagation |
| `"ExtendedStabilizer"` | Extended stabilizer |

## Hamiltonian VQE Example

Hamiltonians are evaluated natively via Maestro's batched `estimate()` — all Pauli terms in a single C++ call:

```python
import pennylane as qml
import numpy as np

n_qubits = 20
H = qml.Hamiltonian(
    [0.5] * (n_qubits - 1) + [0.3] * n_qubits,
    [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits - 1)] +
    [qml.PauliX(i) for i in range(n_qubits)]
)

dev = qml.device("maestro.qubit", wires=n_qubits)

@qml.qnode(dev, diff_method="parameter-shift")
def vqe_circuit(params):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    return qml.expval(H)

params = np.random.random(n_qubits)
energy = vqe_circuit(params)
gradient = qml.grad(vqe_circuit)(params)
```

## Supported Operations

The plugin delegates execution to Maestro. If you use an operation not directly implemented in Maestro (e.g. `Rot`), PennyLane automatically decomposes it into supported gates.

## GPU Acceleration

Maestro supports CUDA-accelerated simulation via a dynamically-loaded GPU backend.

> **Note:** The GPU backend is **not included** in the open-source version of Maestro.
> Contact [Qoro Quantum](https://qoroquantum.de) for access to the GPU release bundle and a license key.

### Setup

1. **Obtain the GPU release bundle** from [Qoro Quantum](https://qoroquantum.de) (<team@qoroquantum.de>). It contains:
   - `libmaestro_gpu_simulators.so` — the GPU simulator library
   - `libLexActivator.so` — license validation library

2. **Install the libraries** into your environment:

   ```bash
   cp libmaestro_gpu_simulators.so $CONDA_PREFIX/lib/
   cp libLexActivator.so $CONDA_PREFIX/lib/
   ```

3. **Set your license key** (provided by Qoro Quantum):

   ```bash
   export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"
   ```

   To persist across sessions:

   ```bash
   echo 'export MAESTRO_LICENSE_KEY="XXXX-XXXX-XXXX-XXXX"' >> ~/.bashrc
   source ~/.bashrc
   ```

   > The first run requires an internet connection for license activation. After that, the license is cached locally for 30 days of offline use.

4. **Use the GPU device** in PennyLane:

   ```python
   import maestro
   import pennylane as qml

   # Initialize GPU backend (required once per session)
   assert maestro.init_gpu(), "GPU init failed — check library path and CUDA install"

   # GPU Statevector
   dev = qml.device("maestro.qubit", wires=28,
                    simulator_type="Gpu",
                    use_double_precision=True)

   # GPU MPS for large circuits
   dev_mps = qml.device("maestro.qubit", wires=200,
                        simulator_type="Gpu",
                        simulation_type="MatrixProductState",
                        max_bond_dimension=256)
   ```

### GPU-Supported Simulation Modes

| Simulation Type       | GPU Support |
|-----------------------|-------------|
| Statevector           | ✅           |
| MPS                   | ✅           |
| Tensor Network        | ✅           |
| Pauli Propagator      | ✅           |
| Stabilizer            | ❌           |
| Extended Stabilizer   | ❌           |

> **Requirements**: Linux with NVIDIA GPU (Ampere/Hopper recommended), CUDA 13.0+, and [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) (`conda install -c conda-forge cuquantum`).

## Authors & License

Maintained by **Qoro Quantum** (<team@qoroquantum.de>).

Licensed under **GPL-3.0**. See the `LICENSE` file for more details.
