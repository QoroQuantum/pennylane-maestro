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
* **Persistent-State Trotter Evolution** — Linear-time simulation of repeated Trotter steps with intermediate snapshots (`qp.Snapshot` or `incremental_evolve`).
* **GPU Ready** — [Fast GPU simulation](https://maestro.qoroquantum.net) on MPS workloads via cuQuantum.

## Installation

```bash
pip install pennylane-maestro
```

That's it. This installs `pennylane` (≥0.38) and `qoro-maestro` (≥0.3.7) automatically.

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

# Finite shots with reproducible seed
dev = qp.device("maestro.qubit", wires=10, shots=10_000, seed=42)

# GPU (requires separate license — see below)
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu")
```

<details>
<summary><strong>All available options</strong></summary>

| `simulator_type` | Description |
|---|---|
| `"QCSim"` | Qoro's optimized CPU simulator **(default)** |
| `"Gpu"` | CUDA-accelerated GPU simulator |
| `"CompositeQCSim"` | Multi-node distributed CPU |
| `"QiskitAer"` | Integrated Qiskit Aer backend |
| `"CompositeQiskitAer"` | Distributed Qiskit Aer backend |
| `"DistributedGpu"` | State distributed across local GPUs |
| `"DistributedMpiGpu"` | State distributed across MPI ranks/GPUs |
| `"QuestSim"` | QuEST simulator |

| `simulation_type` | Description |
|---|---|
| `"Statevector"` | Full statevector **(default)** |
| `"MatrixProductState"` | MPS / tensor-train for large qubit counts |
| `"Stabilizer"` | Clifford-only stabilizer |
| `"TensorNetwork"` | General tensor network |
| `"PauliPropagator"` | Pauli propagation |
| `"ExtendedStabilizer"` | Extended stabilizer |
| `"DensityMatrix"` | Density matrix for mixed states |
| `"MatrixProductOperator"` | MPO simulation |
| `"PathIntegral"` | Path integral simulation |

</details>

### Device options

Further keyword arguments map onto Maestro's `SimulatorConfig`:

```python
# Double precision on GPU / Qiskit Aer
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu",
                precision="double")

# Pauli propagation with truncation
dev = qp.device("maestro.qubit", wires=200,
                simulation_type="PauliPropagator",
                pp_max_pauli_weight=8, pp_gates_between_trims=10)

# Force gate fusion on; turn off pre-execution reductions
dev = qp.device("maestro.qubit", wires=20, gate_fusion=True,
                enable_causal_cone_reduction=False, auto_reduce=False)
```

| Option | Default | Description |
|---|---|---|
| `precision` | `None` | `"single"` or `"double"` for GPU and Qiskit Aer backends. `None` keeps each backend's default; QCSim always uses double. |
| `max_bond_dimension` | `None` | MPS bond-dimension cap. |
| `singular_value_threshold` | `None` | MPS SVD truncation cutoff. |
| `disable_optimized_swapping` | `False` | Disable MPS swap optimisation. |
| `pp_coefficient_threshold` | `None` | PauliPropagator: drop Pauli strings with smaller coefficients. |
| `pp_max_pauli_weight` | `None` | PauliPropagator: drop Pauli strings acting on more qubits than this. |
| `pp_gates_between_trims` | `None` | PauliPropagator: gates between truncation passes. |
| `pp_gates_between_deduplications` | `None` | PauliPropagator: gates between merges of duplicate strings. |
| `enable_causal_cone_reduction` | `True` | Simulate only the causal cone of the measured observables. |
| `auto_reduce` | `True` | Detect invariant subspaces and simulate the reduced register. |
| `enable_observable_clustering` | `True` | Split multi-observable estimates into groups with disjoint lightcones. Requires a Maestro release newer than 0.3.7; ignored otherwise. |
| `max_cluster_qubits` | `None` | Cap each observable cluster at this many lightcone qubits. Requires a Maestro release newer than 0.3.7; ignored otherwise. |
| `gate_fusion` | `None` | Fuse compatible gates. `None` uses each backend's default. |
| `seed` | `None` | Seed for reproducible sampling and noise. |
| `adjoint_backend` | `"auto"` | How adjoint derivatives are computed (see [Gradients](#gradients)): `"engine"`, `"inner_product"`, or `"auto"` (engine when available). |

`use_double_precision=True/False` is still accepted as shorthand for `precision="double"/"single"`. The legacy names `pp_pauli_weight_threshold` (→ `pp_max_pauli_weight`) and `pp_steps_between_trims` (→ `pp_gates_between_trims`) still work but emit a `DeprecationWarning`.

### Gradients

On the CPU statevector (`simulator_type="QCSim"`, `simulation_type="Statevector"`, the defaults), analytic expectation values of Pauli strings, tensor products and Hamiltonians/Sums, and `qml.probs`, are differentiated with the adjoint method. `diff_method="best"` (the QNode default) resolves to `"adjoint"`:

```python
dev = qp.device("maestro.qubit", wires=8)

@qp.qnode(dev)                       # diff_method="best" -> "adjoint"
def circuit(weights, x):
    qp.AngleEmbedding(x, wires=range(8))
    qp.StronglyEntanglingLayers(weights, wires=range(8))
    return qp.expval(qp.Z(0))
```

* With a qoro-maestro build that includes `AdjointProgram`, the adjoint sweep runs in the engine, restricted to each observable's causal cone. A broadcast batch of inputs (e.g. `x` of shape `(32, 8)`) is evaluated and differentiated in one engine call.
* `qml.probs` needs the engine path. A Jacobian costs one backward sweep per outcome, so it is used for up to 10 wires with the default `device_vjp=False` and up to 24 wires with `device_vjp=True` (one sweep in total); wider probabilities use parameter-shift.
* With older qoro-maestro releases, `adjoint_backend="inner_product"` is used: one `maestro.inner_product` per gate parameter and Pauli term (expectation values only).
* Finite shots, mid-circuit measurements, other simulation types (MPS, stabilizer, ...) and other measurements (`var`, `state`, samples, `probs` beyond the wire limits above) use parameter-shift, as before. `diff_method="adjoint"` raises for these.
* With PennyLane's default `device_vjp=False`, the Jacobian is computed together with the forward pass. This is the fastest choice for training, but plain forward calls with trainable (`requires_grad=True`) parameters also pay for the Jacobian. For inference-heavy use, pass `device_vjp=True` to the QNode: forward calls then only simulate, and the gradient is a separate VJP pass (on a 20-qubit, 32-sample VQC: forward 0.46 s instead of 1.73 s, loss+grad 2.55 s instead of 1.82 s).

## Examples

### Hamiltonian VQE

Hamiltonians are evaluated natively via Maestro's batched `estimate()` — all Pauli terms in a single C++ call:

```python
import pennylane as qp
from pennylane import numpy as np  # trainable arrays for qp.grad

n_qubits = 20
H = qp.Hamiltonian(
    [0.5] * (n_qubits - 1) + [0.3] * n_qubits,
    [qp.PauliZ(i) @ qp.PauliZ(i+1) for i in range(n_qubits - 1)] +
    [qp.PauliX(i) for i in range(n_qubits)]
)

dev = qp.device("maestro.qubit", wires=n_qubits)

@qp.qnode(dev)  # adjoint gradients on the statevector
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

### Incremental Trotter Evolution & Snapshots

Time evolution algorithms often measure expectation values across intermediate steps. Standard simulation re-runs from time $t=0$ for each step (quadratic $O(N^2)$ cost). Maestro retains persistent simulator state across steps for linear $O(N)$ execution.

#### Via PennyLane Snapshots (Automatic Acceleration)

When a circuit repeats a Trotter step with intermediate `qp.Snapshot` calls, `maestro.qubit` automatically detects the pattern and uses Maestro's persistent-state execution:

```python
import pennylane as qp

dev = qp.device("maestro.qubit", wires=2)

@qp.qnode(dev)
def trotter_circuit():
    qp.PauliX(0)
    for step in range(1, 4):
        qp.IsingXX(0.2, wires=[0, 1])
        qp.Snapshot(f"step_{step}", measurement=qp.expval(qp.PauliZ(0)))
    return qp.expval(qp.PauliZ(0))

results = qp.snapshots(trotter_circuit)()
# results contains "step_1", "step_2", "step_3", and "execution_results"
```

#### Via Direct Incremental Evolution API

For large workloads (e.g. 100+ steps on MPS), bypass tape generation entirely with `incremental_evolve`:

```python
import pennylane as qp
from pennylane_maestro import incremental_evolve

dev = qp.device("maestro.qubit", wires=2, simulation_type="MatrixProductState")

res = dev.incremental_evolve(
    init=lambda: qp.PauliX(0),
    trotter_step=lambda: qp.IsingXX(0.2, wires=[0, 1]),
    measure_at_steps=[10, 20, 30],
    observables=[qp.PauliZ(0), qp.PauliZ(1)],
)
# res["expectation_values"] has shape (len(measure_at_steps), len(observables))
```

### 100-Qubit Ising Quench with MPS Shot Sampling

See [`examples/ising_phase_transition.py`](examples/ising_phase_transition.py) for a full demo that simulates a 100-qubit transverse-field Ising model and uses Maestro's exclusive MPS shot sampling to extract magnetization distributions and spatial correlations. Runs in ~30 seconds.

### Supported Operations

* **Automatic Gate Decomposition** — PennyLane decomposes any unsupported gate (e.g. `IsingXX`) into Maestro's native gate set automatically. `Rot` and `U3` are native.
* **Mid-Circuit Measurements (MCM)** — Supports mid-circuit measurements (`qp.measure`), resets, and conditional operations (`qp.cond`).
* **Snapshots** — Native support for `qp.Snapshot` with batched Pauli expectation estimation.
* **Parameter Broadcasting** — Broadcast analytic Pauli expectation values run as one batched engine call (with an `AdjointProgram`-enabled qoro-maestro); other broadcast circuits are split per batch element.
* **Delay** — Native support for `qp.Delay` timing operations.

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
