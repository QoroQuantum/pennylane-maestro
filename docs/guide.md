# pennylane-maestro guide

Everything the [README](../README.md) leaves out:

- [Options](#options)
- [Gradients](#gradients)
- [Qoro cloud](#qoro-cloud)
- [Time evolution](#time-evolution)
- [GPU](#gpu)
- [What's supported](#whats-supported)
- [Performance](#performance)

All examples use `import pennylane as qp`.

## Options

Every option is a keyword argument of `qp.device("maestro.qubit", ...)`:

```python
# MPS for 100+ qubits
dev = qp.device("maestro.qubit", wires=100,
                simulation_type="MatrixProductState", max_bond_dimension=256)

# Stabilizer for Clifford circuits
dev = qp.device("maestro.qubit", wires=1000, simulation_type="Stabilizer")

# Finite shots with a reproducible seed
dev = qp.device("maestro.qubit", wires=10, shots=10_000, seed=42)

# Pauli propagation with truncation
dev = qp.device("maestro.qubit", wires=200, simulation_type="PauliPropagator",
                pp_max_pauli_weight=8, pp_gates_between_trims=10)
```

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

On the [Qoro cloud](#qoro-cloud), leaving `simulation_type` and `simulator_type` unset lets the service choose.

The other options map onto Maestro's [`SimulatorConfig`](https://qoroquantum.github.io/maestro/api/configuration.html):

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
| `enable_observable_clustering` | `True` | Split multi-observable estimates into groups with disjoint lightcones. |
| `max_cluster_qubits` | `None` | Cap each observable cluster at this many lightcone qubits. |
| `gate_fusion` | `None` | Fuse compatible gates. `None` uses each backend's default. |
| `seed` | `None` | Seed for reproducible sampling and noise. |
| `adjoint_backend` | `"auto"` | How adjoint derivatives are computed (see [Gradients](#gradients)): `"engine"`, `"inner_product"`, or `"auto"` (engine by default). |
| `backend` | `"local"` | Where circuits run: `"local"`, `"qoro"` (the Qoro cloud), or a Divi `CircuitRunner`. See [Qoro cloud](#qoro-cloud). |

`use_double_precision=True/False` is still accepted as shorthand for `precision="double"/"single"`. The legacy names `pp_pauli_weight_threshold` (→ `pp_max_pauli_weight`) and `pp_steps_between_trims` (→ `pp_gates_between_trims`) still work but emit a `DeprecationWarning`.

## Gradients

On the CPU statevector (`simulator_type="QCSim"`, `simulation_type="Statevector"`, the defaults), analytic expectation values of Pauli strings, tensor products and Hamiltonians/Sums, and `qp.probs`, are differentiated with the adjoint method. `diff_method="best"` (the QNode default) resolves to `"adjoint"`:

```python
dev = qp.device("maestro.qubit", wires=8)

@qp.qnode(dev)                       # diff_method="best" -> "adjoint"
def circuit(weights, x):
    qp.AngleEmbedding(x, wires=range(8))
    qp.StronglyEntanglingLayers(weights, wires=range(8))
    return qp.expval(qp.Z(0))
```

- With a qoro-maestro build that includes `AdjointProgram`, the adjoint sweep runs in the engine, restricted to each observable's causal cone. A broadcast batch of inputs (e.g. `x` of shape `(32, 8)`) is evaluated and differentiated in one engine call. This includes trainable broadcast parameters, as in data re-uploading (`qp.Rot(w * x + b, ...)`): `diff_method="best"` then uses a device VJP, which returns one gradient entry per batch element. With `device_vjp=False` set explicitly, with `jax.jit`, or with torch when only some trainable parameters are broadcast, such a batch is split into one circuit per element.
- `qp.probs` needs the engine path. A Jacobian costs one backward sweep per outcome, so it is used for up to 10 wires with the default `device_vjp=False` and up to 24 wires with `device_vjp=True` (one sweep in total); wider probabilities use parameter-shift.
- The inner-product fallback (`adjoint_backend="inner_product"`) is also available, evaluating one `maestro.inner_product` per gate parameter and Pauli term (expectation values only).
- Finite shots, mid-circuit measurements, other simulation types (MPS, stabilizer, ...) and other measurements (`var`, `state`, samples, `probs` beyond the wire limits above) use parameter-shift. `diff_method="adjoint"` raises for these.
- With PennyLane's default `device_vjp=False`, the Jacobian is computed together with the forward pass. This is the fastest choice for training, but plain forward calls with trainable (`requires_grad=True`) parameters also pay for the Jacobian. For inference-heavy use, pass `device_vjp=True` to the QNode: forward calls then only simulate, and the gradient is a separate VJP pass (on a 20-qubit, 32-sample VQC: forward 0.46 s instead of 1.73 s, loss+grad 2.55 s instead of 1.82 s).

## Qoro cloud

`backend="qoro"` sends circuits through [Divi](https://github.com/QoroQuantum/divi) to the Qoro cloud instead of simulating them in-process. The `cloud` extra installs Divi 0.16 or later, which needs Python 3.11+:

```bash
pip install "pennylane-maestro[cloud]"
export QORO_API_KEY=...        # or put it in a .env file
```

```python
dev = qp.device("maestro.qubit", wires=20, backend="qoro",
                simulation_type="MatrixProductState", max_bond_dimension=64)
```

### Target and options

The job target (a simulator cluster or a QPU system) comes from the defaults saved with `divi-config-helper`, or is the `qoro_maestro` simulator if none are saved. Device options that differ from Maestro's defaults (`max_bond_dimension`, `seed`, ...), and `simulator_type`/`simulation_type` when given, are layered over the saved Maestro defaults; options left unset are the service's choice. They apply to simulator targets only: with a QPU target the device warns and leaves them out.

`simulator_type="Gpu"` sends the jobs to Qoro's GPU cluster, `qoro_gpu`, whatever the saved target, so no GPU of your own is needed:

```python
dev = qp.device("maestro.qubit", wires=30, backend="qoro", simulator_type="Gpu")
```

Maestro's adjoint method needs the CPU statevector, so on the GPU cluster `diff_method="best"` uses parameter-shift: two jobs per optimizer step, one for the circuit and one for all its shifted copies.

For full control, pass a configured Divi backend instead; its own `maestro_config` applies:

```python
from divi.backends import JobConfig, MaestroConfig, QoroService

service = QoroService(job_config=JobConfig(simulator_cluster="qoro_maestro"),
                      maestro_config=MaestroConfig(seed=7))
dev = qp.device("maestro.qubit", wires=20, backend=service)
print(service.get_credit_balance())
```

Any Divi `CircuitRunner` works the same way, e.g. `divi.backends.MaestroSimulator()` to try a script offline.

### How circuits run

- Each batch of tapes is at most one expectation-value job and one sampling job.
  - The expectation-value job takes analytic tapes, and finite-shot tapes of Pauli-word expectation values (exact, as on the local engine). `var` follows from `<O^2>`; `qp.probs`, mid-circuit measurement values and observables without a Pauli form (`Hermitian`, `Projector`, `qp.probs(op=...)`) are read exactly from the Pauli-Z expectation values over their wires (at most 10), after the observable's diagonalizing gates.
  - The sampling job takes the other finite-shot tapes: one circuit per measurement basis, as default.qubit groups its samples, and per shot copy.
- `diff_method="adjoint"` (and `"best"`) differentiates analytic Pauli/Hamiltonian expectation values on the service with Divi's adjoint VJPs, as one `submit_vjp` job per register width in the batch; with the default `device_vjp=False` the values come back in the same job. This needs a CPU statevector target (`QCSim`, `Statevector`, which the service picks when they are unset). Otherwise, and for finite shots or second derivatives (`max_diff=2`), `"best"` uses parameter-shift, run as ordinary jobs.
- Mid-circuit measurements are deferred (no postselection), and broadcast tapes are split per element. `qp.Snapshot`s with a measurement run as extra tapes in the same jobs under `qp.snapshots`.

### Not available on the cloud yet

`qp.state()` and other state-based measurements, state snapshots, `dev.incremental_evolve`, and analytic `qp.probs` (or observables without a Pauli form) on more than 10 wires. Each raises an error.

## Time evolution

Time-evolution algorithms often measure expectation values at intermediate steps. Re-running from t = 0 for each step costs O(N²) over N steps; Maestro keeps one persistent simulator state across steps instead, for O(N).

When a circuit repeats a Trotter step with `qp.Snapshot`s in between, `maestro.qubit` detects the pattern and uses this automatically:

```python
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

For large workloads (e.g. 100+ steps on MPS), `incremental_evolve` skips building tapes entirely:

```python
dev = qp.device("maestro.qubit", wires=2, simulation_type="MatrixProductState")

res = dev.incremental_evolve(
    init=lambda: qp.PauliX(0),
    trotter_step=lambda: qp.IsingXX(0.2, wires=[0, 1]),
    measure_at_steps=[10, 20, 30],
    observables=[qp.PauliZ(0), qp.PauliZ(1)],
)
# res["expectation_values"] has shape (len(measure_at_steps), len(observables))
```

## GPU

`simulator_type="Gpu"` runs on Maestro's CUDA backend (statevector, MPS and tensor network, via NVIDIA cuQuantum). You need GPU access: see [Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html). With `backend="qoro"`, the same option runs on Qoro's GPU cluster instead, with no GPU or licence of your own: see [Qoro cloud](#target-and-options).

```python
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu", precision="double")
```

## What's supported

- **Gates**: Maestro's native gates (including `Rot` and `U3`) run directly; PennyLane decomposes the others (e.g. `IsingXX`) automatically.
- **Measurements**: `expval`, `var`, `probs`, `sample`, `counts` and, locally, `state`, with analytic or finite shots and shot vectors.
- **Mid-circuit measurements**: `qp.measure`, resets and `qp.cond`.
- **Snapshots**: `qp.Snapshot`, with batched Pauli expectation estimation.
- **Parameter broadcasting**: broadcast analytic Pauli expectation values and probabilities run as one batched engine call (with an `AdjointProgram`-enabled qoro-maestro); other broadcast circuits are split per batch element.
- **Delay**: `qp.Delay` timing operations.

## Performance

Measured with [`examples/benchmark_lightning_vs_maestro.py`](../examples/benchmark_lightning_vs_maestro.py) on an Apple M4 Pro (8 performance and 4 efficiency cores, 48 GB) with macOS 26.6, Python 3.12, PennyLane 0.45.1, pennylane-lightning 0.45.0, quimb 1.15.0 and qoro-maestro 0.3.14, default thread settings. Each time is the mean of 3 runs after a warm-up; DNF means the warm-up and the runs took over 60 s.

### Statevector

`StronglyEntanglingLayers` (3 layers), `⟨Z₀⟩`, analytic:

| Qubits | `default.qubit` | `lightning.qubit` | **`maestro.qubit`** | vs `default.qubit` | vs `lightning.qubit` |
|---|---|---|---|---|---|
| 20 | 1.61 s | 112 ms | **22.8 ms** | 70× | 4.9× |
| 22 | 7.07 s | 534 ms | **95.0 ms** | 74× | 5.6× |
| 24 | DNF | 2.33 s | **391 ms** | — | 6.0× |
| 26 | DNF | 10.1 s | **1.68 s** | — | 6.0× |

### MPS, up to 1,000 qubits

RY + CNOT ladder (3 layers), bond dimension 128. Expectation value `⟨Z₀Z₁⟩`, and 10,000 shots of every qubit. `default.qubit` runs out of memory at these sizes, and `default.tensor` doesn't support finite shots:

| Qubits | `default.tensor` (quimb) | **`maestro.qubit`** | Speedup | **`maestro.qubit`**, 10,000 shots |
|---|---|---|---|---|
| 100 | 80.9 ms | **8.1 ms** | 10× | **216 ms** |
| 200 | 183 ms | **15.7 ms** | 12× | **607 ms** |
| 500 | 570 ms | **49.0 ms** | 12× | **1.55 s** |
| 1000 | 1.60 s | **91.2 ms** | 18× | **3.27 s** |

### GPU

Fermi-Hubbard MPS (48 qubits, χ = 256), from an earlier GPU run (not re-measured here):

| Simulator | Relative runtime |
|---|---|
| **Maestro GPU** | **1×** |
| Maestro CPU | 5× |
| Qibo GPU | 7.5× |
| Qiskit CPU | 14× |
| PennyLane GPU | 64× |
