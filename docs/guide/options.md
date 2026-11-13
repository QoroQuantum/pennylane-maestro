# Options and simulation methods

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

## Simulation and simulator types

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

On the [Qoro cloud](gpu_and_cloud.md#qoro-cloud), leaving `simulation_type` and `simulator_type` unset lets the service choose.

## Other options

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
| `adjoint_backend` | `"auto"` | How adjoint derivatives are computed (see [Gradients](gradients.md)): `"engine"`, `"inner_product"`, or `"auto"` (engine by default). |
| `backend` | `"local"` | Where circuits run: `"local"`, `"qoro"` (the Qoro cloud), or a Divi `CircuitRunner`. See [Qoro cloud](gpu_and_cloud.md#qoro-cloud). |

`use_double_precision=True/False` is still accepted as shorthand for `precision="double"/"single"`. The legacy names `pp_pauli_weight_threshold` (→ `pp_max_pauli_weight`) and `pp_steps_between_trims` (→ `pp_gates_between_trims`) still work but emit a `DeprecationWarning`.
