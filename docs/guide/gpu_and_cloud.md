# GPU and Qoro cloud

## GPU

`simulator_type="Gpu"` runs on Maestro's CUDA backend (statevector, MPS and tensor network, via NVIDIA cuQuantum). You need GPU access: see [Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html). With `backend="qoro"`, the same option runs on Qoro's GPU cluster instead, with no GPU or licence of your own: see [Target and options](#target-and-options).

```python
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu", precision="double")
```

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

[The Qoro cloud example](../examples/qoro_cloud.md) trains a circuit and samples a 50-qubit state on the cloud.

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
