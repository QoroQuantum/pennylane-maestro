# pennylane-maestro

[![PyPI version](https://badge.fury.io/py/pennylane-maestro.svg)](https://badge.fury.io/py/pennylane-maestro)
[![Tests](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml/badge.svg)](https://github.com/QoroQuantum/pennylane-maestro/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/QoroQuantum/pennylane-maestro/blob/main/LICENSE)

<!-- The documentation's home page (docs/index.md) shows the README from here to "end-docs". -->
<!-- start-docs -->

Run your PennyLane code on the [Maestro](https://github.com/QoroQuantum/maestro)
quantum simulator, on your machine or in the Qoro cloud, by changing the device name.

- **Drop-in for `default.qubit`**: the same QNodes, measurements, mid-circuit
  measurements and broadcasting, with adjoint gradients by default.
- **Beyond statevector**: matrix product state (MPS) for hundreds of qubits,
  with shot sampling, plus stabilizer, tensor network and Pauli propagation.
- **Noise**: PennyLane's noise channels and `qp.add_noise` models, exact on
  the density matrix and the MPO as on `default.mixed`, or sampled shot by shot
  on the statevector and the MPS for circuits a density matrix cannot hold.
- **Fast**: 5–6× faster than `lightning.qubit` on 20–26-qubit statevector
  circuits, and 10–18× faster than `default.tensor` on MPS circuits up to
  1,000 qubits ([benchmarks](https://qoroquantum.github.io/pennylane-maestro/guide/performance.html)).
- **CPU, GPU or cloud**: run locally, on your own GPU, or on Qoro's CPUs and
  GPUs in the cloud with no hardware of your own.

## Install

```bash
pip install pennylane-maestro            # local simulation, Python 3.10+
pip install "pennylane-maestro[cloud]"   # adds the Qoro cloud, Python 3.11+
```

## Quick start

```python
import pennylane as qp
from pennylane import numpy as pnp

dev = qp.device("maestro.qubit", wires=2)   # was: qp.device("default.qubit", wires=2)

@qp.qnode(dev)
def circuit(theta):
    qp.RX(theta, wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.Z(1))

theta = pnp.array(0.4, requires_grad=True)
print(circuit(theta), qp.grad(circuit)(theta))   # the gradient uses the adjoint method
```

## Run on the Qoro cloud

1. Sign up at [dash.qoroquantum.net](https://dash.qoroquantum.net) to get an
   API key. New accounts get 1,000 free tokens, with no credit card.
2. Install the cloud extra and set your key:

   ```bash
   pip install "pennylane-maestro[cloud]"
   export QORO_API_KEY=...   # or put it in a .env file
   ```

3. Add `backend="qoro"` to the device. Nothing else changes:

   ```python
   dev = qp.device("maestro.qubit", wires=2, backend="qoro")
   ```

The quick start above then runs in the cloud, adjoint gradient included. Each
optimizer step usually goes as a single job, results come back as they do
locally, and options such as `simulation_type="MatrixProductState"` travel with
the job. Add `simulator_type="Gpu"` to run on Qoro's GPU cluster.
[examples/qoro_cloud.py](https://github.com/QoroQuantum/pennylane-maestro/blob/main/examples/qoro_cloud.py)
trains a circuit and samples a 50-qubit state in the cloud. A few features, such
as `qp.state()`, run locally only for now: see the
[guide](https://qoroquantum.github.io/pennylane-maestro/guide/gpu_and_cloud.html#not-available-on-the-cloud-yet).

## Pick a simulation method

```python
dev = qp.device("maestro.qubit", wires=100,
                simulation_type="MatrixProductState", max_bond_dimension=64)
```

| `simulation_type` | Use it for |
|---|---|
| `"Statevector"` (default) | exact results; memory doubles with each qubit |
| `"MatrixProductState"` | many qubits with limited entanglement, including shot sampling |
| `"Stabilizer"` | Clifford circuits, any size |
| `"PauliPropagator"` | expectation values on many qubits, with Pauli-string truncation |

Add `simulator_type="Gpu"` for your own GPU
([get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html)).
The [guide](https://qoroquantum.github.io/pennylane-maestro/guide/options.html)
lists every method and option.

<!-- end-docs -->

## Learn more

- [Documentation](https://qoroquantum.github.io/pennylane-maestro/):
  - [Guide](https://qoroquantum.github.io/pennylane-maestro/guide.html): every
    option, gradients, the cloud in detail, time evolution with `qp.Snapshot`,
    what's supported, and benchmarks.
  - [Examples](https://qoroquantum.github.io/pennylane-maestro/examples/): the
    cloud, VQE, QAOA, a data re-uploading classifier, a 100-qubit Ising quench
    with MPS sampling, Trotter time evolution, and the benchmark suite.
  - [API reference](https://qoroquantum.github.io/pennylane-maestro/api.html).
- [Maestro docs](https://qoroquantum.github.io/maestro/).
- [Changelog](https://github.com/QoroQuantum/pennylane-maestro/blob/main/CHANGELOG.md).
- [Contributing](https://github.com/QoroQuantum/pennylane-maestro/blob/main/CONTRIBUTING.md):
  running the tests and building the docs.

## Citation

```bibtex
@misc{pennylane_maestro,
  author       = {{Qoro Quantum}},
  title        = {PennyLane-Maestro: High-Performance C++ Backend for PennyLane},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/QoroQuantum/pennylane-maestro}}
}
```

## License

GPL-3.0. See [LICENSE](https://github.com/QoroQuantum/pennylane-maestro/blob/main/LICENSE).
