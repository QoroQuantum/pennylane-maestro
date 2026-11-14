# Noise

PennyLane's noise channels run on the device as they do on `default.mixed`:
put them in the circuit, or let `qp.add_noise` insert them from a
`qp.NoiseModel`.

```python
dev = qp.device("maestro.qubit", wires=2, simulation_type="DensityMatrix")

@qp.qnode(dev)
def circuit(theta):
    qp.RX(theta, wires=0)
    qp.CNOT(wires=[0, 1])
    qp.AmplitudeDamping(0.05, wires=0)
    qp.DepolarizingChannel(0.02, wires=1)
    return qp.expval(qp.Z(0) @ qp.Z(1))
```

A noise model attaches the channels to the gates it matches:

```python
model = qp.NoiseModel({
    qp.noise.op_eq(qp.CNOT): qp.noise.partial_wires(qp.DepolarizingChannel, 0.02),
    qp.noise.op_in(["RX", "RY"]): qp.noise.partial_wires(qp.AmplitudeDamping, 0.01),
})
noisy_circuit = qp.add_noise(circuit, model)
```

Noise needs a qoro-maestro release that has `QuantumChannel`: any release after
0.3.14. An older release raises an error that says so.

## Exact or sampled

How a channel runs depends on the simulation type.

| `simulation_type` | Noise | Analytic (`shots=None`) | Finite shots |
|---|---|---|---|
| `"DensityMatrix"`, `"MatrixProductOperator"` | Exact: the state is the mixed state. | Exact values, as on `default.mixed`. | Samples of the mixed state. |
| `"PauliPropagator"` | Exact: each channel acts on the observable. | Exact values. | Samples. |
| `"Statevector"`, `"MatrixProductState"` | Sampled: each run of the circuit applies one Kraus operator of each channel, with the probability the state gives it. | Not available: one run is one branch, not the channel. | Each shot is its own run, so the shots average to the channel. |
| the others | Not available. | | |

On the statevector and the MPS a noisy circuit therefore needs `shots`. The
device says so instead of returning one branch as if it were the result:

```python
dev = qp.device("maestro.qubit", wires=20, shots=10_000, seed=7)   # statevector
```

With shots there, expectation values of Pauli words, and of sums of them, are
the average over one run per shot, each with its own branches. That average
has no sampling error from the measurement itself, only from the noise. Other
measurements (`qp.var`, samples, counts, `qp.Hermitian`) read the shots
themselves, taken in the observable's basis.

Sampling keeps the circuit on its own qubits, so it reaches sizes a density
matrix cannot: a density matrix of n qubits costs as much as a statevector of
2n.

## Channels

Every PennyLane channel on one or two wires is taken: `AmplitudeDamping`,
`GeneralizedAmplitudeDamping`, `PhaseDamping`, `DepolarizingChannel`,
`BitFlip`, `PhaseFlip`, `PauliError`, `ResetError`, `ThermalRelaxationError`
and `QubitChannel` with your own Kraus operators. A channel on three or more
wires raises an error.

Maestro recognizes a Pauli channel or a mixture of unitaries from its Kraus
operators, and draws those without reading the state. Relaxation
(`AmplitudeDamping`, `GeneralizedAmplitudeDamping`, `ThermalRelaxationError`)
is passed in Maestro's own form, after checking that it is the map PennyLane's
Kraus operators define.

## Measurements on a mixed state

A density matrix, an MPO and the Pauli propagator have no statevector. On
those, analytic `qp.expval`, `qp.var` and `qp.probs` are computed from Pauli
expectation values, in one engine call per circuit, with or without noise:

- `qp.expval` and `qp.var` take any observable with a Pauli decomposition,
  `qp.Hermitian` included.
- `qp.probs` takes up to 16 wires; use shots beyond that.
- `qp.state()` is not available: use the measurements above.

## Gradients

The adjoint method differentiates a unitary circuit, so a circuit with
channels uses parameter-shift, also on the CPU statevector. On the density
matrix the shifted circuits are exact:

```python
from pennylane import numpy as pnp

theta = pnp.array(0.6, requires_grad=True)
qp.grad(noisy_circuit)(theta)
```

## Not available

- Noise through the Qoro cloud (`backend="qoro"` or a Divi backend): it runs
  on the local engine.
- Readout error and Maestro's own `NoiseModel` (T1, T2 and gate errors by
  qubit): use PennyLane's channels and noise models.
