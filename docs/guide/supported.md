# Supported features

- **Gates**: Maestro's native gates (including `Rot` and `U3`) run directly; PennyLane decomposes the others (e.g. `IsingXX`) automatically.
- **Measurements**: `expval`, `var`, `probs`, `sample`, `counts` and, locally, `state`, with analytic or finite shots and shot vectors.
  With shots, each measurement reads samples taken in its observable's basis: measurements that need different bases on a wire take one run of the shots per basis. Expectation values of Pauli words and of sums of them, alone in a circuit, come from Maestro's `estimate` instead and have no shot noise.
- **Mid-circuit measurements**: `qp.measure`, resets and `qp.cond`.
- **Noise**: PennyLane's noise channels on one or two wires and `qp.add_noise` models, exact on the density matrix, the MPO and the Pauli propagator and sampled per shot on the statevector and the MPS. See [Noise](noise.md).
- **Snapshots**: `qp.Snapshot`, with batched Pauli expectation estimation.
- **Parameter broadcasting**: broadcast analytic Pauli expectation values and probabilities run as one batched engine call (with an `AdjointProgram`-enabled qoro-maestro); other broadcast circuits are split per batch element.
- **Delay**: `qp.Delay` timing operations.

A few of these run locally only for now: see
[Not available on the cloud yet](gpu_and_cloud.md#not-available-on-the-cloud-yet).
