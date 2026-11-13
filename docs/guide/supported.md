# Supported features

- **Gates**: Maestro's native gates (including `Rot` and `U3`) run directly; PennyLane decomposes the others (e.g. `IsingXX`) automatically.
- **Measurements**: `expval`, `var`, `probs`, `sample`, `counts` and, locally, `state`, with analytic or finite shots and shot vectors.
- **Mid-circuit measurements**: `qp.measure`, resets and `qp.cond`.
- **Snapshots**: `qp.Snapshot`, with batched Pauli expectation estimation.
- **Parameter broadcasting**: broadcast analytic Pauli expectation values and probabilities run as one batched engine call (with an `AdjointProgram`-enabled qoro-maestro); other broadcast circuits are split per batch element.
- **Delay**: `qp.Delay` timing operations.

A few of these run locally only for now: see
[Not available on the cloud yet](gpu_and_cloud.md#not-available-on-the-cloud-yet).
