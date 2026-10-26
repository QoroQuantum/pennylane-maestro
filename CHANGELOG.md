# Changelog

All notable changes to `pennylane-maestro` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor
releases may contain breaking changes).

## [Unreleased]

## [0.6.0] - 2026-10-26

### Added
- Adjoint differentiation on the CPU statevector for analytic expectation
  values of Pauli strings, tensor products and Hamiltonians/Sums.
  `diff_method="best"` now resolves to `"adjoint"` there instead of
  parameter-shift. The device implements `compute_derivatives`,
  `execute_and_compute_derivatives`, `compute_vjp` and
  `execute_and_compute_vjp`.
- `adjoint_backend` option: `"engine"` (Maestro's batched adjoint sweep,
  `maestro.AdjointProgram`; default via `"auto"`) or `"inner_product"`
  (inner-product fallback).
- Broadcast tapes with analytic Pauli expectation values run as one batched
  engine call instead of one tape per batch element (needs
  `maestro.AdjointProgram`).
- Adjoint differentiation of `qml.probs` with the engine backend: up to 10
  wires with a device Jacobian (`device_vjp=False`, the default), up to 24
  with `device_vjp=True`. Broadcast tapes returning probabilities also run as
  one batched engine call.
- Trainable broadcast parameters (e.g. data re-uploading, `qml.Rot(w * x + b)`)
  stay in one batched engine call when differentiated with a device VJP,
  which returns one gradient entry per batch element. `diff_method="best"`
  selects the device VJP for such circuits; explicit `device_vjp=False`,
  `jax.jit`, and torch with a mix of broadcast and plain trainable
  parameters still split the batch per element.
- `Rot` is a native gate.

### Changed
- **Requires `qoro-maestro>=0.3.8`** (was `>=0.3.7`). Adds adjoint differentiation
  support via Maestro's native `AdjointProgram`.

### Fixed
- `qml.adjoint(qml.U3(θ, φ, λ))` applied `U3(-θ, -φ, -λ)` instead of
  `U3(-θ, -λ, -φ)`.
- Expectation values and variances of non-diagonal observables returned
  next to `qml.state()` or `qml.probs()` (and in non-Pauli snapshots) were
  computed without rotating into the observable's eigenbasis.

## [0.5.0] - 2026-10-21

### Changed
- **Requires `qoro-maestro>=0.3.7`** (was `>=0.3.3`). Maestro 0.3.7 renamed
  several `SimulatorConfig` fields, so this release no longer works with
  older Maestro versions.
- `use_double_precision` is now an alias for the new `precision` option.
  `MaestroQubitDevice` forwards `precision` (`"single"` / `"double"`) to
  `SimulatorConfig`. The default changed from `use_double_precision=False`
  to `None`, which keeps each backend's own default instead of forcing
  single precision. Conflicting `precision` and `use_double_precision`
  values, or an unknown precision string, raise a `ValueError`.
- PauliPropagator options follow Maestro's new names:
  `pp_max_pauli_weight`, `pp_gates_between_trims` and the new
  `pp_gates_between_deduplications`.

### Added
- New `MaestroQubitDevice` options forwarded to `SimulatorConfig`:
  - `precision`
  - `enable_causal_cone_reduction` (default `True`)
  - `auto_reduce` (default `True`)
  - `gate_fusion` (default `None`, i.e. backend default)
  - `enable_observable_clustering` (default `True`) and `max_cluster_qubits`
    (default `None`). These need a Maestro release newer than 0.3.7;
    on 0.3.7 they are ignored, with a `UserWarning` if set to a non-default
    value.
- README section documenting all device options.

### Deprecated
- `pp_pauli_weight_threshold` → use `pp_max_pauli_weight`.
- `pp_steps_between_trims` → use `pp_gates_between_trims`.

  Both still work and emit a `DeprecationWarning`. Passing an old and a new
  name with different values raises a `ValueError`.

### Fixed
- **Wrong results from the Trotter snapshot fast path.** Repeated-step
  detection compared gate parameters with `np.isclose`, so segments with
  slightly different angles (e.g. `RZ(40.0)` and `RZ(40.0003)`) were treated
  as the same step and simulated with the wrong angles, silently. Parameters
  must now match exactly. Genuine repeated Trotter steps are unaffected.
- `MaestroQubitDevice.incremental_evolve` failed for devices with custom
  wire labels (e.g. `wires=["a", "b"]` or `wires=[0, 2]`). Operations and
  observables are now mapped from device wires to qubit indices.

### Tests
- Added coverage for precision handling, legacy and new PauliPropagator
  names, reduction flags, gate fusion, clustering options, the exact-match
  parameter fix and `incremental_evolve` with custom wires.

## [0.4.1] - 2026-09-24

### Added
- Support for Maestro 0.3.3 features.
- `seed` option for reproducible sampling and noise.
- Native `Delay` operation support.

## [0.4.0] - 2026-09-06

### Added
- Incremental Trotter evolution: tapes of repeated Trotter steps with
  intermediate `qml.Snapshot` expectation values are detected and run through
  Maestro's persistent-state `incremental_evolve` (linear instead of
  quadratic cost).
- `dev.incremental_evolve(...)` and `pennylane_maestro.incremental_evolve(...)`
  for calling the engine directly.
- PauliPropagator configuration parameters and automatic GPU backend
  initialisation.

## [0.3.2] - 2026-06-24

### Added
- PauliPropagator configuration support.

### Changed
- Faster batched estimation for Hamiltonian measurements.

## [0.3.1] - 2026-06-24

### Added
- `qml.Snapshot` support in the `maestro.qubit` device.

### Changed
- Consecutive Pauli-expval snapshots are batched into a single estimate call.

[Unreleased]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/QoroQuantum/pennylane-maestro/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/QoroQuantum/pennylane-maestro/releases/tag/v0.3.1
