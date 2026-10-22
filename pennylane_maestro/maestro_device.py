"""MaestroQubitDevice — PennyLane device backed by the Maestro simulator.

Implements the **new** ``pennylane.devices.Device`` interface (schema 3).
"""

import warnings
from dataclasses import replace
from os import path

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import decompose as _decompose, validate_device_wires
from pennylane.devices.qubit import measure as _qubit_measure
from pennylane.exceptions import DeviceError
from pennylane.measurements import ExpectationMP, MidMeasureMP, SampleMP
from pennylane.ops import Snapshot
from pennylane.ops.op_math import Adjoint, Conditional
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import broadcast_expand, defer_measurements
try:
    from pennylane.transforms.core import CompilePipeline
except ImportError:
    from pennylane.transforms.core import TransformProgram as CompilePipeline
from pennylane.typing import Result, ResultBatch

import maestro
from maestro import SimulatorConfig

from pennylane_maestro import adjoint as _adjoint
from pennylane_maestro.converter import (
    GATE_MAP,
    ADJOINT_MAP,
    tape_to_maestro,
    tape_to_maestro_native,
    observable_to_pauli_string,
    decompose_hamiltonian_to_pauli_terms,
    _apply_operation,
    operations_to_maestro,
    extract_pauli_terms,
    _to_float,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMULATOR_TYPE_MAP = {name: member for name, member in maestro.SimulatorType.__members__.items()}
_SIMULATION_TYPE_MAP = {name: member for name, member in maestro.SimulationType.__members__.items()}


def _resolve_enum(value, enum_map, enum_name: str):
    """Resolve a string or enum to the corresponding Maestro enum member."""
    if isinstance(value, str):
        if value not in enum_map:
            raise ValueError(
                f"Unknown {enum_name} '{value}'. "
                f"Valid options: {list(enum_map.keys())}"
            )
        return enum_map[value]
    return value


_PRECISIONS = ("single", "double")


def _resolve_precision(precision, use_double_precision):
    """Combine ``precision`` and the legacy ``use_double_precision`` flag.

    Returns ``"single"``, ``"double"`` or ``None`` (keep the backend default).
    """
    if precision is not None and precision not in _PRECISIONS:
        raise ValueError(
            f"Unknown precision '{precision}'. Valid options: {list(_PRECISIONS)}"
        )
    if use_double_precision is None:
        return precision
    legacy = "double" if use_double_precision else "single"
    if precision is not None and precision != legacy:
        raise ValueError(
            f"Conflicting settings: precision='{precision}' and "
            f"use_double_precision={use_double_precision}."
        )
    return legacy


def _resolve_alias(value, legacy_value, name: str, legacy_name: str):
    """Merge a renamed keyword argument with its deprecated legacy name."""
    if legacy_value is None:
        return value
    warnings.warn(
        f"'{legacy_name}' is deprecated; use '{name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if value is not None and value != legacy_value:
        raise ValueError(
            f"Conflicting settings: {name}={value} and {legacy_name}={legacy_value}."
        )
    return legacy_value


# Observable clustering landed in Maestro after the 0.3.7 release; forward it
# only when the installed SimulatorConfig exposes it.
_SUPPORTS_OBSERVABLE_CLUSTERING = hasattr(SimulatorConfig(), "max_cluster_qubits")


def _lsb_to_msb_statevector(state: np.ndarray, num_wires: int) -> np.ndarray:
    """Reorder a statevector from Maestro's LSB-first convention to
    PennyLane's MSB-first convention.

    Maestro qubit 0 is the *least* significant bit of the state index,
    but PennyLane expects qubit 0 to be the *most* significant bit.
    This amounts to reversing the bit pattern of every basis-state index.
    """
    n = num_wires
    size = 1 << n
    if len(state) < size:
        padded = np.zeros(size, dtype=state.dtype)
        padded[: len(state)] = state
        state = padded
    # Build a permutation: for each index i, reverse its n-bit representation
    perm = np.zeros(size, dtype=np.intp)
    for i in range(size):
        rev = 0
        val = i
        for _ in range(n):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        perm[i] = rev
    return state[perm]


def _measure_state(mp, state: np.ndarray, num_wires: int):
    """Measure a PennyLane-ordered statevector on standard wires.

    Unlike ``mp.process_state``, this rotates into the eigenbasis of
    non-diagonal observables (``expval(X)`` next to ``qml.state()``).
    """
    return _qubit_measure(mp, np.reshape(state, (2,) * num_wires))


def _counts_to_samples(counts: dict, num_wires: int) -> np.ndarray:
    """Expand a counts dict ``{'01': 5, '10': 3}`` into an
    ``(total_shots, num_wires)`` array of 0/1 samples."""
    samples = []
    for bitstring, count in counts.items():
        row = [int(b) for b in bitstring]
        # Pad/truncate to num_wires (should not normally be needed)
        if len(row) < num_wires:
            row = [0] * (num_wires - len(row)) + row
        for _ in range(count):
            samples.append(row)
    return np.array(samples, dtype=np.int64)


# ---------------------------------------------------------------------------
# Helpers for Trotter snapshot detection & incremental evolution
# ---------------------------------------------------------------------------

def _ops_equal(op1: qml.operation.Operator, op2: qml.operation.Operator) -> bool:
    """Check if two PennyLane operations are identical in gate, wires, and parameters.

    Parameters must match exactly: a tolerance would let a slowly varying
    schedule collapse into one repeated step and silently simulate the
    wrong angles.  Genuine Trotter steps rebuild bit-identical parameters.
    """
    if op1.name != op2.name or op1.wires != op2.wires:
        return False
    if len(op1.parameters) != len(op2.parameters):
        return False
    for p1, p2 in zip(op1.parameters, op2.parameters):
        try:
            if _to_float(p1) != _to_float(p2):
                return False
        except (TypeError, ValueError):
            if not np.array_equal(qml.math.to_numpy(p1), qml.math.to_numpy(p2)):
                return False
    return True


def _blocks_equal(block1: list, block2: list) -> bool:
    """Check if two lists of operations are identical."""
    if len(block1) != len(block2):
        return False
    return all(_ops_equal(o1, o2) for o1, o2 in zip(block1, block2))


def _detect_trotter_snapshot_pattern(operations, measurements, num_wires: int):
    """Detect if a sequence of operations is a Trotterized evolution with intermediate snapshots.

    Returns a dict with pattern details, or None if the pattern does not match.
    """
    gate_segments = []
    snapshot_clusters = []
    current_gates = []
    current_snaps = []

    for op in operations:
        if isinstance(op, Snapshot):
            if current_gates:
                gate_segments.append(current_gates)
                current_gates = []
            elif not gate_segments and not snapshot_clusters:
                gate_segments.append([])
            current_snaps.append(op)
        else:
            if current_snaps:
                snapshot_clusters.append(current_snaps)
                current_snaps = []
            current_gates.append(op)

    if current_snaps:
        snapshot_clusters.append(current_snaps)
    gate_segments.append(current_gates)

    if not snapshot_clusters:
        return None

    # Check that all snapshots have Pauli / Hamiltonian expectation values
    parsed_snapshot_terms = []
    for cluster in snapshot_clusters:
        cluster_terms = []
        for snap_op in cluster:
            m = snap_op.hyperparameters.get("measurement")
            if m is None:
                # Full state snapshot -> cannot use Pauli incremental_evolve
                return None
            if isinstance(m, (list, tuple)):
                op_terms = []
                for sub_m in m:
                    if not isinstance(sub_m, ExpectationMP):
                        return None
                    terms = extract_pauli_terms(sub_m.obs, num_wires)
                    if terms is None:
                        return None
                    op_terms.append(terms)
                cluster_terms.append((snap_op, op_terms, True))
            else:
                if not isinstance(m, ExpectationMP):
                    return None
                terms = extract_pauli_terms(m.obs, num_wires)
                if terms is None:
                    return None
                cluster_terms.append((snap_op, terms, False))
        parsed_snapshot_terms.append(cluster_terms)

    # Check terminal measurements (must also be Pauli/Hamiltonian expvals, or empty)
    terminal_terms = []
    for m in measurements:
        if not isinstance(m, ExpectationMP):
            return None
        terms = extract_pauli_terms(m.obs, num_wires)
        if terms is None:
            return None
        terminal_terms.append(terms)

    m = len(snapshot_clusters)
    G0 = gate_segments[0]
    candidate_step = None
    step_multiples = []

    if m >= 2:
        G1 = gate_segments[1]
        if not G1:
            return None

        len_G1 = len(G1)
        base_block = None
        for k in range(1, len_G1 + 1):
            if len_G1 % k == 0:
                sub = G1[:k]
                reps = len_G1 // k
                if all(_blocks_equal(G1[i * k : (i + 1) * k], sub) for i in range(reps)):
                    base_block = sub
                    break
        if not base_block:
            return None

        candidate_step = base_block
        k = len(candidate_step)

        for j in range(1, m):
            Gj = gate_segments[j]
            if len(Gj) == 0 or len(Gj) % k != 0:
                return None
            reps = len(Gj) // k
            if not all(_blocks_equal(Gj[i * k : (i + 1) * k], candidate_step) for i in range(reps)):
                return None
            step_multiples.append(reps)

        len_G0 = len(G0)
        n0 = 0
        rem_len = len_G0
        while rem_len >= k and _blocks_equal(G0[rem_len - k : rem_len], candidate_step):
            n0 += 1
            rem_len -= k

        if n0 < 1:
            return None

        init_ops = G0[:rem_len]

    else:
        # m == 1: single snapshot cluster. Find repeating suffix of G0
        len_G0 = len(G0)
        found = False
        for k in range(1, len_G0 // 2 + 1):
            sub = G0[len_G0 - k : len_G0]
            n_reps = 0
            rem_len = len_G0
            while rem_len >= k and _blocks_equal(G0[rem_len - k : rem_len], sub):
                n_reps += 1
                rem_len -= k
            if n_reps >= 1:
                candidate_step = sub
                init_ops = G0[:rem_len]
                n0 = n_reps
                found = True
                break
        if not found:
            return None

    Gm = gate_segments[m]
    k = len(candidate_step)
    if len(Gm) > 0:
        if len(Gm) % k != 0:
            return None
        n_trailing = len(Gm) // k
        if not all(_blocks_equal(Gm[i * k : (i + 1) * k], candidate_step) for i in range(n_trailing)):
            return None
    else:
        n_trailing = 0

    steps = []
    curr_step = n0
    steps.append(curr_step)
    for reps in step_multiples:
        curr_step += reps
        steps.append(curr_step)

    final_step = curr_step + n_trailing

    return {
        "init_ops": init_ops,
        "step_ops": candidate_step,
        "steps": steps,
        "snapshot_clusters": snapshot_clusters,
        "parsed_snapshot_terms": parsed_snapshot_terms,
        "terminal_terms": terminal_terms,
        "final_step": final_step,
    }


# ---------------------------------------------------------------------------
# Stopping condition for gate decomposition
# ---------------------------------------------------------------------------

def _maestro_stopping_condition(op: qml.operation.Operator) -> bool:
    """Return ``True`` for ops that the Maestro converter handles natively.

    PennyLane will decompose any op for which this returns ``False``.
    After ``defer_measurements`` has been applied, ``MidMeasureMP`` and
    ``Conditional`` nodes will already have been removed, but we keep
    them in the true-set as a safety net for when the stopping condition
    is evaluated on pre-deferred tapes.
    """
    if isinstance(op, (MidMeasureMP, Conditional, Snapshot)):
        return True
    if op.name == "Delay":
        return True
    if isinstance(op, Adjoint):
        base_name = op.base.name
        return base_name in GATE_MAP or base_name in ADJOINT_MAP
    return op.name in GATE_MAP


@qml.transform
def _broadcast_expand_unless_native(tape: QuantumScript, native):
    """Split a broadcast tape into one tape per batch element, unless the
    device evaluates the whole batch in one engine call."""
    if tape.batch_size is None or native(tape):
        return (tape,), lambda results: results[0]
    return broadcast_expand(tape)


def _row_vjps_supported(config: ExecutionConfig, tape: QuantumScript) -> bool:
    """Whether the interface accepts a VJP with one entry per batch row for
    the broadcast parameters of ``tape``.

    jax-jit rebuilds JVPs from it and cannot; torch stacks a tape's VJPs
    into one array, so every trainable parameter must be broadcast.
    """
    interface = getattr(config.interface, "value", config.interface)
    if interface == "jax-jit":
        return False
    if interface == "torch":
        trainable = qml.math.get_trainable_indices(tape.get_parameters(trainable_only=False))
        return set(trainable) <= _adjoint.batched_params(tape)
    return True


_ADJOINT_BACKENDS = ("auto", "engine", "inner_product")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@simulator_tracking
@single_tape_support
class MaestroQubitDevice(Device):
    """PennyLane device for the Maestro quantum simulator by Qoro Quantum.

    Args:
        wires (int or Iterable): Number of wires, or explicit wire labels.
        shots (int or None): Default shot count.  ``None`` → analytic mode.
        simulator_type (str): Backend engine. Options:

            - ``"QCSim"`` — Qoro's optimised CPU simulator (default)
            - ``"Gpu"`` — CUDA-accelerated GPU simulator
            - ``"CompositeQCSim"`` — Multi-node distributed CPU
            - ``"QiskitAer"`` — Integrated Qiskit Aer backend
            - ``"CompositeQiskitAer"`` — Distributed Qiskit Aer backend
            - ``"DistributedGpu"`` — State distributed across local GPUs
            - ``"DistributedMpiGpu"`` — State distributed across MPI ranks/GPUs
            - ``"QuestSim"`` — QuEST simulator

        simulation_type (str): Simulation algorithm. Options:

            - ``"Statevector"`` — Full statevector (default)
            - ``"MatrixProductState"`` — MPS / tensor-train
            - ``"Stabilizer"`` — Clifford-only stabilizer
            - ``"TensorNetwork"`` — General tensor network
            - ``"PauliPropagator"`` — Pauli propagation
            - ``"ExtendedStabilizer"`` — Extended stabilizer
            - ``"DensityMatrix"`` — Density matrix for mixed states
            - ``"MatrixProductOperator"`` — MPO simulation
            - ``"PathIntegral"`` — Path integral simulation

        max_bond_dimension (int or None): MPS truncation (default: None).
        singular_value_threshold (float or None): MPS SVD cutoff.
        precision (str or None): Floating-point precision, ``"single"`` or
            ``"double"``, for the GPU and Qiskit Aer backends.  ``None``
            (default) keeps each backend's default; QCSim always uses double.
        use_double_precision (bool or None): Legacy alias for ``precision``:
            ``True`` → ``"double"``, ``False`` → ``"single"``.
        disable_optimized_swapping (bool): Disable MPS swap optimisation
            (default: False).  Automatically enabled for native MCM
            circuits where the optimisation is incompatible with
            mid-circuit measure/reset.
        pp_coefficient_threshold (float or None): PauliPropagator — drop
            Pauli strings with smaller coefficients.
        pp_max_pauli_weight (int or None): PauliPropagator — drop Pauli
            strings acting on more qubits than this.
        pp_gates_between_trims (int or None): PauliPropagator — gates
            applied between truncation passes.
        pp_gates_between_deduplications (int or None): PauliPropagator —
            gates applied between merges of duplicate Pauli strings.
        pp_pauli_weight_threshold: Deprecated alias for ``pp_max_pauli_weight``.
        pp_steps_between_trims: Deprecated alias for ``pp_gates_between_trims``.
        enable_causal_cone_reduction (bool): Simulate only the causal cone of
            the measured observables (default: True).
        auto_reduce (bool): Detect invariant subspaces (e.g. from checks)
            and simulate the reduced register (default: True).
        enable_observable_clustering (bool): With causal-cone reduction,
            split multi-observable estimates into groups with disjoint
            lightcones, each simulated on its own register (default: True).
            Requires a Maestro release newer than 0.3.7; ignored otherwise.
        max_cluster_qubits (int or None): With observable clustering, cap
            each group at this many lightcone qubits.  ``None`` (default)
            keeps overlapping lightcones together.  Requires a Maestro
            release newer than 0.3.7; ignored otherwise.
        gate_fusion (bool or None): Fuse compatible gates.  ``None``
            (default) uses each backend's default.
        seed (int, np.random.Generator, or None): Random seed for reproducible
            stochastic simulations (sampling, measurements, noise).
        adjoint_backend (str): How analytic derivatives are computed for
            ``diff_method="adjoint"`` (and ``"best"``) on the CPU statevector
            (``simulator_type="QCSim"``, ``simulation_type="Statevector"``):

            - ``"auto"`` (default) — ``"engine"`` when the installed
              qoro-maestro provides ``AdjointProgram``, else ``"inner_product"``
            - ``"engine"`` — Maestro's batched adjoint sweep
            - ``"inner_product"`` — one ``maestro.inner_product`` per
              parameter and Pauli term

            Other configurations, and finite shots, use parameter-shift.

    Usage examples::

        # CPU statevector (default)
        dev = qml.device("maestro.qubit", wires=20)

        # GPU statevector
        dev = qml.device("maestro.qubit", wires=20, simulator_type="Gpu")

        # GPU with double precision
        dev = qml.device("maestro.qubit", wires=20,
                         simulator_type="Gpu", precision="double")

        # MPS for large qubit counts
        dev = qml.device("maestro.qubit", wires=100,
                         simulation_type="MatrixProductState",
                         max_bond_dimension=256)

        # Stabilizer for Clifford circuits
        dev = qml.device("maestro.qubit", wires=1000,
                         simulation_type="Stabilizer")

        # Reproducible random sampling with seed
        dev = qml.device("maestro.qubit", wires=5, shots=1000, seed=42)
    """

    name = "maestro.qubit"
    config_filepath = path.join(path.dirname(__file__), "config.toml")
    _debugger = None  # Snapshot compatibility — see pennylane.debugging.snapshot

    def __init__(
        self,
        wires=None,
        shots=None,
        simulator_type: str = "QCSim",
        simulation_type: str = "Statevector",
        max_bond_dimension=None,
        singular_value_threshold=None,
        precision: str | None = None,
        use_double_precision: bool | None = None,
        disable_optimized_swapping: bool = False,
        pp_coefficient_threshold=None,
        pp_max_pauli_weight=None,
        pp_gates_between_trims=None,
        pp_gates_between_deduplications=None,
        pp_pauli_weight_threshold=None,
        pp_steps_between_trims=None,
        enable_causal_cone_reduction: bool = True,
        auto_reduce: bool = True,
        enable_observable_clustering: bool = True,
        max_cluster_qubits: int | None = None,
        gate_fusion: bool | None = None,
        seed=None,
        adjoint_backend: str = "auto",
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Setting shots on device is deprecated",
            )
            super().__init__(wires=wires, shots=shots)
        self._simulator_type = _resolve_enum(
            simulator_type, _SIMULATOR_TYPE_MAP, "SimulatorType"
        )
        self._simulation_type = _resolve_enum(
            simulation_type, _SIMULATION_TYPE_MAP, "SimulationType"
        )

        # Auto-initialize GPU backend when requested
        if self._simulator_type == maestro.SimulatorType.Gpu:
            if not maestro.init_gpu():
                warnings.warn(
                    "maestro.init_gpu() failed — GPU simulation may "
                    "fall back to CPU silently.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._max_bond_dimension = max_bond_dimension
        self._singular_value_threshold = singular_value_threshold
        self._precision = _resolve_precision(precision, use_double_precision)
        self._disable_optimized_swapping = disable_optimized_swapping
        self._pp_coefficient_threshold = pp_coefficient_threshold
        self._pp_max_pauli_weight = _resolve_alias(
            pp_max_pauli_weight, pp_pauli_weight_threshold,
            "pp_max_pauli_weight", "pp_pauli_weight_threshold",
        )
        self._pp_gates_between_trims = _resolve_alias(
            pp_gates_between_trims, pp_steps_between_trims,
            "pp_gates_between_trims", "pp_steps_between_trims",
        )
        self._pp_gates_between_deduplications = pp_gates_between_deduplications
        self._enable_causal_cone_reduction = enable_causal_cone_reduction
        self._auto_reduce = auto_reduce
        self._enable_observable_clustering = enable_observable_clustering
        self._max_cluster_qubits = max_cluster_qubits
        self._gate_fusion = gate_fusion
        if not _SUPPORTS_OBSERVABLE_CLUSTERING and (
            not enable_observable_clustering or max_cluster_qubits is not None
        ):
            warnings.warn(
                "The installed qoro-maestro does not support observable "
                "clustering; 'enable_observable_clustering' and "
                "'max_cluster_qubits' are ignored.",
                UserWarning,
                stacklevel=2,
            )
        if seed is not None:
            if isinstance(seed, (int, np.integer)):
                self._seed = int(seed)
            elif hasattr(seed, "integers"):
                self._seed = int(seed.integers(0, 2**31 - 1))
            else:
                self._seed = int(seed)
        else:
            self._seed = None

        if adjoint_backend not in _ADJOINT_BACKENDS:
            raise ValueError(
                f"Unknown adjoint_backend '{adjoint_backend}'. "
                f"Valid options: {list(_ADJOINT_BACKENDS)}"
            )
        if adjoint_backend == "engine" and not _adjoint.HAS_ENGINE_ADJOINT:
            raise ValueError(
                "adjoint_backend='engine' needs a qoro-maestro build with "
                "AdjointProgram."
            )
        if adjoint_backend == "auto":
            adjoint_backend = "engine" if _adjoint.HAS_ENGINE_ADJOINT else "inner_product"
        self._adjoint_backend = adjoint_backend
        self._templates = _adjoint.TemplateCache()

    @property
    def seed(self) -> int | None:
        """Return the random seed configured on the device."""
        return self._seed

    def _build_config(self) -> SimulatorConfig:
        """Build a Maestro ``SimulatorConfig`` from device settings."""
        cfg = SimulatorConfig()
        cfg.simulator_type = self._simulator_type
        cfg.simulation_type = self._simulation_type
        if self._precision is not None:
            cfg.precision = self._precision
        if self._max_bond_dimension is not None:
            cfg.max_bond_dimension = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            cfg.singular_value_threshold = self._singular_value_threshold
        cfg.disable_optimized_swapping = self._disable_optimized_swapping
        # PauliPropagator truncation settings
        if self._pp_coefficient_threshold is not None:
            cfg.pp_coefficient_threshold = self._pp_coefficient_threshold
        if self._pp_max_pauli_weight is not None:
            cfg.pp_max_pauli_weight = self._pp_max_pauli_weight
        if self._pp_gates_between_trims is not None:
            cfg.pp_gates_between_trims = self._pp_gates_between_trims
        if self._pp_gates_between_deduplications is not None:
            cfg.pp_gates_between_deduplications = self._pp_gates_between_deduplications
        # Pre-execution reductions and gate fusion
        cfg.enable_causal_cone_reduction = self._enable_causal_cone_reduction
        cfg.auto_reduce = self._auto_reduce
        if _SUPPORTS_OBSERVABLE_CLUSTERING:
            cfg.enable_observable_clustering = self._enable_observable_clustering
            cfg.max_cluster_qubits = self._max_cluster_qubits
        cfg.gate_fusion = self._gate_fusion
        if self._seed is not None:
            cfg.seed = self._seed
        return cfg

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> CompilePipeline:
        """Return the preprocessing pipeline for this device.

        Mid-circuit measurement (MCM) strategy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        When ``mcm_method="deferred"`` (the explicit default), MCMs are
        handled by :func:`~pennylane.transforms.defer_measurements`.

        When ``mcm_method`` is ``None`` (unset), MCMs are executed
        **natively** via Maestro's built-in ``measure`` / ``reset``
        instructions with true wavefunction collapse.  This keeps the
        circuit at the physical qubit count (no ancilla inflation) and
        produces physically correct results for QEC-style circuits.
        """
        config = execution_config or ExecutionConfig()
        mcm_method = (
            config.mcm_config.mcm_method
            if config.mcm_config is not None
            else None
        )

        program = CompilePipeline()

        # ── Wire validation ──────────────────────────────────────────────
        program.add_transform(validate_device_wires, self.wires, name=self.name)

        # ── Parameter broadcasting / batching ───────────────────────────
        # Analytic expectation values and probabilities keep the batch in
        # one tape and run it in one engine call; everything else is split
        # per element. Trainable broadcast parameters stay batched only
        # with a device VJP, which returns one gradient entry per row.
        device_vjp = config.gradient_method == "adjoint" and bool(
            config.use_device_jacobian_product
        )

        def native(tape):
            trainable_ok = device_vjp and _row_vjps_supported(config, tape)
            return self._batches_natively(tape, trainable_ok=trainable_ok)

        program.add_transform(_broadcast_expand_unless_native, native=native)

        # ── MCM handling ────────────────────────────────────────────────
        # Only apply defer_measurements when explicitly requested.
        # When mcm_method is None (default), we use native MCM execution
        # which preserves the MidMeasureMP ops in the tape for the device
        # to handle directly.
        if mcm_method == "deferred":
            program.add_transform(
                defer_measurements,
                allow_postselect=False,
            )

        # ── Gate decomposition ──────────────────────────────────────────
        program.add_transform(
            _decompose,
            stopping_condition=_maestro_stopping_condition,
            name=self.name,
        )

        return program

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        results = []
        for tape in circuits:
            if tape.batch_size is not None:
                if self._batches_natively(tape, trainable_ok=True):
                    results.append(self._execute_parametric(tape))
                    continue
                tapes, fn = broadcast_expand(tape)
                tape_results = [self._execute_single(t) for t in tapes]
                results.append(fn(tape_results))
            else:
                results.append(self._execute_single(tape))
        return tuple(results)

    # ------------------------------------------------------------------
    # Parametric statevector path: batches and analytic derivatives
    # ------------------------------------------------------------------

    def _parametric_available(self) -> bool:
        """CPU statevector, where the parametric path is exact."""
        return (
            self._simulator_type == maestro.SimulatorType.QCSim
            and self._simulation_type == maestro.SimulationType.Statevector
        )

    def _max_probability_wires(self, vjp: bool = True) -> int:
        """Most ``qml.probs`` wires taken on the adjoint path."""
        if self._adjoint_backend != "engine" or not _adjoint.HAS_ENGINE_PROBABILITIES:
            return 0
        if vjp:
            return _adjoint.MAX_PROBABILITY_WIRES_VJP
        return _adjoint.MAX_PROBABILITY_WIRES_JACOBIAN

    def _batches_natively(self, tape: QuantumScript, trainable_ok: bool = False) -> bool:
        """Whether a broadcast tape runs as one batch on the engine.

        Unless ``trainable_ok``, a tape whose trainable parameters carry the
        broadcast dimension is split: a device Jacobian has no entry per
        batch element for them."""
        return (
            _adjoint.HAS_ENGINE_ADJOINT
            and self._parametric_available()
            and _adjoint.is_supported(
                tape,
                trainable=(),
                max_probability_wires=_adjoint.MAX_PROBABILITY_WIRES_VJP
                if _adjoint.HAS_ENGINE_PROBABILITIES
                else 0,
            )
            and (trainable_ok or not _adjoint.batched_trainable(tape))
        )

    def _prepare_parametric(self, tape: QuantumScript):
        tape = tape.map_to_standard_wires()
        num_wires = max(len(tape.wires), 1)
        template = self._templates.get(tape)
        angles, batch = template.angles(tape)
        readouts, widths = _adjoint.readouts(tape, num_wires)
        return tape, template, angles, batch, readouts, widths

    def _execute_parametric(self, tape: QuantumScript) -> Result:
        """Expectation values and probabilities of a (possibly broadcast) tape."""
        _, template, angles, batch, readouts, widths = self._prepare_parametric(tape)
        raw = template.program.expectation_values(
            readouts, angles, self._build_config()
        )
        return _adjoint.format_values(raw["expectation_values"], batch, widths)

    def _parametric_jacobian(self, tape: QuantumScript):
        """Values ``(rows, V)``, Jacobian ``(rows, V, trainable)``, batch size
        and the number of values per measurement."""
        tape, template, angles, batch, observables, widths = self._prepare_parametric(tape)
        config = self._build_config()
        if self._adjoint_backend == "engine":
            raw = template.program.jacobian(observables, angles, config)
            values, jac = raw["expectation_values"], raw["jacobian"]
        else:
            values = np.empty((len(angles), len(observables)))
            jac = np.empty((len(angles), len(observables), len(template.slot_index)))
            all_paulis = []
            term_slices = []
            for terms in observables:
                start = len(all_paulis)
                coeffs = [c for c, _ in terms]
                all_paulis.extend(ps for _, ps in terms)
                term_slices.append((start, len(terms), coeffs))

            for row, a in enumerate(angles):
                qc = template.circuit(a)
                if all_paulis:
                    ev = qc.estimate(all_paulis, config)["expectation_values"]
                    for m, (start, count, coeffs) in enumerate(term_slices):
                        values[row, m] = sum(
                            c * v for c, v in zip(coeffs, ev[start:start + count])
                        )
                else:
                    values[row, :] = 0.0
                jac[row] = _adjoint.inner_product_jacobian(template, a, observables, config)
        jac = (jac @ template.selector)[:, :, list(tape.trainable_params)]
        return values, jac, batch, widths

    def _parametric_vjp(self, tape: QuantumScript, dy):
        """Values ``(rows, V)``, the VJP per trainable parameter (one entry
        per row for broadcast parameters), batch size and the number of
        values per measurement."""
        if self._adjoint_backend != "engine":
            values, jac, batch, widths = self._parametric_jacobian(tape)
            cot = _adjoint.cotangent_matrix(dy, widths, batch)
            row_vjp = np.einsum("rm,rmk->rk", cot, jac)
            return values, _adjoint.reduce_vjp(row_vjp, tape), batch, widths
        mapped, template, angles, batch, readouts, widths = self._prepare_parametric(tape)
        cot = _adjoint.cotangent_matrix(dy, widths, batch)
        raw = template.program.vjp(readouts, cot, angles, self._build_config())
        row_vjp = (raw["vjp"] @ template.selector)[:, list(mapped.trainable_params)]
        return raw["expectation_values"], _adjoint.reduce_vjp(row_vjp, tape), batch, widths

    def _check_adjoint(self, tape: QuantumScript, vjp: bool = False):
        if not (
            self._parametric_available()
            and _adjoint.is_supported(
                tape,
                trainable=tape.trainable_params,
                max_probability_wires=self._max_probability_wires(vjp=vjp),
            )
        ):
            raise DeviceError(
                f"{self.name} cannot compute adjoint derivatives of this circuit."
            )

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------

    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Adjoint derivatives of analytic Pauli/Hamiltonian expectation values
        on the CPU statevector, and of ``qml.probs`` with the engine backend
        (up to 10 wires with a device Jacobian, 24 with a device VJP). Other
        cases fall back to PennyLane's gradient transforms."""
        if execution_config is None:
            return self._parametric_available()
        if execution_config.gradient_method not in ("adjoint", "best"):
            return False
        if not self._parametric_available():
            return False
        if circuit is None:
            return True
        vjp = bool(execution_config.use_device_jacobian_product)
        return _adjoint.is_supported(
            circuit, max_probability_wires=self._max_probability_wires(vjp)
        )

    def setup_execution_config(
        self,
        config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> ExecutionConfig:
        config = config or ExecutionConfig()
        if config.gradient_method in ("best", "adjoint"):
            vjp = config.use_device_jacobian_product
            if vjp is None or (
                config.gradient_method == "best"
                and circuit is not None
                and _adjoint.batched_trainable(circuit)
                and _row_vjps_supported(config, circuit)
            ):
                # A device Jacobian would split the batch into one tape per
                # element; the VJP keeps it in one engine call.
                vjp = True
            if self.supports_derivatives(
                replace(config, use_device_jacobian_product=vjp), circuit
            ):
                updates = {
                    "gradient_method": "adjoint",
                    "use_device_jacobian_product": vjp,
                    "use_device_gradient": True
                    if config.use_device_gradient is None
                    else config.use_device_gradient,
                }
                if config.grad_on_execution is None:
                    updates["grad_on_execution"] = not vjp
                config = replace(config, **updates)
            elif config.gradient_method == "best":
                config = replace(config, gradient_method=qml.gradients.param_shift)
        return super().setup_execution_config(config, circuit)

    def compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        return self.execute_and_compute_derivatives(circuits, execution_config)[1]

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        results, jacs = [], []
        for tape in circuits:
            self._check_adjoint(tape, vjp=False)
            values, jac, batch, widths = self._parametric_jacobian(tape)
            results.append(_adjoint.format_values(values, batch, widths))
            jacs.append(_adjoint.format_jacobian(jac, batch, widths))
        return tuple(results), tuple(jacs)

    def supports_vjp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        return self.supports_derivatives(execution_config, circuit)

    def compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents,
        execution_config: ExecutionConfig | None = None,
    ):
        return self.execute_and_compute_vjp(circuits, cotangents, execution_config)[1]

    def execute_and_compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents,
        execution_config: ExecutionConfig | None = None,
    ):
        results, vjps = [], []
        for tape, dy in zip(circuits, cotangents):
            self._check_adjoint(tape, vjp=True)
            values, vjp, batch, widths = self._parametric_vjp(tape, dy)
            results.append(_adjoint.format_values(values, batch, widths))
            vjps.append(vjp)
        return tuple(results), tuple(vjps)

    # ------------------------------------------------------------------
    # Single tape execution
    # ------------------------------------------------------------------

    def _execute_single(self, tape: QuantumScript) -> Result:
        """Execute a single (already preprocessed) tape."""
        # Map wires to consecutive 0-indexed integers
        tape = tape.map_to_standard_wires()
        num_wires = len(tape.wires) if len(tape.wires) > 0 else 1

        # ── Snapshot path: incremental circuit building ──
        has_snapshots = any(
            isinstance(op, Snapshot) for op in tape.operations
        )
        if has_snapshots and not tape.shots:
            return self._execute_with_snapshots(tape, num_wires)

        # Check if this tape has native MCMs (not yet deferred)
        has_mcm = any(
            isinstance(op, MidMeasureMP) for op in tape.operations
        )

        is_analytic = not tape.shots

        if has_mcm and not is_analytic:
            # Native MCM path — requires finite shots
            shot_results = []
            for shot_copy in tape.shots:
                shot_results.append(
                    self._execute_native_mcm(tape, num_wires, shot_copy)
                )
            if not tape.shots.has_partitioned_shots:
                return shot_results[0]
            return tuple(shot_results)

        if is_analytic:
            return self._execute_analytic(tape, num_wires)

        # Finite shots — may have a shot vector
        shot_results = []
        for shot_copy in tape.shots:
            shot_results.append(
                self._execute_finite_shots(tape, num_wires, shot_copy)
            )

        if not tape.shots.has_partitioned_shots:
            return shot_results[0]
        return tuple(shot_results)

    # ------------------------------------------------------------------
    # Analytic execution  (shots=None)
    # ------------------------------------------------------------------

    def _execute_analytic(self, tape: QuantumScript, num_wires: int) -> Result:
        """Compute results from exact simulation.

        Fast path: if every measurement is an expectation value of a Pauli
        observable, delegate to Maestro's ``estimate()`` which computes
        exact expectation values without materialising the full 2^n
        statevector.  This is dramatically faster for large qubit counts.

        Slow path: fall back to full statevector extraction for anything
        else (variance, probabilities, state, Hermitian observables, etc.).
        """
        # ── Fast path: all-Pauli expval → use estimate() ──
        all_pauli_expval = all(
            isinstance(mp, ExpectationMP)
            and observable_to_pauli_string(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )
        if all_pauli_expval:
            return self._execute_estimate(tape, num_wires)

        # ── Fast path: Hamiltonian / Sum expval → batched estimate() ──
        all_hamiltonian_expval = all(
            isinstance(mp, ExpectationMP)
            and decompose_hamiltonian_to_pauli_terms(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )
        if all_hamiltonian_expval:
            return self._execute_hamiltonian(tape, num_wires)

        # ── Slow path: full statevector ──
        qc = tape_to_maestro(tape, num_wires)
        config = self._build_config()

        amplitudes = qc.get_statevector(config)
        state = np.array(amplitudes, dtype=np.complex128)
        state = _lsb_to_msb_statevector(state, num_wires)

        results = tuple(
            _measure_state(mp, state, num_wires) for mp in tape.measurements
        )
        if len(tape.measurements) == 1:
            return results[0]
        return results

    # ------------------------------------------------------------------
    # Finite-shots execution
    # ------------------------------------------------------------------

    def _execute_finite_shots(
        self, tape: QuantumScript, num_wires: int, shots: int
    ) -> Result:
        """Execute with a finite number of shots."""
        # Check if ALL measurements can use the estimate fast-path
        # (Pauli expvals only — no samples, counts, probs, etc.)
        all_pauli_expval = all(
            isinstance(mp, ExpectationMP)
            and observable_to_pauli_string(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )

        if all_pauli_expval:
            return self._execute_estimate(tape, num_wires)

        # General path: sample from counts
        qc = tape_to_maestro(tape, num_wires)
        qc.measure_all()
        config = self._build_config()

        raw = qc.execute(config, shots=shots)
        counts = raw["counts"]

        samples = _counts_to_samples(counts, num_wires)

        results = tuple(
            mp.process_samples(samples, tape.wires) for mp in tape.measurements
        )
        if len(tape.measurements) == 1:
            return results[0]
        return results

    # ------------------------------------------------------------------
    # Native MCM execution (true wavefunction collapse)
    # ------------------------------------------------------------------

    def _execute_native_mcm(
        self, tape: QuantumScript, num_wires: int, shots: int
    ) -> Result:
        """Execute a tape with native mid-circuit measurements.

        Uses Maestro's built-in ``measure([(qubit, classical_bit)])`` and
        ``reset()`` instructions.  The circuit stays at the physical qubit
        count with true wavefunction collapse — no ancilla inflation.

        The counts bitstring from Maestro has ``num_classical_bits``
        positions for MCM outcomes, followed by ``num_wires`` positions
        for the final qubit-state measurement.
        """
        # Convert tape with native MCM tracking
        qc, tracker = tape_to_maestro_native(tape, num_wires)

        # Add final measurement of all qubits.
        # Classical bits for final measurement start after MCM bits.
        n_mcm_bits = tracker.num_classical_bits
        final_pairs = [
            (q, n_mcm_bits + q) for q in range(num_wires)
        ]
        qc.measure(final_pairs)

        total_classical_bits = n_mcm_bits + num_wires

        config = self._build_config()
        # MPS optimized swapping is incompatible with mid-circuit
        # measure/reset — disable it for native MCM circuits.
        config.disable_optimized_swapping = True

        raw = qc.execute(config, shots=shots)
        counts = raw["counts"]

        # Build samples array: (shots, total_classical_bits)
        all_samples = _counts_to_samples(counts, total_classical_bits)

        # Split into MCM samples and qubit samples
        mcm_samples = all_samples[:, :n_mcm_bits]     # (shots, n_mcm_bits)
        qubit_samples = all_samples[:, n_mcm_bits:]    # (shots, num_wires)

        # Process each measurement
        results = []
        for mp in tape.measurements:
            if hasattr(mp, 'mv') and mp.mv is not None:
                # This is a measurement of an MCM value (e.g. qml.sample(m))
                # Find which classical bit(s) this MeasurementValue references
                mcm_ids = []
                for mid_mp in mp.mv.measurements:
                    meas_uid = getattr(mid_mp, "meas_uid", None)
                    if meas_uid is not None and meas_uid in tracker.id_to_bit:
                        mcm_ids.append(meas_uid)
                    elif getattr(mid_mp, "id", None) is not None and mid_mp.id in tracker.id_to_bit:
                        mcm_ids.append(mid_mp.id)
                    elif id(mid_mp) in tracker.id_to_bit:
                        mcm_ids.append(id(mid_mp))
                if len(mcm_ids) == 1 and mcm_ids[0] in tracker.id_to_bit:
                    bit_idx = tracker.id_to_bit[mcm_ids[0]]
                    # Extract the column for this classical bit
                    col = mcm_samples[:, bit_idx]
                    # For SampleMP, return the raw array
                    if isinstance(mp, SampleMP):
                        results.append(col)
                    else:
                        # For other types, build a single-column samples array
                        single_sample = col.reshape(-1, 1)
                        results.append(
                            mp.process_samples(single_sample, qml.wires.Wires([0]))
                        )
                else:
                    # Composite MeasurementValue — apply processing_fn
                    mcm_cols = []
                    for mid_id in mcm_ids:
                        if mid_id in tracker.id_to_bit:
                            mcm_cols.append(mcm_samples[:, tracker.id_to_bit[mid_id]])
                    if mcm_cols:
                        combined = np.column_stack(mcm_cols)
                        results.append(
                            mp.process_samples(combined, qml.wires.Wires(range(len(mcm_cols))))
                        )
                    else:
                        raise ValueError(
                            f"MCM id(s) {mcm_ids} not found in tracker. "
                            "Ensure all mid-circuit measurements are tracked."
                        )
            else:
                # Terminal measurement on qubits (expval, sample, counts, etc.)
                results.append(
                    mp.process_samples(qubit_samples, tape.wires)
                )

        if len(results) == 1:
            return results[0]
        return tuple(results)

    # ------------------------------------------------------------------
    # Fast Pauli-expectation path via Maestro estimate()
    # ------------------------------------------------------------------

    def _execute_estimate(
        self, tape: QuantumScript, num_wires: int
    ) -> Result:
        """Use Maestro's estimate() for pure-Pauli expval measurements."""
        qc = tape_to_maestro(tape, num_wires)

        pauli_strings = []
        for mp in tape.measurements:
            ps = observable_to_pauli_string(mp.obs, num_wires)
            pauli_strings.append(ps)

        config = self._build_config()

        raw = qc.estimate(pauli_strings, config)
        exp_vals = raw["expectation_values"]

        # Handle SProd (scalar * Pauli) — multiply by the scalar
        results = []
        for mp, ev in zip(tape.measurements, exp_vals):
            if isinstance(mp.obs, qml.ops.SProd):
                ev = float(mp.obs.scalar) * ev
            results.append(np.float64(ev))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    # ------------------------------------------------------------------
    # Batched Hamiltonian path via Maestro estimate()
    # ------------------------------------------------------------------

    def _execute_hamiltonian(
        self, tape: QuantumScript, num_wires: int
    ) -> Result:
        """Evaluate Hamiltonian/Sum expvals via a single batched estimate().

        All Pauli terms across all Hamiltonian measurements are collected
        into one list, sent to Maestro's ``estimate()`` in a single C++
        call, and the weighted sum is computed in Python.
        """
        qc = tape_to_maestro(tape, num_wires)

        # Collect all Pauli terms and track which measurement owns which
        all_pauli_strings = []
        term_slices = []  # (start_idx, count, coeffs) per measurement

        for mp in tape.measurements:
            terms = decompose_hamiltonian_to_pauli_terms(mp.obs, num_wires)
            start = len(all_pauli_strings)
            coeffs = []
            for coeff, ps in terms:
                coeffs.append(coeff)
                all_pauli_strings.append(ps)
            term_slices.append((start, len(terms), coeffs))

        config = self._build_config()

        raw = qc.estimate(all_pauli_strings, config)
        all_exp_vals = raw["expectation_values"]

        # Reconstruct each Hamiltonian expval as Σ cᵢ⟨Pᵢ⟩
        results = []
        for start, count, coeffs in term_slices:
            expval = sum(
                c * ev for c, ev in zip(coeffs, all_exp_vals[start:start + count])
            )
            results.append(np.float64(expval))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    # ------------------------------------------------------------------
    # Snapshot execution (incremental circuit building / persistent evolution)
    # ------------------------------------------------------------------

    def _execute_with_snapshots(
        self, tape: QuantumScript, num_wires: int
    ) -> Result:
        """Execute a tape containing ``qml.Snapshot`` operations.

        Fast path: if the tape follows a Trotterized time-evolution pattern
        (an optional initial state followed by repeated step blocks separated
        by Pauli/Hamiltonian snapshots), uses Maestro's ``incremental_evolve``
        for persistent-state execution with O(N) cost instead of O(N^2).

        Fallback path: builds a single Maestro ``QuantumCircuit`` incrementally,
        batching Pauli expvals into ``estimate()`` calls and non-Pauli snapshots
        into ``get_statevector()``.
        """
        # ── Fast path: persistent incremental evolve ──
        pattern = _detect_trotter_snapshot_pattern(
            tape.operations, tape.measurements, num_wires
        )
        if pattern is not None:
            return self._execute_incremental_snapshots(tape, num_wires, pattern)

        # ── Fallback: sequential accumulating circuit building ──
        from maestro.circuits import QuantumCircuit

        qc = QuantumCircuit()
        # Allocate qubits (same pattern as tape_to_maestro: Z;Z = I)
        for q in range(num_wires):
            qc.z(q)
            qc.z(q)

        config = self._build_config()

        # Collect consecutive Snapshot ops so we can batch estimate()
        pending_snapshots = []

        for op in tape.operations:
            if isinstance(op, Snapshot):
                if self._debugger is not None and self._debugger.active:
                    pending_snapshots.append(op)
                # Snapshot is a no-op on the quantum state
            else:
                # Flush any pending snapshots before applying the next gate
                if pending_snapshots:
                    self._flush_snapshots(
                        qc, pending_snapshots, num_wires, tape, config
                    )
                    pending_snapshots = []
                _apply_operation(qc, op)

        # Flush any trailing snapshots (at end of circuit, before terminal measurements)
        if pending_snapshots:
            self._flush_snapshots(
                qc, pending_snapshots, num_wires, tape, config
            )

        # ── Terminal measurements ──
        if not tape.measurements:
            return ()

        # Fast path: all Pauli expvals → estimate()
        all_pauli_expval = all(
            isinstance(mp, ExpectationMP)
            and observable_to_pauli_string(mp.obs, num_wires) is not None
            for mp in tape.measurements
        )
        if all_pauli_expval:
            pauli_strings = [
                observable_to_pauli_string(mp.obs, num_wires)
                for mp in tape.measurements
            ]
            raw = qc.estimate(pauli_strings, config)
            exp_vals = raw["expectation_values"]
            results = []
            for mp, ev in zip(tape.measurements, exp_vals):
                if isinstance(mp.obs, qml.ops.SProd):
                    ev = float(mp.obs.scalar) * ev
                results.append(np.float64(ev))
            if len(results) == 1:
                return results[0]
            return tuple(results)

        # Fast path: Hamiltonian/Sum expvals → batched estimate()
        all_hamiltonian_expval = all(
            isinstance(mp, ExpectationMP)
            and decompose_hamiltonian_to_pauli_terms(mp.obs, num_wires)
            is not None
            for mp in tape.measurements
        )
        if all_hamiltonian_expval:
            all_pauli_strings = []
            term_slices = []
            for mp in tape.measurements:
                terms = decompose_hamiltonian_to_pauli_terms(
                    mp.obs, num_wires
                )
                start = len(all_pauli_strings)
                coeffs = []
                for coeff, ps in terms:
                    coeffs.append(coeff)
                    all_pauli_strings.append(ps)
                term_slices.append((start, len(terms), coeffs))
            raw = qc.estimate(all_pauli_strings, config)
            all_exp_vals = raw["expectation_values"]
            results = []
            for start, count, coeffs in term_slices:
                expval = sum(
                    c * ev
                    for c, ev in zip(
                        coeffs, all_exp_vals[start : start + count]
                    )
                )
                results.append(np.float64(expval))
            if len(results) == 1:
                return results[0]
            return tuple(results)

        # Slow path: full statevector
        amplitudes = qc.get_statevector(config)
        state = np.array(amplitudes, dtype=np.complex128)
        state = _lsb_to_msb_statevector(state, num_wires)
        results = tuple(
            _measure_state(mp, state, num_wires) for mp in tape.measurements
        )
        if len(tape.measurements) == 1:
            return results[0]
        return results

    def _flush_snapshots(self, qc, snapshot_ops, num_wires, tape, config):
        """Evaluate a batch of consecutive Snapshot ops efficiently.

        All Pauli-expval snapshots in the batch are combined into a single
        ``estimate()`` call.  Non-Pauli snapshots (e.g. full state) fall
        back to ``get_statevector()`` (called at most once per batch).
        """
        # Separate into Pauli-batchable and fallback snapshots
        pauli_batch = []        # (tag, measurement, pauli_string)
        hamiltonian_batch = []  # (tag, measurement, [(coeff, pauli_string), ...])
        fallback_batch = []     # (tag, measurement)

        for op in snapshot_ops:
            measurement = op.hyperparameters["measurement"]
            tag = (
                op.tag
                if op.tag is not None
                else len(self._debugger.snapshots) + len(pauli_batch) + len(fallback_batch)
            )

            if isinstance(measurement, ExpectationMP):
                # Try single Pauli string first
                ps = observable_to_pauli_string(measurement.obs, num_wires)
                if ps is not None:
                    pauli_batch.append((tag, measurement, ps))
                    continue

                # Try Hamiltonian/Sum decomposition into Pauli terms
                terms = decompose_hamiltonian_to_pauli_terms(
                    measurement.obs, num_wires
                )
                if terms is not None:
                    hamiltonian_batch.append((tag, measurement, terms))
                    continue

            fallback_batch.append((tag, measurement))

        # ── Fast path: batch all Pauli expvals into ONE estimate() call ──
        # Collect ALL Pauli strings from both single-Pauli and Hamiltonian
        # snapshots into one batch for a single estimate() call.
        all_pauli_strings = []

        # Single-Pauli snapshot indices
        pauli_indices = []  # (tag, measurement, idx_in_all)
        for tag, measurement, ps in pauli_batch:
            pauli_indices.append((tag, measurement, len(all_pauli_strings)))
            all_pauli_strings.append(ps)

        # Hamiltonian snapshot slices
        ham_slices = []  # (tag, start, count, coeffs)
        for tag, measurement, terms in hamiltonian_batch:
            start = len(all_pauli_strings)
            coeffs = []
            for coeff, ps in terms:
                coeffs.append(coeff)
                all_pauli_strings.append(ps)
            ham_slices.append((tag, start, len(terms), coeffs))

        if all_pauli_strings:
            raw = qc.estimate(all_pauli_strings, config)
            exp_vals = raw["expectation_values"]

            # Store single-Pauli snapshots
            for tag, measurement, idx in pauli_indices:
                ev = exp_vals[idx]
                if isinstance(measurement.obs, qml.ops.SProd):
                    ev = float(measurement.obs.scalar) * ev
                self._store_snapshot(tag, np.float64(ev))

            # Store Hamiltonian snapshots (weighted sum)
            for tag, start, count, coeffs in ham_slices:
                expval = sum(
                    c * ev
                    for c, ev in zip(coeffs, exp_vals[start : start + count])
                )
                self._store_snapshot(tag, np.float64(expval))

        # ── Fallback: statevector for non-Pauli measurements ──
        if fallback_batch:
            amplitudes = qc.get_statevector(config)
            state = np.array(amplitudes, dtype=np.complex128)
            state = _lsb_to_msb_statevector(state, num_wires)

            for tag, measurement in fallback_batch:
                result = _measure_state(measurement, state, num_wires)
                self._store_snapshot(tag, result)

    def _store_snapshot(self, tag, result):
        """Store a snapshot result in the debugger (same logic as default.qubit)."""
        if tag not in self._debugger.snapshots:
            self._debugger.snapshots[tag] = result
        elif isinstance(self._debugger.snapshots[tag], list):
            self._debugger.snapshots[tag].append(result)
        else:
            self._debugger.snapshots[tag] = [
                self._debugger.snapshots[tag],
                result,
            ]

    def _execute_incremental_snapshots(
        self, tape: QuantumScript, num_wires: int, pattern: dict
    ) -> Result:
        """Execute Trotterized evolution with snapshots via maestro.incremental_evolve."""
        init_circuit = operations_to_maestro(pattern["init_ops"], num_wires)
        trotter_step = operations_to_maestro(pattern["step_ops"], num_wires)

        all_paulis = []

        def _add_pauli(ps: str):
            if ps not in all_paulis:
                all_paulis.append(ps)

        for cluster_terms in pattern["parsed_snapshot_terms"]:
            for snap_op, terms, is_list in cluster_terms:
                if is_list:
                    for sub_terms in terms:
                        for coeff, ps in sub_terms:
                            _add_pauli(ps)
                else:
                    for coeff, ps in terms:
                        _add_pauli(ps)

        for terms in pattern["terminal_terms"]:
            for coeff, ps in terms:
                _add_pauli(ps)

        measure_at_steps = sorted(
            set(
                pattern["steps"]
                + ([pattern["final_step"]] if pattern["terminal_terms"] else [])
            )
        )

        config = self._build_config()

        raw = maestro.incremental_evolve(
            init_circuit,
            trotter_step,
            measure_at_steps,
            all_paulis,
            config,
        )

        raw_exp = raw["expectation_values"]
        step_val_map = {}
        for idx, s in enumerate(measure_at_steps):
            step_val_map[s] = {
                ps: raw_exp[idx][p_idx] for p_idx, ps in enumerate(all_paulis)
            }

        # Populate snapshots in debugger
        if self._debugger is not None and self._debugger.active:
            for step, cluster_terms in zip(
                pattern["steps"], pattern["parsed_snapshot_terms"]
            ):
                vals = step_val_map[step]
                for snap_op, terms, is_list in cluster_terms:
                    tag = (
                        snap_op.tag
                        if snap_op.tag is not None
                        else len(self._debugger.snapshots)
                    )
                    if is_list:
                        res_list = [
                            np.float64(sum(c * vals[ps] for c, ps in sub_terms))
                            for sub_terms in terms
                        ]
                        self._store_snapshot(tag, res_list)
                    else:
                        val = sum(c * vals[ps] for c, ps in terms)
                        self._store_snapshot(tag, np.float64(val))

        # Reconstruct terminal measurements
        if not pattern["terminal_terms"]:
            return ()

        final_vals = step_val_map[pattern["final_step"]]
        results = []
        for terms in pattern["terminal_terms"]:
            val = sum(c * final_vals[ps] for c, ps in terms)
            results.append(np.float64(val))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def incremental_evolve(
        self,
        init,
        trotter_step,
        measure_at_steps,
        observables,
    ):
        """Perform persistent incremental time evolution using Maestro's incremental_evolve.

        Executes ``init`` once, then applies ``trotter_step`` incrementally, evaluating
        expectation values for ``observables`` at each step in ``measure_at_steps``
        without re-simulating from scratch.

        Args:
            init (callable, list[Operator], or QuantumCircuit): Initial state preparation.
            trotter_step (callable, list[Operator], or QuantumCircuit): Single Trotter step.
            measure_at_steps (Sequence[int]): Step indices at which to measure.
            observables (list[Operator] or list[str]): Observables to evaluate.

        Returns:
            dict: with keys:
                - ``"expectation_values"``: NumPy array of shape ``(len(measure_at_steps), len(observables))``
                - ``"steps"``: list of measured step indices
                - ``"time_taken"``: total elapsed seconds
                - ``"time_per_step"``: seconds per step interval
                - ``"dynamic_bond_dims"``: bond dimensions per step (for MPS)
                - ``"max_bond_dim_reached"``: max bond dimension reached
        """
        num_wires = len(self.wires) if len(self.wires) > 0 else 1
        # Map device wire labels to consecutive 0-indexed qubits
        wire_map = {w: i for i, w in enumerate(self.wires)}

        def _to_qc(item):
            if isinstance(item, maestro.circuits.QuantumCircuit):
                return item
            if callable(item):
                with qml.queuing.AnnotatedQueue() as q:
                    item()
                ops = q.queue
            elif hasattr(item, "operations"):
                ops = item.operations
            elif isinstance(item, (list, tuple)):
                ops = list(item)
            else:
                raise TypeError(f"Unsupported circuit type for incremental_evolve: {type(item)}")

            tape = qml.tape.QuantumScript(ops)
            pipeline = self.preprocess_transforms()
            tapes, _ = pipeline([tape])
            ops = tapes[0].operations
            if wire_map:
                ops = [qml.map_wires(op, wire_map) for op in ops]
            return operations_to_maestro(ops, num_wires)

        init_circuit = _to_qc(init)
        step_circuit = _to_qc(trotter_step)

        obs_terms = []
        all_paulis = []

        def _add_p(ps):
            if ps not in all_paulis:
                all_paulis.append(ps)

        for obs in observables:
            if isinstance(obs, str):
                _add_p(obs)
                obs_terms.append([(1.0, obs)])
            else:
                if wire_map:
                    obs = qml.map_wires(obs, wire_map)
                terms = extract_pauli_terms(obs, num_wires)
                if terms is None:
                    raise ValueError(f"Observable {obs} cannot be represented as a Pauli string.")
                for c, ps in terms:
                    _add_p(ps)
                obs_terms.append(terms)

        config = self._build_config()
        raw = maestro.incremental_evolve(
            init_circuit,
            step_circuit,
            sorted(measure_at_steps),
            all_paulis,
            config,
        )

        raw_exp = raw["expectation_values"]
        measured_steps = raw["steps"]

        results_matrix = np.zeros((len(measured_steps), len(observables)), dtype=np.float64)
        for idx, s in enumerate(measured_steps):
            vals = {ps: raw_exp[idx][p_idx] for p_idx, ps in enumerate(all_paulis)}
            for obs_idx, terms in enumerate(obs_terms):
                results_matrix[idx, obs_idx] = sum(c * vals[ps] for c, ps in terms)

        return {
            "expectation_values": results_matrix,
            "steps": measured_steps,
            "time_taken": raw.get("time_taken"),
            "time_per_step": raw.get("time_per_step"),
            "dynamic_bond_dims": raw.get("dynamic_bond_dims"),
            "max_bond_dim_reached": raw.get("max_bond_dim_reached"),
        }

