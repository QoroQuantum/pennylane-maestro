"""MaestroQubitDevice — PennyLane device backed by the Maestro simulator.

Implements the **new** ``pennylane.devices.Device`` interface (schema 3).
"""

import warnings
from os import path
from typing import Union

import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import decompose as _decompose
from pennylane.measurements import ExpectationMP, MidMeasureMP, SampleMP
from pennylane.ops import Snapshot
from pennylane.ops.op_math import Adjoint, Conditional
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import defer_measurements
try:
    from pennylane.transforms.core import CompilePipeline
except ImportError:
    from pennylane.transforms.core import TransformProgram as CompilePipeline
from pennylane.typing import Result, ResultBatch

import maestro
from maestro import SimulatorConfig

from pennylane_maestro.converter import (
    GATE_MAP,
    ADJOINT_MAP,
    tape_to_maestro,
    tape_to_maestro_native,
    MCMTracker,
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
    """Check if two PennyLane operations are identical in gate, wires, and parameters."""
    if op1.name != op2.name or op1.wires != op2.wires:
        return False
    if len(op1.parameters) != len(op2.parameters):
        return False
    for p1, p2 in zip(op1.parameters, op2.parameters):
        try:
            if not np.isclose(_to_float(p1), _to_float(p2)):
                return False
        except (TypeError, ValueError):
            if p1 != p2:
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
    if isinstance(op, Adjoint):
        base_name = op.base.name
        return base_name in GATE_MAP or base_name in ADJOINT_MAP
    return op.name in GATE_MAP


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

        simulation_type (str): Simulation algorithm. Options:

            - ``"Statevector"`` — Full statevector (default)
            - ``"MatrixProductState"`` — MPS / tensor-train
            - ``"Stabilizer"`` — Clifford-only stabilizer
            - ``"TensorNetwork"`` — General tensor network
            - ``"PauliPropagator"`` — Pauli propagation
            - ``"ExtendedStabilizer"`` — Extended stabilizer

        max_bond_dimension (int or None): MPS truncation (default: None).
        singular_value_threshold (float or None): MPS SVD cutoff.
        use_double_precision (bool): Use FP64 on GPU (default: False).
        disable_optimized_swapping (bool): Disable MPS swap optimisation
            (default: False).  Automatically enabled for native MCM
            circuits where the optimisation is incompatible with
            mid-circuit measure/reset.

    Usage examples::

        # CPU statevector (default)
        dev = qml.device("maestro.qubit", wires=20)

        # GPU statevector
        dev = qml.device("maestro.qubit", wires=20, simulator_type="Gpu")

        # GPU with double precision
        dev = qml.device("maestro.qubit", wires=20,
                         simulator_type="Gpu", use_double_precision=True)

        # MPS for large qubit counts
        dev = qml.device("maestro.qubit", wires=100,
                         simulation_type="MatrixProductState",
                         max_bond_dimension=256)

        # Stabilizer for Clifford circuits
        dev = qml.device("maestro.qubit", wires=1000,
                         simulation_type="Stabilizer")
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
        use_double_precision: bool = False,
        disable_optimized_swapping: bool = False,
        pp_coefficient_threshold=None,
        pp_pauli_weight_threshold=None,
        pp_steps_between_trims=None,
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
        self._use_double_precision = use_double_precision
        self._disable_optimized_swapping = disable_optimized_swapping
        self._pp_coefficient_threshold = pp_coefficient_threshold
        self._pp_pauli_weight_threshold = pp_pauli_weight_threshold
        self._pp_steps_between_trims = pp_steps_between_trims

    def _build_config(self) -> SimulatorConfig:
        """Build a Maestro ``SimulatorConfig`` from device settings."""
        cfg = SimulatorConfig()
        cfg.simulator_type = self._simulator_type
        cfg.simulation_type = self._simulation_type
        cfg.use_double_precision = self._use_double_precision
        if self._max_bond_dimension is not None:
            cfg.max_bond_dimension = self._max_bond_dimension
        if self._singular_value_threshold is not None:
            cfg.singular_value_threshold = self._singular_value_threshold
        cfg.disable_optimized_swapping = self._disable_optimized_swapping
        # PauliPropagator truncation settings
        if self._pp_coefficient_threshold is not None:
            cfg.pp_coefficient_threshold = self._pp_coefficient_threshold
        if self._pp_pauli_weight_threshold is not None:
            cfg.pp_pauli_weight_threshold = self._pp_pauli_weight_threshold
        if self._pp_steps_between_trims is not None:
            cfg.pp_steps_between_trims = self._pp_steps_between_trims
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

        from pennylane.devices.preprocess import validate_device_wires
        from pennylane.transforms import broadcast_expand

        program = CompilePipeline()

        # ── Wire validation ──────────────────────────────────────────────
        program.add_transform(validate_device_wires, self.wires, name=self.name)

        # ── Parameter broadcasting / batching ───────────────────────────
        program.add_transform(broadcast_expand)

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
    ) -> Union[Result, ResultBatch]:
        results = []
        for tape in circuits:
            if tape.batch_size is not None:
                from pennylane.transforms import broadcast_expand
                tapes, fn = broadcast_expand(tape)
                tape_results = [self._execute_single(t) for t in tapes]
                results.append(fn(tape_results))
            else:
                results.append(self._execute_single(tape))
        return tuple(results)

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
            mp.process_state(state, tape.wires) for mp in tape.measurements
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
            mp.process_state(state, tape.wires) for mp in tape.measurements
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
                result = measurement.process_state(state, tape.wires)
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
            list(
                set(
                    pattern["steps"]
                    + ([pattern["final_step"]] if pattern["terminal_terms"] else [])
                )
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
            return operations_to_maestro(tapes[0].operations, num_wires)

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
            sorted(list(measure_at_steps)),
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

