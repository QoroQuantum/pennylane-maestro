"""Analytic derivatives and batched expectation values for statevector tapes.

A tape is split into a *template* (gate structure, cached across calls) and
its parameter values. Every parametric Maestro gate angle is one tape
parameter times ``±1``, so a whole batch of parameter values becomes one
``(batch, num_gate_params)`` matrix.

Two derivative routes are provided:

* the engine's ``maestro.AdjointProgram`` (adjoint sweep in C++, batched,
  available in qoro-maestro builds that include it), and
* :func:`inner_product_jacobian`, which needs only ``maestro.inner_product``
  and uses ``dE/dθ_k = Σ c·a Im <0|(P U)† U_{>k} G U_{<=k}|0>`` with the
  generator ``G`` written as Pauli terms.
"""

from collections import OrderedDict

import numpy as np
import pennylane as qml
from pennylane.measurements import ExpectationMP, MidMeasureMP, ProbabilityMP
from pennylane.ops import Snapshot
from pennylane.ops.op_math import Adjoint, Conditional

import maestro
from maestro.circuits import QuantumCircuit

from pennylane_maestro.converter import ADJOINT_MAP, GATE_MAP, extract_pauli_terms

HAS_ENGINE_ADJOINT = hasattr(maestro, "AdjointProgram")
# Engine builds with probability readouts for AdjointProgram.
HAS_ENGINE_PROBABILITIES = HAS_ENGINE_ADJOINT and hasattr(maestro, "Probabilities")

# Most qml.probs wires taken on the adjoint path: a Jacobian costs one
# backward sweep per outcome (2**k), a VJP one sweep in total.
MAX_PROBABILITY_WIRES_JACOBIAN = 10
MAX_PROBABILITY_WIRES_VJP = 24

# One-angle gates: PennyLane name -> Maestro builder.
_ONE_ANGLE = {
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "p",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "ControlledPhaseShift": "cp",
}

# Generators of the one-angle builders, as Pauli terms over the gate's wires
# (index into the wire list), for U(t) = exp(-i t G / 2) up to a global phase.
# Identity terms are dropped: they contribute Im <psi|O|psi> = 0.
_GENERATORS = {
    "rx": [(1.0, {0: "X"})],
    "ry": [(1.0, {0: "Y"})],
    "rz": [(1.0, {0: "Z"})],
    "p": [(1.0, {0: "Z"})],
    "crx": [(0.5, {1: "X"}), (-0.5, {0: "Z", 1: "X"})],
    "cry": [(0.5, {1: "Y"}), (-0.5, {0: "Z", 1: "Y"})],
    "crz": [(0.5, {1: "Z"}), (-0.5, {0: "Z", 1: "Z"})],
    "cp": [(0.5, {0: "Z"}), (0.5, {1: "Z"}), (-0.5, {0: "Z", 1: "Z"})],
}

_TEMPLATE_CACHE_SIZE = 128


class UnsupportedTape(ValueError):
    """The tape cannot be evaluated on the parametric statevector path."""


def _records(op, start):
    """Maestro gates for ``op`` as ``(builder, wires, slots)`` records.

    ``slots`` holds one ``(tape_param_index, sign)`` per gate angle; tape
    parameters of ``op`` start at index ``start``. Wires must be standard.
    """
    wires = [int(w) for w in op.wires]
    name = op.name
    if isinstance(op, Adjoint):
        base = op.base
        if base.name in _ONE_ANGLE:
            return [(_ONE_ANGLE[base.name], wires, [(start, -1.0)])]
        if base.name == "Rot":
            # Rot(a, b, c)^† = RZ(-a) RY(-b) RZ(-c): applied as rz(-c), ry(-b), rz(-a).
            return [("rz", wires, [(start + 2, -1.0)]), ("ry", wires, [(start + 1, -1.0)]),
                    ("rz", wires, [(start, -1.0)])]
        if base.name == "U3":
            # U3(t, f, l)^† = U3(-t, -l, -f)
            return [("u", wires, [(start, -1.0), (start + 2, -1.0), (start + 1, -1.0)])]
        if base.name in ADJOINT_MAP:
            return [(ADJOINT_MAP[base.name], wires, [])]
        if base.name in GATE_MAP and not GATE_MAP[base.name][1]:
            return [(GATE_MAP[base.name][0], wires, [])]  # self-inverse
        raise UnsupportedTape(f"Unsupported operation {name}")
    if name in _ONE_ANGLE:
        return [(_ONE_ANGLE[name], wires, [(start, 1.0)])]
    if name == "Rot":
        return [("rz", wires, [(start, 1.0)]), ("ry", wires, [(start + 1, 1.0)]),
                ("rz", wires, [(start + 2, 1.0)])]
    if name == "U3":
        return [("u", wires, [(start, 1.0), (start + 1, 1.0), (start + 2, 1.0)])]
    if name == "Delay":
        return []
    if name in GATE_MAP and not GATE_MAP[name][1]:
        return [(GATE_MAP[name][0], wires, [])]
    raise UnsupportedTape(f"Unsupported operation {name}")


class CircuitTemplate:
    """The gate structure of a tape, independent of its parameter values."""

    def __init__(self, operations, num_wires):
        self.num_wires = num_wires
        self.records = []
        start = 0
        for op in operations:
            self.records.extend(_records(op, start))
            start += len(op.data)
        self.num_op_params = start
        slots = [s for _, _, rec_slots in self.records for s in rec_slots]
        self.slot_index = np.array([i for i, _ in slots], dtype=np.intp)
        self.slot_sign = np.array([s for _, s in slots], dtype=np.float64)
        # d(gate angles)/d(tape params): jacobians are mapped with J @ selector.
        self.selector = np.zeros((len(slots), start))
        self.selector[np.arange(len(slots)), self.slot_index] = self.slot_sign
        self._program = None

    @property
    def program(self):
        """The compiled engine program (gate angles are passed per call)."""
        if self._program is None:
            self._program = maestro.AdjointProgram(self.circuit(np.zeros(len(self.slot_index))))
        return self._program

    def circuit(self, angles):
        """A Maestro circuit with the given gate angles."""
        qc = QuantumCircuit()
        for q in range(self.num_wires):
            qc.z(q)
            qc.z(q)
        k = 0
        for builder, wires, slots in self.records:
            n = len(slots)
            getattr(qc, builder)(*wires, *(float(a) for a in angles[k:k + n]))
            k += n
        return qc

    def angles(self, tape):
        """Gate angles, shape ``(batch, num_gate_params)``, and the batch size."""
        params = tape.get_parameters(trainable_only=False)[: self.num_op_params]
        batch = tape.batch_size
        values = np.empty((batch or 1, self.num_op_params))
        for i, p in enumerate(params):
            values[:, i] = qml.math.to_numpy(p)
        return values[:, self.slot_index] * self.slot_sign, batch


def _structure_key(tape):
    return (len(tape.wires), tuple((op.name, tuple(op.wires)) for op in tape.operations))


class TemplateCache:
    """Least-recently-used cache of templates keyed by gate structure."""

    def __init__(self, size=_TEMPLATE_CACHE_SIZE):
        self._size = size
        self._items = OrderedDict()

    def get(self, tape):
        """Template for a tape whose wires are already standard."""
        key = _structure_key(tape)
        template = self._items.get(key)
        if template is None:
            template = CircuitTemplate(tape.operations, max(len(tape.wires), 1))
            self._items[key] = template
            if len(self._items) > self._size:
                self._items.popitem(last=False)
        else:
            self._items.move_to_end(key)
        return template


def readouts(tape, num_wires):
    """Engine readouts for the measurements, and how many values each gives.

    Expectation values become ``[(coeff, pauli_string), ...]``; ``qml.probs``
    becomes ``maestro.Probabilities`` with the wires reversed, so that the
    engine's outcome index (first qubit least significant) equals
    PennyLane's (first wire most significant).
    """
    out, widths = [], []
    for mp in tape.measurements:
        if isinstance(mp, ProbabilityMP) and mp.obs is None and mp.mv is None:
            wires = [int(w) for w in (mp.wires or tape.wires)]
            out.append(maestro.Probabilities(wires[::-1]))
            widths.append(2 ** len(wires))
            continue
        if not isinstance(mp, ExpectationMP) or mp.obs is None:
            raise UnsupportedTape("Only expectation values and probabilities are supported.")
        terms = extract_pauli_terms(mp.obs, num_wires)
        if terms is None:
            raise UnsupportedTape(f"Observable {mp.obs} is not a sum of Pauli strings.")
        out.append(terms)
        widths.append(1)
    return out, widths


def is_supported(tape, trainable=None, max_probability_wires=0):
    """Whether ``tape`` fits the parametric path (structure only, any wires).

    ``trainable`` lists the trainable parameter indices; by default they are
    read from the parameter types (autograd, torch, ...). Trainable
    observable parameters are not supported. ``qml.probs`` is supported on
    up to ``max_probability_wires`` wires.
    """
    if tape.shots:
        return False
    for op in tape.operations:
        if isinstance(op, (MidMeasureMP, Conditional, Snapshot)):
            return False
    for mp in tape.measurements:
        if isinstance(mp, ProbabilityMP) and mp.obs is None and mp.mv is None:
            if len(mp.wires or tape.wires) > max_probability_wires:
                return False
            continue
        if not isinstance(mp, ExpectationMP) or mp.obs is None:
            return False
        try:
            sentence = qml.pauli.pauli_sentence(mp.obs)
        except (ValueError, TypeError, AttributeError):
            return False
        if sentence is None or len(sentence) == 0:
            return False
    if trainable is None:
        trainable = qml.math.get_trainable_indices(tape.get_parameters(trainable_only=False))
    num_op_params = sum(len(op.data) for op in tape.operations)
    return all(i < num_op_params for i in trainable)


def batched_params(tape):
    """Indices of the tape parameters that carry the broadcast dimension."""
    if tape.batch_size is None:
        return set()
    out = set()
    index = 0
    for op in tape.operations:
        if op.batch_size is not None:
            for j, p in enumerate(op.data):
                if qml.math.ndim(p) > op.ndim_params[j]:
                    out.add(index + j)
        index += len(op.data)
    return out


def batched_trainable(tape):
    """Whether a trainable parameter carries the broadcast dimension."""
    batched = batched_params(tape)
    if not batched:
        return False
    trainable = qml.math.get_trainable_indices(tape.get_parameters(trainable_only=False))
    return not batched.isdisjoint(trainable)


def reduce_vjp(row_vjp, tape):
    """VJP per trainable parameter from the VJP of each batch row.

    ``row_vjp`` has shape ``(rows, num_trainable)``. A parameter that carries
    the broadcast dimension gets one entry per row; the others are summed.
    """
    batched = batched_params(tape)
    return tuple(
        row_vjp[:, k].copy() if p in batched else np.float64(row_vjp[:, k].sum())
        for k, p in enumerate(tape.trainable_params)
    )


# ---------------------------------------------------------------------------
# Result formatting (PennyLane's nesting conventions)
# ---------------------------------------------------------------------------


def _slices(widths):
    starts = np.cumsum([0] + list(widths))
    return [slice(a, b) for a, b in zip(starts[:-1], starts[1:])]


def _leaf(block, batch, width):
    """``block`` has shape ``(rows, width)``: one measurement's values."""
    if width == 1:
        return block[:, 0] if batch else np.asarray(block[0, 0])
    return block if batch else block[0]


def format_values(values, batch, widths):
    """``values`` has shape ``(rows, num_values)``."""
    cols = []
    for sl, w in zip(_slices(widths), widths):
        leaf = _leaf(values[:, sl], batch, w)
        cols.append(np.float64(leaf) if (w == 1 and not batch) else leaf)
    return cols[0] if len(cols) == 1 else tuple(cols)


def format_jacobian(jac, batch, widths):
    """``jac`` has shape ``(rows, num_values, num_trainable)``."""
    num_params = jac.shape[2]
    num_meas = len(widths)
    if num_params == 0:
        return np.zeros((1, 0)) if num_meas == 1 else tuple(np.zeros((1, 0)) for _ in range(num_meas))
    per_meas = []
    for sl, w in zip(_slices(widths), widths):
        leaves = [_leaf(jac[:, sl, k], batch, w) for k in range(num_params)]
        per_meas.append(leaves[0] if num_params == 1 else tuple(leaves))
    return per_meas[0] if num_meas == 1 else tuple(per_meas)


def cotangent_matrix(dy, widths, batch):
    """Cotangents as ``(rows, num_values)``."""
    per_meas = [dy] if len(widths) == 1 else list(dy)
    rows = batch or 1
    return np.concatenate(
        [np.asarray(c, dtype=np.float64).reshape(rows, w) for c, w in zip(per_meas, widths)], axis=1
    )


# ---------------------------------------------------------------------------
# Route A: inner products (works with any qoro-maestro that has inner_product)
# ---------------------------------------------------------------------------


def _split_u(records, angles):
    """Replace ``u`` records by rz, ry, rz (same global phase on both sides)."""
    flat = []
    k = 0
    for builder, wires, slots in records:
        n = len(slots)
        a = list(angles[k:k + n])
        if builder == "u":
            t, f, l = a
            flat += [("rz", wires, [l], k + 2), ("ry", wires, [t], k), ("rz", wires, [f], k + 1)]
        else:
            flat.append((builder, wires, a, k if n else None))
        k += n
    return flat


def _build(num_wires, flat, insert_at=None, insert=None, tail=None):
    qc = QuantumCircuit()
    for q in range(num_wires):
        qc.z(q)
        qc.z(q)
    for i, (builder, wires, a, _) in enumerate(flat):
        getattr(qc, builder)(*wires, *a)
        if i == insert_at:
            for q, p in insert.items():
                getattr(qc, p.lower())(q)
    if tail:
        for q, p in tail.items():
            getattr(qc, p.lower())(q)
    return qc


def _pauli_dict(pauli_string):
    return {q: p for q, p in enumerate(pauli_string) if p != "I"}


def inner_product_jacobian(template, angles, observables, config):
    """Jacobian ``(num_measurements, num_gate_params)`` for one angle row."""
    flat = _split_u(template.records, angles)
    n = template.num_wires
    lhs = [
        [(c, _build(n, flat, tail=_pauli_dict(s))) for c, s in terms if _pauli_dict(s)]
        for terms in observables
    ]
    jac = np.zeros((len(observables), len(template.slot_index)))
    for i, (builder, wires, _, slot) in enumerate(flat):
        if slot is None:
            continue
        for a, gen in _GENERATORS[builder]:
            rhs = _build(n, flat, insert_at=i, insert={wires[w]: p for w, p in gen.items()})
            for m, terms in enumerate(lhs):
                for c, qc in terms:
                    jac[m, slot] += c * a * maestro.inner_product(qc, rhs, config).imag
    return jac
