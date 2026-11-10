"""Run tapes through Divi on a remote backend such as the QoroService cloud.

With ``backend="qoro"`` (or a Divi ``CircuitRunner``) the device hands each
batch of tapes to Divi instead of the local Maestro engine:

* tapes whose results follow from expectation values (analytic tapes, and
  finite-shot tapes of Pauli-word expectation values, as on the local engine)
  become one expectation-value job. Expectation values and variances of
  Pauli sums are read from their Pauli terms (variances from ``<O^2>``);
  probabilities, mid-circuit measurement values and observables without a
  Pauli form are read from the Pauli-Z expectation values over their wires,
  after the observable's diagonalizing gates;
* every other finite-shot tape is sampled in one counts job: one circuit per
  measurement basis (and shot copy), as default.qubit groups its samples;
* adjoint derivatives become one ``submit_vjp`` job per register width, the
  Jacobian as one cotangent row per measurement.

Circuits reach Divi as PennyLane tapes (``divi.circuits.qscript_to_meta``)
and leave as OpenQASM 2. Wire ``i`` of a tape mapped to standard wires is
qubit ``q[i]``. Divi Pauli terms put qubit 0 first; Divi counts put it last.

Divi is imported lazily, so the local device does not need it.
"""

import re
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MutualInfoMP,
    ProbabilityMP,
    PurityMP,
    SampleMP,
    ShadowExpvalMP,
    StateMP,
    VarianceMP,
    VnEntropyMP,
)
from pennylane.ops import Snapshot
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

from pennylane_maestro import adjoint as _adjoint
from pennylane_maestro.converter import _to_float, observable_to_pauli_string

#: PennyLane gates handed to Divi as they are. Divi converts each to one
#: OpenQASM gate and differentiates every angle among them; the device
#: decomposes everything else first (state preparations and matrix-valued
#: gates such as ``QubitUnitary`` included, so their parameters reach Divi as
#: angles). ``Identity`` is kept on one wire only.
DIVI_GATES = frozenset(
    {
        "Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "SX",
        "Adjoint(SX)",
        "T",
        "Adjoint(T)",
        "RX",
        "RY",
        "RZ",
        "PhaseShift",
        "U2",
        "U3",
        "CNOT",
        "CZ",
        "CRX",
        "CRY",
        "CRZ",
        "SWAP",
        "Toffoli",
        "CSWAP",
    }
)

#: Most wires an analytic measurement read from its computational-basis
#: distribution may span (``qml.probs``, mid-circuit measurement values,
#: observables without a Pauli form): ``k`` wires take ``2**k - 1`` terms.
MAX_ANALYTIC_PROBABILITY_WIRES = 10

#: Most Pauli terms of ``O**2`` an analytic ``qml.var(O)`` may need.
MAX_VARIANCE_TERMS = 10_000

#: Largest imaginary part tolerated in an observable's Pauli coefficients.
_IMAGINARY_TOLERANCE = 1e-10

#: Significant digits of fixed angles in the OpenQASM sent to Divi.
_QASM_PRECISION = 17

_ID_LINE = re.compile(r"id q\[(\d+)\];")

# Measurements that need the quantum state itself, which no Divi backend
# returns.
_STATE_ONLY = (StateMP, VnEntropyMP, MutualInfoMP, PurityMP)


# ---------------------------------------------------------------------------
# Divi import and backend resolution
# ---------------------------------------------------------------------------

_IPYTHON_HOOKS = ("showtraceback", "_showtraceback", "showsyntaxerror")
_UNSET = object()


def import_divi():
    """Import Divi's backend and circuit modules, or explain how to get them.

    Importing Divi installs Rich's traceback handler for the whole process;
    when this is the first import, the previous handlers are put back.
    """
    if "divi" in sys.modules:
        import divi.backends
        import divi.circuits

        return divi
    excepthook = sys.excepthook
    shell = None
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        pass
    saved = {name: vars(shell).get(name, _UNSET) for name in _IPYTHON_HOOKS} if shell else {}
    try:
        import divi.backends
        import divi.circuits
    except ImportError as exc:
        raise ImportError(
            "Running circuits through Divi needs the qoro-divi package "
            "(Python 3.11+): pip install 'pennylane-maestro[cloud]'."
        ) from exc
    finally:
        sys.excepthook = excepthook
        for name, value in saved.items():
            if value is _UNSET:
                vars(shell).pop(name, None)
            else:
                setattr(shell, name, value)
    return divi


def maestro_options(simulator_config, explicit: Mapping[str, Any] | None = None) -> dict:
    """``divi.backends.MaestroConfig`` keyword arguments for the options of a
    ``maestro.SimulatorConfig`` that differ from Maestro's defaults, plus the
    ``explicit`` ones (set by the user even where they equal the defaults)."""
    divi = import_divi()
    config = divi.backends.MaestroConfig.from_simulator_config(simulator_config)
    return dict(config.model_extra or {}) | dict(explicit or {})


def resolve_backend(backend, simulator_config, explicit=None):
    """The Divi runner for the device's ``backend`` setting.

    ``"qoro"`` builds a ``QoroService`` (API key from ``QORO_API_KEY`` or a
    ``.env`` file; with Divi 0.16+ also the defaults saved by
    ``divi-config-helper``) and layers the device's Maestro options over its
    Maestro defaults. A ``CircuitRunner`` instance is used as it is. Maestro
    options that cannot apply (a runner passed in, a QPU target) warn.
    """
    divi = import_divi()
    options = maestro_options(simulator_config, explicit)
    if isinstance(backend, str):
        service = divi.backends.QoroService()
        if options:
            if getattr(service.job_config, "qpu_system", None) is not None:
                _warn_unused(options, f"the QPU target {service.job_config.qpu_system.name!r}")
            else:
                base = service.maestro_config or divi.backends.MaestroConfig()
                service.maestro_config = base.override(**options)
        return service
    if not isinstance(backend, divi.backends.CircuitRunner):
        raise TypeError(
            "backend must be 'local', 'qoro' or a divi.backends.CircuitRunner; "
            f"got {type(backend).__name__}."
        )
    if options:
        _warn_unused(options, f"a {type(backend).__name__} passed as backend; set them on its maestro_config instead")
    return backend


def _warn_unused(options, target):
    warnings.warn(
        f"The device's Maestro options {sorted(options)} do not apply to {target}.",
        UserWarning,
        stacklevel=5,
    )


def _named(value, name: str) -> bool:
    """Whether a Maestro enum setting is unset or ``name``."""
    return value is None or getattr(value, "name", str(value)) == name


# ---------------------------------------------------------------------------
# Tapes to Divi payloads
# ---------------------------------------------------------------------------


def _standardize(tape: QuantumScript) -> tuple[QuantumScript, int]:
    """``tape`` on wires ``0..n-1`` and its register width ``n``."""
    tape = tape.map_to_standard_wires()
    return tape, max(len(tape.wires), 1)


def _to_qasm(operations, num_wires: int, slots: Sequence[int] = (), measure: bool = False):
    """OpenQASM 2 for ``operations`` on wires ``0..num_wires-1`` (wire ``i``
    is ``q[i]``) and its free parameters.

    The operation parameters at indices ``slots`` become the free parameters
    ``p0, p1, ...`` in that order; all others are bound.
    """
    import sympy

    divi = import_divi()
    # Identity on every wire, in order, fixes the register to wires 0..n-1:
    # Divi leaves integer wires 0..n-1 as they are and remaps any others.
    ops = [qml.Identity(w) for w in range(num_wires)] + list(operations)
    script = QuantumScript(ops, [], trainable_params=[])
    if slots:
        symbols = [sympy.Symbol(f"p{k}") for k in range(len(slots))]
        script = script.bind_new_parameters(symbols, list(slots))
    meta = divi.circuits.qscript_to_meta(script, precision=_QASM_PRECISION)
    ((_, dag),) = meta.circuit_bodies
    body = divi.circuits.dag_to_qasm_body(dag, precision=_QASM_PRECISION)
    # Drop the padding identities: the first one on each wire. Identities in
    # the circuit itself stay (noisy runners treat them as idle steps).
    lines, padded = [], set()
    for line in (raw.strip() for raw in body.splitlines()):
        match = _ID_LINE.fullmatch(line)
        if match and int(match.group(1)) not in padded:
            padded.add(int(match.group(1)))
            continue
        if line:
            lines.append(line)
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_wires}];",
        f"creg c[{num_wires}];",
    ]
    if measure:
        lines += [f"measure q[{i}] -> c[{i}];" for i in range(num_wires)]
    return "\n".join(header + lines) + "\n", tuple(meta.parameters)


def _bound_payload(label: str, operations, num_wires: int, measure: bool):
    divi = import_divi()
    circuit, _ = _to_qasm(operations, num_wires, measure=measure)
    return divi.circuits.CircuitPayload(circuit, (), ((label, ()),))


def _group_observables(observables: Sequence[str]):
    """``ham_ops`` and ``circuit_ham_map`` giving each circuit its terms: each
    run of consecutive circuits with the same terms shares one group."""
    groups, ranges = [], []
    for index, observable in enumerate(observables):
        if groups and groups[-1] == observable:
            ranges[-1][1] = index + 1
        else:
            groups.append(observable)
            ranges.append([index, index + 1])
    return "|".join(groups), ranges


def _shot_groups(shots: Sequence[int]) -> list[list[int]]:
    """Per-circuit shots as Divi ``[start, end, shots]`` ranges."""
    ranges: list[list[int]] = []
    for index, count in enumerate(shots):
        if ranges and ranges[-1][2] == count:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1, int(count)])
    return ranges


def _counts_to_samples(counts: Mapping[str, int], num_wires: int) -> np.ndarray:
    """``(shots, num_wires)`` samples from Divi counts, whose keys put qubit 0
    last; column ``i`` of the result is wire ``i``."""
    keys = list(counts)
    if not keys:
        return np.zeros((0, num_wires), dtype=np.int64)
    rows = np.array(
        [[int(b) for b in key[::-1].ljust(num_wires, "0")[:num_wires]] for key in keys],
        dtype=np.int64,
    )
    return np.repeat(rows, [int(counts[key]) for key in keys], axis=0)


# ---------------------------------------------------------------------------
# Pauli algebra
# ---------------------------------------------------------------------------


def _pauli_sentence(obs):
    """``obs`` as a ``PauliSentence``, or ``None`` if it has no Pauli form."""
    try:
        return qml.pauli.pauli_sentence(obs)
    except (ValueError, TypeError, AttributeError, NotImplementedError):
        return None


def _dense_terms(sentence, num_wires: int, obs=None) -> tuple[float, dict[str, float]]:
    """The identity coefficient of ``sentence`` and its other terms as dense
    Pauli strings (character ``i`` acts on wire ``i``) -> coefficient."""
    constant, terms = 0.0, {}
    for word, coeff in sentence.items():
        coeff = complex(qml.math.toarray(coeff))
        if abs(coeff.imag) > _IMAGINARY_TOLERANCE:
            raise DeviceError(
                f"Observable {obs if obs is not None else sentence} is not Hermitian."
            )
        if not word:
            constant += coeff.real
            continue
        chars = ["I"] * num_wires
        for wire, pauli in word.items():
            chars[int(wire)] = pauli
        key = "".join(chars)
        terms[key] = terms.get(key, 0.0) + coeff.real
    return constant, terms


def _observable_terms(obs, num_wires: int) -> tuple[float, dict[str, float]]:
    sentence = _pauli_sentence(obs)
    if sentence is None:
        raise DeviceError(f"Observable {obs} is not a sum of Pauli words.")
    return _dense_terms(sentence, num_wires, obs)


def _hadamard(k: int) -> np.ndarray:
    """``H[b, s] = (-1) ** popcount(b & s)`` over ``k`` bits."""
    h = np.ones((1, 1))
    for _ in range(k):
        h = np.kron(h, np.array([[1.0, 1.0], [1.0, -1.0]]))
    return h


def _z_strings(wires: Sequence[int], num_wires: int) -> list[str]:
    """Pauli-Z products over every non-empty subset of ``wires``, indexed by
    a bitmask whose most significant bit is ``wires[0]``."""
    k = len(wires)
    out = []
    for mask in range(1, 2**k):
        chars = ["I"] * num_wires
        for j, wire in enumerate(wires):
            if mask >> (k - 1 - j) & 1:
                chars[wire] = "Z"
        out.append("".join(chars))
    return out


def _rotation(basis: Mapping[int, str]) -> list:
    """Gates taking each wire's Pauli eigenbasis in ``basis`` to the
    computational basis (PennyLane's diagonalizing gates)."""
    gates = []
    for wire, pauli in sorted(basis.items()):
        if pauli == "X":
            gates += qml.X(wire).diagonalizing_gates()
        elif pauli == "Y":
            gates += qml.Y(wire).diagonalizing_gates()
    return gates


# ---------------------------------------------------------------------------
# Analytic readouts
# ---------------------------------------------------------------------------


@dataclass
class _Readout:
    """One measurement read from the expectation values of ``terms`` on the
    circuit followed by ``rotation``."""

    rotation: list
    terms: frozenset
    finish: Callable[[Mapping[str, float]], Any]


def _linear(constant, terms):
    return lambda ev: constant + sum(c * ev[t] for t, c in terms.items())


def _distribution_readout(mp, wires: Sequence[int], num_wires: int, rotation) -> _Readout:
    """``mp`` read from the exact distribution over ``wires`` after
    ``rotation``: the Pauli-Z expectation values give it through a
    Walsh-Hadamard transform, and PennyLane finishes the measurement from it
    as from counts."""
    wires = sorted(wires)
    k = len(wires)
    if k > MAX_ANALYTIC_PROBABILITY_WIRES:
        raise DeviceError(
            f"{mp} reads {k} wires; a Divi backend estimates analytic "
            f"distributions on at most {MAX_ANALYTIC_PROBABILITY_WIRES}. Use finite shots."
        )
    strings = _z_strings(wires, num_wires)
    # A measurement over all wires (no wires given) is finished over the
    # wires the distribution was read on.
    target = mp if mp.wires else qml.probs(wires=Wires(wires))

    def finish(ev):
        values = np.concatenate([[1.0], [ev[s] for s in strings]])
        probs = np.clip(_hadamard(k) @ values / 2**k, 0.0, None)
        counts = {format(b, f"0{k}b"): probs[b] for b in range(2**k)}
        return target.process_counts(counts, wire_order=Wires(wires))

    return _Readout(rotation, frozenset(strings), finish)


def _estimate_readout(mp, num_wires: int) -> _Readout:
    """The readout of one measurement of a tape on standard wires."""
    if mp.obs is None:
        # Computational-basis measurements: qml.probs(wires) and measurements
        # of mid-circuit measurement values.
        if not isinstance(mp, (ExpectationMP, VarianceMP, ProbabilityMP)):
            raise DeviceError(f"A Divi backend cannot return {mp} without shots.")
        return _distribution_readout(mp, [int(w) for w in mp.wires or range(num_wires)], num_wires, [])
    sentence = _pauli_sentence(mp.obs)
    if sentence is not None and isinstance(mp, ExpectationMP):
        constant, terms = _dense_terms(sentence, num_wires, mp.obs)
        value = _linear(constant, terms)
        return _Readout([], frozenset(terms), lambda ev: np.float64(value(ev)))
    if sentence is not None and isinstance(mp, VarianceMP):
        c1, t1 = _dense_terms(sentence, num_wires, mp.obs)
        c2, t2 = _dense_terms(sentence @ sentence, num_wires, mp.obs)
        if len(t2) > MAX_VARIANCE_TERMS:
            raise DeviceError(
                f"qml.var({mp.obs}) needs {len(t2)} Pauli terms; a Divi backend "
                f"estimates analytic variances of at most {MAX_VARIANCE_TERMS}. "
                "Use finite shots."
            )
        first, second = _linear(c1, t1), _linear(c2, t2)
        return _Readout(
            [], frozenset(t1) | frozenset(t2), lambda ev: np.float64(second(ev) - first(ev) ** 2)
        )
    if isinstance(mp, (ExpectationMP, VarianceMP, ProbabilityMP)):
        # No Pauli form (Hermitian, Projector, ...) or probabilities in an
        # observable's eigenbasis: rotate it diagonal and read the
        # distribution over its wires.
        return _distribution_readout(
            mp, [int(w) for w in mp.obs.wires], num_wires, list(mp.diagonalizing_gates())
        )
    raise DeviceError(f"A Divi backend cannot return {mp} without shots.")


def _pauli_words_only(tape: QuantumScript, num_wires: int) -> bool:
    """Whether every measurement is the expectation value of one Pauli word,
    which the device computes exactly even with shots (as the local engine)."""
    return bool(tape.measurements) and all(
        isinstance(mp, ExpectationMP)
        and mp.obs is not None
        and mp.mv is None
        and observable_to_pauli_string(mp.obs, num_wires) is not None
        for mp in tape.measurements
    )


def _bundle(values: Sequence):
    return values[0] if len(values) == 1 else tuple(values)


def _check_measurements(tape: QuantumScript):
    for mp in tape.measurements:
        if isinstance(mp, _STATE_ONLY):
            raise DeviceError(
                f"{mp} needs the quantum state, which a Divi backend does not return."
            )
        if isinstance(mp, (ClassicalShadowMP, ShadowExpvalMP)):
            raise DeviceError(f"A Divi backend does not support {type(mp).__name__}.")


# ---------------------------------------------------------------------------
# Per-tape tasks
# ---------------------------------------------------------------------------


@dataclass
class _EstimateTask:
    """A tape answered from expectation values: one circuit for its Pauli
    readouts and one per rotated readout."""

    label: str
    tape: QuantumScript
    num_wires: int
    readouts: list
    circuits: list = field(init=False)  # (label, rotation, sorted terms)
    index: list = field(init=False)  # circuit of each readout

    def __post_init__(self):
        base = [r for r in self.readouts if not r.rotation]
        self.circuits = [(f"{self.label}.0", [], sorted(set().union(*(r.terms for r in base))))]
        self.index = []
        for readout in self.readouts:
            if readout.rotation:
                self.circuits.append(
                    (f"{self.label}.{len(self.circuits)}", readout.rotation, sorted(readout.terms))
                )
                self.index.append(len(self.circuits) - 1)
            else:
                self.index.append(0)

    def finish(self, ev: Mapping[str, Mapping[str, float]]):
        result = _bundle(
            [r.finish(ev.get(self.circuits[c][0], {})) for r, c in zip(self.readouts, self.index)]
        )
        if self.tape.shots.has_partitioned_shots:
            return tuple(result for _ in range(self.tape.shots.num_copies))
        return result


@dataclass
class _SampleReader:
    """How one measurement is computed from the samples of its circuit(s)."""

    circuits: list  # circuit indices
    finish: Callable[[Sequence[np.ndarray]], Any]


class _SampleTask:
    """A finite-shot tape answered from sampled circuits: one per
    measurement basis (Pauli words share a basis where their wires agree)
    and one per observable that needs its own diagonalizing gates, each
    sampled once per shot copy."""

    def __init__(self, label: str, tape: QuantumScript):
        self.label = label
        self.shots = tape.shots
        self.tape, self.num_wires = _standardize(tape)
        self.bases: list[dict] = []  # basis of each shared circuit
        self.special: list[list] = []  # rotation of each other circuit
        self.readers = [self._reader(mp) for mp in self.tape.measurements]

    def _basis_index(self, basis: Mapping[int, str]) -> int:
        for i, existing in enumerate(self.bases):
            if all(existing.get(w, p) == p for w, p in basis.items()):
                existing.update(basis)
                return i
        self.bases.append(dict(basis))
        return len(self.bases) - 1

    def _special_index(self, rotation) -> int:
        self.special.append(rotation)
        return -len(self.special)  # negative: an entry of self.special

    def _reader(self, mp) -> _SampleReader:
        wire_order = Wires(range(self.num_wires))
        if mp.obs is None:
            wires = [int(w) for w in mp.wires or range(self.num_wires)]
            c = self._basis_index({w: "Z" for w in wires})
            return _SampleReader([c], lambda s: mp.process_samples(s[0], wire_order))
        sentence = _pauli_sentence(mp.obs)
        if sentence is not None and isinstance(mp, ExpectationMP):
            constant, terms = _dense_terms(sentence, self.num_wires, mp.obs)
            words = list(terms)
            circuits = [
                self._basis_index({i: p for i, p in enumerate(word) if p != "I"}) for word in words
            ]
            masks = [np.array([p != "I" for p in word]) for word in words]

            def finish(samples):
                total = constant
                for word, mask, s in zip(words, masks, samples):
                    parity = s[:, mask].sum(axis=1) % 2
                    total += terms[word] * np.mean(1 - 2 * parity)
                return np.float64(total)

            return _SampleReader(circuits, finish)
        if sentence is not None and len(sentence) == 1:
            # One (scaled) Pauli word, or the identity: shares a basis.
            (word,) = sentence
            c = self._basis_index({int(w): p for w, p in word.items()})
            return _SampleReader([c], lambda s: mp.process_samples(s[0], wire_order))
        c = self._special_index(list(mp.diagonalizing_gates()))
        return _SampleReader([c], lambda s: mp.process_samples(s[0], wire_order))

    def _rotation(self, c: int):
        return _rotation(self.bases[c]) if c >= 0 else self.special[-c - 1]

    def _all_circuits(self):
        if not self.bases and not self.special:
            self.bases.append({})
        return list(range(len(self.bases))) + [-i - 1 for i in range(len(self.special))]

    def circuits(self):
        """``(label, operations, width, shots)`` for every sampled circuit."""
        for c in self._all_circuits():
            ops = list(self.tape.operations) + self._rotation(c)
            for copy_index, shots in enumerate(self.shots):
                yield f"{self.label}.{c}.{copy_index}", ops, self.num_wires, shots

    def finish(self, counts: Mapping[str, Mapping[str, int]]):
        copies = []
        for copy_index, _ in enumerate(self.shots):

            def samples(c):
                return _counts_to_samples(
                    counts[f"{self.label}.{c}.{copy_index}"], self.num_wires
                )

            copies.append(
                _bundle([r.finish([samples(c) for c in r.circuits]) for r in self.readers])
            )
        return tuple(copies) if self.shots.has_partitioned_shots else copies[0]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class DiviExecutor:
    """Executes preprocessed tapes on a Divi ``CircuitRunner``."""

    def __init__(self, runner):
        self.runner = runner

    @property
    def supports_expval(self) -> bool:
        """Whether the runner estimates expectation values (analytic tapes)."""
        return bool(getattr(self.runner, "supports_expval", False))

    @property
    def supports_vjp(self) -> bool:
        """Whether the runner computes adjoint VJPs.

        Divi compiles the adjoint program locally, which needs a qoro-maestro
        build with ``AdjointProgram``; Maestro differentiates on the CPU
        statevector (``QCSim``, ``Statevector``) of a simulator target only.
        """
        if not (_adjoint.HAS_ENGINE_ADJOINT and getattr(self.runner, "supports_vjp", False)):
            return False
        if getattr(getattr(self.runner, "job_config", None), "qpu_system", None) is not None:
            return False
        config = getattr(self.runner, "maestro_config", None)
        return _named(getattr(config, "simulation_type", None), "Statevector") and _named(
            getattr(config, "simulator_type", None), "QCSim"
        )

    # -- running jobs ----------------------------------------------------

    def _run(self, payloads, **kwargs) -> dict:
        """Run ``payloads`` as one job and return the results by label."""
        result = self.runner.submit_circuits(payloads, **kwargs)
        if result.is_async():
            self.runner.poll_job_status(result, loop_until_complete=True, verbose=False)
            result = self.runner.get_job_results(result)
        return self._by_label(result)

    @staticmethod
    def _by_label(result) -> dict:
        if result.results is None:
            raise DeviceError("The Divi backend returned no results.")
        return {entry["label"]: entry for entry in result.results}

    @staticmethod
    def _entry(by_label, label):
        try:
            return by_label[label]
        except KeyError:
            raise DeviceError(f"The Divi backend returned no result for '{label}'.") from None

    # -- forward execution -------------------------------------------------

    def _task(self, label: str, tape: QuantumScript):
        _check_measurements(tape)
        if not tape.measurements:
            return None
        std, width = _standardize(tape)
        if self.supports_expval and (not tape.shots or _pauli_words_only(std, width)):
            readouts = [_estimate_readout(mp, width) for mp in std.measurements]
            return _EstimateTask(label, std, width, readouts)
        if not tape.shots:
            raise DeviceError(
                f"{type(self.runner).__name__} does not estimate expectation "
                "values, so this tape needs finite shots."
            )
        return _SampleTask(label, tape)

    def execute(self, tapes: Sequence[QuantumScript], debugger=None) -> tuple:
        """Results of ``tapes`` in PennyLane's format.

        Snapshots run as extra tapes in the same jobs when ``debugger`` is
        active, and are dropped otherwise.
        """
        tapes, snapshots, mains = _split_snapshots(tapes, debugger)
        tasks = [self._task(f"t{i}", tape) for i, tape in enumerate(tapes)]

        circuits = [
            (lab, task.tape.operations + rotation, task.num_wires, terms)
            for task in tasks
            if isinstance(task, _EstimateTask)
            for lab, rotation, terms in task.circuits
            if terms
        ]
        expvals = {}
        if circuits:
            payloads = [_bound_payload(lab, ops, w, False) for lab, ops, w, _ in circuits]
            ham_ops, ranges = _group_observables([";".join(terms) for *_, terms in circuits])
            kwargs = {"ham_ops": ham_ops}
            if len(ranges) > 1:
                kwargs["circuit_ham_map"] = ranges
            by_label = self._run(payloads, **kwargs)
            expvals = {lab: self._entry(by_label, lab)["results"] for lab, *_ in circuits}

        sampled = [c for task in tasks if isinstance(task, _SampleTask) for c in task.circuits()]
        counts = {}
        if sampled:
            payloads = [_bound_payload(lab, ops, w, True) for lab, ops, w, _ in sampled]
            by_label = self._run(payloads, shot_groups=_shot_groups([c[3] for c in sampled]))
            counts = {lab: self._entry(by_label, lab)["results"] for lab, *_ in sampled}

        results = [
            task.finish(expvals if isinstance(task, _EstimateTask) else counts)
            if task is not None
            else ()
            for task in tasks
        ]
        for tag, index in snapshots:
            if tag is None:
                tag = len(debugger.snapshots)
            _store_snapshot(debugger, tag, results[index])
        return tuple(results[i] for i in mains)

    # -- derivatives -------------------------------------------------------

    def gradients(self, tapes: Sequence[QuantumScript], cotangents: Sequence[np.ndarray]):
        """Expectation values and cotangent-weighted gradients of ``tapes``.

        ``cotangents[i]`` has shape ``(rows, measurements)``: each row is one
        VJP of tape ``i``. Returns, per tape, the values ``(measurements,)``
        and the VJPs ``(rows, trainable parameters)``. Tapes of one register
        width share one ``submit_vjp`` job; tapes without trainable
        parameters only need their values, from one expectation-value job.
        """
        divi = import_divi()
        prepared, constant = [], []
        for i, (tape, rows) in enumerate(zip(tapes, cotangents)):
            rows = np.atleast_2d(rows)
            if not tape.trainable_params:
                constant.append(i)
                prepared.append(None)
                continue
            std, width = _standardize(tape)
            readouts = [_observable_terms(mp.obs, width) for mp in std.measurements]
            slots = list(std.trainable_params)
            circuit, parameters = _to_qasm(std.operations, width, slots)
            all_values = std.get_parameters(trainable_only=False)
            by_name = {f"p{k}": _to_float(all_values[s]) for k, s in enumerate(slots)}
            row_values = tuple(by_name[p.name] for p in parameters)
            labels = [f"g{i}.{r}" for r in range(len(rows))]
            payload = divi.circuits.CircuitPayload(
                circuit, parameters, tuple((label, row_values) for label in labels)
            )
            weights = []
            for row in rows:
                w: dict[str, float] = {}
                for dy, (_, terms) in zip(row, readouts):
                    for term, coeff in terms.items():
                        w[term] = w.get(term, 0.0) + float(dy) * coeff
                weights.append(w)
            prepared.append((width, readouts, len(slots), payload, labels, weights))

        entries: dict = {}
        submitted = set()
        jobs = [p for p in prepared if p is not None]
        for width in sorted({p[0] for p in jobs}):
            group = [p for p in jobs if p[0] == width]
            union = sorted({t for p in group for _, terms in p[1] for t in terms})
            if not union:
                continue
            submitted.add(width)
            cot = {
                label: np.array([w.get(t, 0.0) for t in union])
                for _, _, _, _, labels, weights in group
                for label, w in zip(labels, weights)
            }
            out = self.runner.submit_vjp(
                [p[3] for p in group], ham_ops=";".join(union), cotangents=cot
            )
            entries.update(self._by_label(out))

        forward = dict(zip(constant, self.execute([tapes[i] for i in constant])))
        values, vjps = [], []
        for i, item in enumerate(prepared):
            if item is None:
                rows = np.atleast_2d(cotangents[i])
                values.append(np.atleast_1d(np.asarray(forward[i], dtype=np.float64)))
                vjps.append(np.zeros((len(rows), 0)))
                continue
            width, readouts, num_params, payload, labels, _ = item
            row_vjp = np.zeros((len(labels), num_params))
            if width not in submitted:
                # Every observable of the width group was a constant.
                values.append(np.array([c for c, _ in readouts]))
                vjps.append(row_vjp)
                continue
            ev = self._entry(entries, labels[0])["results"]
            values.append(
                np.array([c + sum(k * ev[t] for t, k in terms.items()) for c, terms in readouts])
            )
            columns = {name: j for j, name in enumerate(payload.parameter_names)}
            for r, label in enumerate(labels):
                vjp = np.asarray(self._entry(entries, label)["metadata"]["vjp"], dtype=np.float64)
                for k in range(num_params):
                    j = columns.get(f"p{k}")
                    if j is not None:
                        row_vjp[r, k] = vjp[j]
            vjps.append(row_vjp)
        return values, vjps


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


@qml.transform
def fill_measurement_wires(tape: QuantumScript):
    """Give ``qml.sample()``, ``qml.counts()`` and ``qml.probs()`` without
    wires the tape's wires, as ``validate_device_wires`` does with device
    wires (so that deferring mid-circuit measurements leaves them alone)."""
    wireless = (SampleMP, CountsMP, ProbabilityMP)
    if not any(
        isinstance(mp, wireless) and mp.obs is None and mp.mv is None and not mp.wires
        for mp in tape.measurements
    ):
        return (tape,), lambda results: results[0]
    # "All wires" in the order map_to_standard_wires gives them, as
    # default.qubit reads them: first use, unless the labels already are
    # 0..n-1.
    wires = tape.wires
    if set(wires) == set(range(len(wires))):
        wires = Wires(range(len(wires)))
    measurements = []
    for mp in tape.measurements:
        if isinstance(mp, wireless) and mp.obs is None and mp.mv is None and not mp.wires:
            mp = copy(mp)
            mp._wires = wires  # pylint: disable=protected-access
        measurements.append(mp)
    return (tape.copy(measurements=measurements),), lambda results: results[0]


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _split_snapshots(tapes, debugger):
    """Tapes without ``Snapshot`` ops plus, when ``debugger`` is active, one
    tape per snapshot measuring it; returns ``(tapes, [(tag, index)], mains)``.
    """
    active = debugger is not None and getattr(debugger, "active", False)
    out, snapshots, mains = [], [], []
    for tape in tapes:
        ops = []
        for op in tape.operations:
            if not isinstance(op, Snapshot):
                ops.append(op)
                continue
            if not active:
                continue
            measurement = op.hyperparameters["measurement"]
            if isinstance(measurement, _STATE_ONLY):
                raise DeviceError(
                    "A Divi backend cannot record the state in a snapshot; give "
                    "qml.Snapshot a measurement, e.g. "
                    "qml.Snapshot(measurement=qml.expval(qml.Z(0)))."
                )
            # "workflow" (the default) takes the tape's shots; anything else,
            # None included, is the snapshot's own.
            shots = op.hyperparameters.get("shots", "workflow")
            if isinstance(shots, str):
                shots = tape.shots
            out.append(QuantumScript(list(ops), [measurement], shots=shots))
            snapshots.append((op.tag, len(out) - 1))
        mains.append(len(out))
        out.append(tape.copy(operations=ops))
    return out, snapshots, mains


def _store_snapshot(debugger, tag, result):
    """Store a snapshot result as default.qubit does."""
    if tag not in debugger.snapshots:
        debugger.snapshots[tag] = result
    elif isinstance(debugger.snapshots[tag], list):
        debugger.snapshots[tag].append(result)
    else:
        debugger.snapshots[tag] = [debugger.snapshots[tag], result]
