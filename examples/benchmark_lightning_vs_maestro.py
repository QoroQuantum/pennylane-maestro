#!/usr/bin/env python3
"""
=============================================================
  pennylane-maestro  •  Performance Benchmark Suite
  default.qubit  vs  lightning.qubit  vs  maestro.qubit
=============================================================

Usage:
    python examples/benchmark_lightning_vs_maestro.py
"""

import time
import signal
import warnings
import numpy as np
import pennylane as qml

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────
WARMUP = 1
RUNS   = 3
SEED   = 42
TIMEOUT = 60


# ── Helpers ───────────────────────────────────────────────

class TimeoutError(Exception): pass
def _handler(s, f): raise TimeoutError()

def timed(fn, warmup=WARMUP, runs=RUNS):
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TIMEOUT)
    try:
        for _ in range(warmup): fn()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t0)
        return np.mean(ts), r
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def timed_grad(grad_fn, params, warmup=WARMUP, runs=RUNS):
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TIMEOUT)
    try:
        for _ in range(warmup): grad_fn(params)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); g = grad_fn(params); ts.append(time.perf_counter() - t0)
        return np.mean(ts), g
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def try_run(dev_name, n, circuit_fn, params=None, **kw):
    try:
        dev = qml.device(dev_name, wires=n, **kw)
        qnode = qml.QNode(circuit_fn, dev)
        return timed(qnode, params) if params is None else timed(lambda: qnode(params))
    except (TimeoutError, Exception):
        return None, None

DNF = "      DNF"
def fmt(t):
    if t is None: return DNF
    return f"{t:9.2f}s " if t >= 1.0 else f"{t*1000:9.2f}ms"
def sp(a, b):
    if a is None and b is not None: return "     ∞"
    if a is None or b is None: return "    —"
    return f"{a/b:6.1f}×"

def header(title):
    print(); print("=" * 85); print(f"  {title}"); print("=" * 85)

def table(rows, cols):
    w = [len(h) for h in cols]
    for r in rows:
        for i, c in enumerate(r): w[i] = max(w[i], len(str(c)))
    print("  " + " | ".join(h.ljust(w[i]) for i, h in enumerate(cols)))
    print("  " + "-+-".join("-" * x for x in w))
    for r in rows:
        print("  " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


# ══════════════════════════════════════════════════════════
# BENCHMARK 1: Statevector — Variational Circuit
# ══════════════════════════════════════════════════════════

def bench_statevector():
    header("Benchmark 1 — Statevector:  Deep Variational Circuit")
    print("  StronglyEntanglingLayers (3 layers)  •  ⟨Z₀⟩  •  analytic")
    print()

    rows = []
    for n in [14, 18, 20, 22, 24, 26]:
        np.random.seed(SEED)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=n)
        params = np.random.random(shape)
        def circ(w):
            qml.StronglyEntanglingLayers(w, wires=range(n))
            return qml.expval(qml.PauliZ(0))

        t_dq, r_dq = try_run("default.qubit", n, circ, params)
        t_lq, r_lq = try_run("lightning.qubit", n, circ, params)
        t_m,  r_m  = try_run("maestro.qubit", n, circ, params)
        ok = "✓" if r_dq is not None and r_m is not None and np.allclose(r_dq, r_m, atol=1e-5) else "—"
        rows.append([f"{n}", fmt(t_dq), fmt(t_lq), fmt(t_m), sp(t_dq, t_m), sp(t_lq, t_m), ok])

    table(rows, ["Qubits", "default.qubit", "lightning.qubit", "maestro.qubit",
                 "vs dq", "vs lq", "OK"])


# ══════════════════════════════════════════════════════════
# BENCHMARK 2: MPS — Scaling to 1000 Qubits
# ══════════════════════════════════════════════════════════

def bench_mps_scale():
    header("Benchmark 2 — MPS:  Scaling to 1000 Qubits")
    print("  RY + CNOT ladder (3 layers)  •  ⟨Z₀Z₁⟩  •  bond dim = 128")
    print("  Competitors: default.tensor (quimb), default.qubit (SV)")
    print()

    rows = []
    for n in [20, 50, 100, 200, 500, 1000]:
        t_dq = t_dt = t_m = None
        r_dq = r_dt = r_m = None

        # default.qubit — only feasible for small n
        if n <= 22:
            try:
                dev = qml.device("default.qubit", wires=n)
                @qml.qnode(dev)
                def c_dq():
                    for l in range(3):
                        for i in range(n): qml.RY(0.3*(i+1)*(l+1), wires=i)
                        for i in range(n-1): qml.CNOT(wires=[i, i+1])
                    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
                t_dq, r_dq = timed(c_dq)
            except (TimeoutError, Exception): pass

        # default.tensor (quimb MPS)
        try:
            dev = qml.device("default.tensor", wires=n, method="mps", max_bond_dim=128)
            @qml.qnode(dev)
            def c_dt():
                for l in range(3):
                    for i in range(n): qml.RY(0.3*(i+1)*(l+1), wires=i)
                    for i in range(n-1): qml.CNOT(wires=[i, i+1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            t_dt, r_dt = timed(c_dt)
        except (TimeoutError, Exception): pass

        # Maestro MPS
        try:
            dev = qml.device("maestro.qubit", wires=n,
                             simulation_type="MatrixProductState",
                             max_bond_dimension=128)
            @qml.qnode(dev)
            def c_m():
                for l in range(3):
                    for i in range(n): qml.RY(0.3*(i+1)*(l+1), wires=i)
                    for i in range(n-1): qml.CNOT(wires=[i, i+1])
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            t_m, r_m = timed(c_m)
        except (TimeoutError, Exception): pass

        col_dq = fmt(t_dq) if n <= 22 else "      OOM"
        rows.append([f"{n}", col_dq, fmt(t_dt), fmt(t_m), sp(t_dt, t_m)])

    table(rows, ["Qubits", "default.qubit", "default.tensor", "maestro (MPS)", "vs dt"])


# ══════════════════════════════════════════════════════════
# BENCHMARK 3: MPS Shot Sampling (only Maestro can do it)
# ══════════════════════════════════════════════════════════

def bench_mps_shots():
    header("Benchmark 3 — MPS Shot Sampling:  10 000 shots")
    print("  RY + CNOT ladder (3 layers)  •  counts  •  bond dim = 128")
    print("  Note: default.tensor does NOT support shots")
    print()

    rows = []
    for n in [20, 50, 100, 200, 500, 1000]:
        t_m = None
        n_unique = 0
        try:
            dev = qml.device("maestro.qubit", wires=n, shots=10_000,
                             simulation_type="MatrixProductState",
                             max_bond_dimension=128)
            @qml.qnode(dev)
            def c_m():
                for l in range(3):
                    for i in range(n): qml.RY(0.3*(i+1)*(l+1), wires=i)
                    for i in range(n-1): qml.CNOT(wires=[i, i+1])
                return qml.counts()
            t_m, r_m = timed(c_m)
            n_unique = len(r_m)
        except (TimeoutError, Exception): pass

        rows.append([f"{n}", fmt(t_m), f"{n_unique:,}"])

    table(rows, ["Qubits", "maestro (MPS)", "Unique Samples"])
    print()
    print("  (default.tensor, Qiskit Aer MPS: NOT SUPPORTED for shot sampling)")


# ══════════════════════════════════════════════════════════
# BENCHMARK 4: Hamiltonian — Batched Estimation
# ══════════════════════════════════════════════════════════

def bench_hamiltonian():
    header("Benchmark 4 — Hamiltonian:  Full 2-Local (all-to-all ZZ + X)")
    print("  StronglyEntanglingLayers (4 layers)  •  ⟨H⟩  •  analytic")
    print()

    rows = []
    for n in [10, 14, 18, 20, 22]:
        np.random.seed(SEED)
        coeffs, ops = [], []
        for i in range(n):
            for j in range(i+1, n):
                coeffs.append(np.random.uniform(-1, 1))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        for i in range(n):
            coeffs.append(np.random.uniform(-1, 1))
            ops.append(qml.PauliX(i))
        H = qml.Hamiltonian(coeffs, ops)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=n)
        params = np.random.random(shape)

        def circ(w):
            qml.StronglyEntanglingLayers(w, wires=range(n))
            return qml.expval(H)

        t_dq, r_dq = try_run("default.qubit", n, circ, params)
        t_m,  r_m  = try_run("maestro.qubit", n, circ, params)
        ok = "✓" if r_dq is not None and r_m is not None and np.isclose(r_dq, r_m, atol=1e-4) else "—"
        rows.append([f"{n}", f"{len(ops)}", fmt(t_dq), fmt(t_m), sp(t_dq, t_m), ok])

    table(rows, ["Qubits", "Terms", "default.qubit", "maestro.qubit", "Speedup", "OK"])


# ══════════════════════════════════════════════════════════
# BENCHMARK 5: Gradient (parameter-shift)
# ══════════════════════════════════════════════════════════

def bench_gradient():
    header("Benchmark 5 — Gradient:  Parameter-Shift Differentiation")
    print("  RY + CNOT ladder  •  ⟨Σ ZᵢZᵢ₊₁⟩  •  analytic")
    print()

    rows = []
    for n in [8, 12, 16, 18, 20, 22]:
        np.random.seed(SEED)
        params = qml.numpy.array(np.random.random(n), requires_grad=True)
        H = qml.Hamiltonian([1.0]*(n-1),
                            [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n-1)])

        def circ(p):
            for i in range(n): qml.RY(p[i], wires=i)
            for i in range(n-1): qml.CNOT(wires=[i, i+1])
            return qml.expval(H)

        t_dq = t_lq = t_m = None
        g_dq = g_m = None

        try:
            dev = qml.device("default.qubit", wires=n)
            q = qml.QNode(circ, dev, diff_method="parameter-shift")
            t_dq, g_dq = timed_grad(qml.grad(q), params)
        except TimeoutError: pass

        try:
            dev = qml.device("lightning.qubit", wires=n)
            q = qml.QNode(circ, dev, diff_method="parameter-shift")
            t_lq, _ = timed_grad(qml.grad(q), params)
        except TimeoutError: pass

        try:
            dev = qml.device("maestro.qubit", wires=n)
            q = qml.QNode(circ, dev, diff_method="parameter-shift")
            t_m, g_m = timed_grad(qml.grad(q), params)
        except TimeoutError: pass

        ok = "✓" if g_dq is not None and g_m is not None and np.allclose(g_dq, g_m, atol=1e-4) else "—"
        rows.append([f"{n}", fmt(t_dq), fmt(t_lq), fmt(t_m), sp(t_dq, t_m), sp(t_lq, t_m), ok])

    table(rows, ["Qubits", "default.qubit", "lightning.qubit", "maestro.qubit",
                 "vs dq", "vs lq", "OK"])


# ══════════════════════════════════════════════════════════
# BENCHMARK 6: Shot Sampling (Statevector)
# ══════════════════════════════════════════════════════════

def bench_shots():
    header("Benchmark 6 — Shot Sampling:  10 000 shots (statevector)")
    print("  StronglyEntanglingLayers (2 layers)  •  ⟨Z₀⟩ + counts")
    print()

    rows = []
    for n in [14, 18, 20, 22, 24]:
        np.random.seed(SEED)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n)
        params = np.random.random(shape)

        def circ(w):
            qml.StronglyEntanglingLayers(w, wires=range(n))
            return qml.expval(qml.PauliZ(0)), qml.counts()

        t_dq, _ = try_run("default.qubit", n, circ, params, shots=10_000)
        t_lq, _ = try_run("lightning.qubit", n, circ, params, shots=10_000)
        t_m,  _ = try_run("maestro.qubit", n, circ, params, shots=10_000)

        rows.append([f"{n}", fmt(t_dq), fmt(t_lq), fmt(t_m), sp(t_dq, t_m), sp(t_lq, t_m)])

    table(rows, ["Qubits", "default.qubit", "lightning.qubit", "maestro.qubit",
                 "vs dq", "vs lq"])


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ╔════════════════════════════════════════════════════════════════════╗")
    print("  ║   pennylane-maestro  •  Performance Benchmark Suite               ║")
    print("  ║   Qoro Quantum — Maestro Simulator                                ║")
    print("  ║                                                                    ║")
    print(f"  ║   PennyLane {qml.__version__}  •  Timeout {TIMEOUT}s/point  •  {RUNS} runs/point          ║")
    print("  ╚════════════════════════════════════════════════════════════════════╝")

    bench_statevector()
    bench_mps_scale()
    bench_mps_shots()
    bench_hamiltonian()
    bench_gradient()
    bench_shots()

    print()
    print("  Done. All benchmarks completed.")
    print()
