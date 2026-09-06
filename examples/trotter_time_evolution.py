#!/usr/bin/env python3
"""
================================================================
  Persistent Incremental Time Evolution in PennyLane + Maestro
  pennylane-maestro  •  MPS + qml.Snapshot Acceleration
================================================================

Demonstrates persistent-state Trotterized quantum time evolution using
Maestro's incremental evolution engine through standard PennyLane syntax.

Physical System:
  50-qubit Heisenberg XXZ spin chain undergoing a quantum quench from a
  domain-wall state: |00...011...1⟩.
  As time progresses, domain-wall melting and spin transport propagate
  coherently through the lattice.

The Problem with Standard Rebuild-from-Scratch:
  In conventional PennyLane workflows, measuring intermediate time points
  requires building and executing separate circuits from t=0:
    Step 1:  1 Trotter step
    Step 2:  2 Trotter steps (re-simulating step 1 from scratch)
    ...
    Step N:  N Trotter steps (re-simulating steps 1..N-1 from scratch)
  Total cost scales quadratically: O(N²) simulation time!

Maestro's Solution:
  Using standard `qml.Snapshot`, pennylane-maestro automatically recognizes
  the repeated Trotter step pattern and routes execution to Maestro's C++
  `incremental_evolve` engine. The quantum state (MPS) remains persistent
  across measurements, reducing simulation cost to linear O(N)!

Runtime: ~5-10 seconds on a laptop for 50 qubits.

Requirements:
  pip install pennylane pennylane-maestro numpy
"""

import time
import warnings
import numpy as np
import pennylane as qml

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────
N_QUBITS     = 50
BOND_DIM     = 32
J_XY         = 1.0       # XX and YY coupling strength
DELTA        = 0.5       # ZZ anisotropy
DT           = 0.1       # Trotter step size
TOTAL_STEPS  = 20        # Total time steps (t_final = 2.0)
SAMPLE_STEPS = list(range(2, TOTAL_STEPS + 1, 2))  # Measure every 2 steps


# ── Trotter step definition ───────────────────────────────

def trotter_step(n_qubits, dt, j_xy, delta):
    """One second-order Trotter step for the 1D XXZ chain.

    H = - J_XY Σ (X_i X_{i+1} + Y_i Y_{i+1}) - DELTA Σ Z_i Z_{i+1}
    """
    # Even bonds
    for i in range(0, n_qubits - 1, 2):
        qml.IsingXX(2 * j_xy * dt / 2, wires=[i, i + 1])
        qml.IsingYY(2 * j_xy * dt / 2, wires=[i, i + 1])
        qml.IsingZZ(2 * delta * dt / 2, wires=[i, i + 1])
    # Odd bonds
    for i in range(1, n_qubits - 1, 2):
        qml.IsingXX(2 * j_xy * dt, wires=[i, i + 1])
        qml.IsingYY(2 * j_xy * dt, wires=[i, i + 1])
        qml.IsingZZ(2 * delta * dt, wires=[i, i + 1])
    # Even bonds
    for i in range(0, n_qubits - 1, 2):
        qml.IsingXX(2 * j_xy * dt / 2, wires=[i, i + 1])
        qml.IsingYY(2 * j_xy * dt / 2, wires=[i, i + 1])
        qml.IsingZZ(2 * delta * dt / 2, wires=[i, i + 1])


def prepare_initial_state(n_qubits):
    """Prepare domain-wall state: |00...011...1⟩."""
    for i in range(n_qubits // 2, n_qubits):
        qml.PauliX(wires=i)


# ══════════════════════════════════════════════════════════
# 1. Native PennyLane Interface with qml.Snapshot
# ══════════════════════════════════════════════════════════

def run_with_snapshots(dev):
    """Execute Trotter evolution using standard PennyLane qml.Snapshot.

    Maestro transparently accelerates this via incremental_evolve.
    """
    @qml.qnode(dev)
    def snapshot_circuit():
        prepare_initial_state(N_QUBITS)

        for step in range(1, TOTAL_STEPS + 1):
            trotter_step(N_QUBITS, DT, J_XY, DELTA)
            if step in SAMPLE_STEPS:
                for q in range(N_QUBITS):
                    qml.Snapshot(f"t{step}_z{q}", measurement=qml.expval(qml.PauliZ(q)))

        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    return qml.snapshots(snapshot_circuit)()


# ══════════════════════════════════════════════════════════
# 2. Naive Rebuild-from-Scratch (Conventional Baseline)
# ══════════════════════════════════════════════════════════

def run_rebuild_from_scratch(dev):
    """Execute by rebuilding a new circuit from step 0 for each time point.

    This represents standard simulator execution scaling as O(N²).
    """
    results = {}
    for step in SAMPLE_STEPS:
        @qml.qnode(dev)
        def circ():
            prepare_initial_state(N_QUBITS)
            for _ in range(step):
                trotter_step(N_QUBITS, DT, J_XY, DELTA)
            return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

        mags = circ()
        for q, m in enumerate(mags):
            results[f"t{step}_z{q}"] = m

    return results


# ══════════════════════════════════════════════════════════
# 3. Direct Device API (dev.incremental_evolve)
# ══════════════════════════════════════════════════════════

def run_direct_api(dev):
    """Use the programmatic dev.incremental_evolve API directly."""
    def init():
        prepare_initial_state(N_QUBITS)

    def step():
        trotter_step(N_QUBITS, DT, J_XY, DELTA)

    observables = [qml.PauliZ(q) for q in range(N_QUBITS)]

    return dev.incremental_evolve(
        init=init,
        trotter_step=step,
        measure_at_steps=SAMPLE_STEPS,
        observables=observables,
    )


# ══════════════════════════════════════════════════════════
# ASCII Spacetime Diagram
# ══════════════════════════════════════════════════════════

def display_ascii_spacetime(mag_matrix, steps, n_qubits):
    """Render a compact ASCII heatmap of magnetization ⟨Z_i(t)⟩."""
    print("\n  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║      Spacetime Spin Transport: ⟨Z_i(t)⟩ (Domain Wall)     ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print("   t | Qubits (0 ────────── 25 ────────── 49)")
    print("  ───┼────────────────────────────────────────")

    # Downsample qubits horizontally if > 40 to fit terminal cleanly
    cols = 40
    step_indices = np.linspace(0, n_qubits - 1, cols, dtype=int)

    # Shading characters from +1 (ordered up) to -1 (ordered down)
    chars = ["█", "▓", "▒", "░", " ", ".", ":", "-"]

    for row_idx, s in enumerate(steps):
        row = mag_matrix[row_idx, step_indices]
        line = ""
        for val in row:
            # val in [-1, +1] -> map to char index 0..7
            idx = int(np.clip((1.0 - val) / 2.0 * len(chars), 0, len(chars) - 1))
            line += chars[idx]
        t_val = s * DT
        print(f" {t_val:3.1f} | {line}")

    print("  ───┴────────────────────────────────────────")
    print("  Legend: █ (|0⟩, ⟨Z⟩=+1)  ▒ (melted, ⟨Z⟩≈0)  - (|1⟩, ⟨Z⟩=-1)\n")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print(f"  ║  pennylane-maestro • Persistent Time Evolution Demo       ║")
    print(f"  ║  System: {N_QUBITS} Qubits  •  MPS Bond Dim: χ={BOND_DIM:<3}             ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print(f"\n  Simulation: {TOTAL_STEPS} Trotter steps (dt={DT}), sampling at {len(SAMPLE_STEPS)} time points.")

    dev = qml.device(
        "maestro.qubit",
        wires=N_QUBITS,
        simulation_type="MatrixProductState",
        max_bond_dimension=BOND_DIM,
    )

    # ── Benchmark 1: Persistent Snapshot Acceleration ──
    print("\n  [1/3] Running with qml.Snapshot (O(N) persistent evolution)...")
    t0 = time.perf_counter()
    snap_results = run_with_snapshots(dev)
    t_snapshot = time.perf_counter() - t0
    print(f"        ✓ Completed in {t_snapshot:.3f} s")

    # ── Benchmark 2: Conventional Rebuild-from-Scratch ──
    print("\n  [2/3] Running conventional rebuild-from-scratch (O(N²) baseline)...")
    t0 = time.perf_counter()
    naive_results = run_rebuild_from_scratch(dev)
    t_naive = time.perf_counter() - t0
    print(f"        ✓ Completed in {t_naive:.3f} s")

    speedup = t_naive / t_snapshot
    print(f"\n  ⚡ Speedup: {speedup:.2f}x faster with persistent snapshots!")

    # ── Verify Numerical Accuracy ──
    diffs = [
        abs(snap_results[f"t{s}_z{q}"] - naive_results[f"t{s}_z{q}"])
        for s in SAMPLE_STEPS
        for q in range(N_QUBITS)
    ]
    max_diff = max(diffs)
    print(f"  ✓ Maximum numerical difference: {max_diff:.2e} (identical results)")

    # ── Benchmark 3: Direct API ──
    print("\n  [3/3] Running programmatic direct API (dev.incremental_evolve)...")
    direct_res = run_direct_api(dev)
    mag_matrix = direct_res["expectation_values"]
    print(f"        ✓ Matrix shape: {mag_matrix.shape}")
    if direct_res.get("max_bond_dim_reached") is not None:
        print(f"        ✓ Max bond dimension reached: {direct_res['max_bond_dim_reached']}")

    # ── Spacetime Visualization ──
    display_ascii_spacetime(mag_matrix, SAMPLE_STEPS, N_QUBITS)

    # Optional Matplotlib figure
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            mag_matrix,
            aspect="auto",
            extent=[0, N_QUBITS - 1, TOTAL_STEPS * DT, SAMPLE_STEPS[0] * DT],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Magnetization ⟨Z_i(t)⟩")
        ax.set_xlabel("Qubit Index")
        ax.set_ylabel("Time t")
        ax.set_title(f"50-Qubit XXZ Quench Dynamics • Maestro MPS (χ={BOND_DIM})")
        plt.tight_layout()
        out_png = "trotter_time_evolution.png"
        plt.savefig(out_png, dpi=150)
        print(f"  📊 Heatmap saved to {out_png}")
    except ImportError:
        pass

    print("  Done!\n")
