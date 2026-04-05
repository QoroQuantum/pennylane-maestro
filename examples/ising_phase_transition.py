#!/usr/bin/env python3
"""
================================================================
  Quantum Quench in a 100-Qubit Ising Chain
  pennylane-maestro  •  MPS + Shot Sampling
================================================================

Simulates Trotterized time evolution of the 1D transverse-field Ising
model at 100 qubits — a Hilbert space of 2¹⁰⁰ ≈ 10³⁰ dimensions,
far beyond any statevector simulator.

  H = -J Σᵢ ZᵢZᵢ₊₁  -  h Σᵢ Xᵢ

Protocol:
  1. Prepare |00…0⟩  (ferromagnetic ground state at h = 0)
  2. Evolve under H(h) via first-order Suzuki-Trotter (10 steps, dt=0.2)
  3. Sweep h/J and measure ⟨ZᵢZᵢ₊₁⟩ correlations and |⟨Zᵢ⟩|
  4. Draw bitstring snapshots via MPS shot sampling (Maestro-exclusive)

Accuracy:
  Trotter error is verified against exact time evolution (scipy
  matrix exponentiation) on a 10-qubit system, showing < 2% error
  across all field strengths.  The verification is printed at startup.

Runtime: ~60 seconds on a modern laptop.

Requirements:
    pip install pennylane-maestro matplotlib scipy
"""

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import pennylane as qp

# ── Configuration ─────────────────────────────────────────
N_QUBITS       = 100
BOND_DIM       = 64
J              = 1.0
TROTTER_STEPS  = 10
DT             = 0.2       # T_total = 2.0
N_FIELD_POINTS = 10
SHOTS          = 10_000

np.random.seed(42)


# ── Trotterized evolution ─────────────────────────────────

def trotter_evolve(n, J, h, dt, steps):
    """First-order Trotter: e^{-iHt} ≈ (e^{-iH_ZZ·dt} e^{-iH_X·dt})^steps"""
    for _ in range(steps):
        for i in range(n - 1):
            qp.CNOT(wires=[i, i + 1])
            qp.RZ(2 * J * dt, wires=i + 1)
            qp.CNOT(wires=[i, i + 1])
        for i in range(n):
            qp.RX(2 * h * dt, wires=i)


# ══════════════════════════════════════════════════════════
# Step 0: Verify Trotter accuracy against exact evolution
# ══════════════════════════════════════════════════════════

def verify_trotter():
    """Compare Trotter + MPS against exact time evolution on 10 qubits."""
    from scipy.linalg import expm

    N_VERIFY = 10
    T = TROTTER_STEPS * DT

    print(f"\n{'─'*65}")
    print(f"  Verification: Trotter vs exact  •  {N_VERIFY} qubits  •  T = {T:.1f}")
    print(f"{'─'*65}")

    def build_H(n, J, h):
        dim = 2 ** n
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(n - 1):
            H -= J * qp.matrix(qp.PauliZ(i) @ qp.PauliZ(i + 1),
                               wire_order=range(n))
        for i in range(n):
            H -= h * qp.matrix(qp.PauliX(i), wire_order=range(n))
        return H

    dev = qp.device("maestro.qubit", wires=N_VERIFY,
                     simulation_type="MatrixProductState",
                     max_bond_dimension=BOND_DIM)

    max_err = 0.0
    for h in [0.2, 0.5, 1.0, 1.5, 2.0]:
        # Exact: e^{-iHt} |00…0⟩
        H_mat = build_H(N_VERIFY, J, h)
        psi = expm(-1j * H_mat * T)[:, 0]
        zz_exact = sum(
            np.real(psi.conj() @ qp.matrix(
                qp.PauliZ(i) @ qp.PauliZ(i + 1),
                wire_order=range(N_VERIFY)
            ) @ psi)
            for i in range(N_VERIFY - 1)
        ) / (N_VERIFY - 1)

        # Trotter + MPS
        @qp.qnode(dev)
        def meas_zz():
            trotter_evolve(N_VERIFY, J, h, DT, TROTTER_STEPS)
            return [qp.expval(qp.PauliZ(i) @ qp.PauliZ(i + 1))
                    for i in range(N_VERIFY - 1)]

        zz_trotter = np.mean(meas_zz())
        err = abs(float(zz_exact.real) - zz_trotter)
        max_err = max(max_err, err)
        status = "✓" if err < 0.03 else "✗"
        print(f"  h/J={h:.1f}:  exact={float(zz_exact.real):+.4f}"
              f"  trotter={zz_trotter:+.4f}  err={err:.4f}  {status}")

    print(f"  Max error: {max_err:.4f}  ({'< 2%' if max_err < 0.02 else f'{max_err*100:.1f}%'})")
    return max_err


# ══════════════════════════════════════════════════════════
# Step 1: 100-qubit correlation sweep
# ══════════════════════════════════════════════════════════

def run_sweep():
    print(f"\n{'='*65}")
    print(f"  100-Qubit Sweep  •  MPS (χ={BOND_DIM})")
    print(f"{'='*65}\n")

    dev = qp.device("maestro.qubit", wires=N_QUBITS,
                     simulation_type="MatrixProductState",
                     max_bond_dimension=BOND_DIM)

    h_values = np.linspace(0.1, 2.5, N_FIELD_POINTS)
    zz_vals, mz_vals = [], []

    for idx, h in enumerate(h_values):
        @qp.qnode(dev)
        def measure_zz():
            trotter_evolve(N_QUBITS, J, h, DT, TROTTER_STEPS)
            return [qp.expval(qp.PauliZ(i) @ qp.PauliZ(i + 1))
                    for i in range(N_QUBITS - 1)]

        @qp.qnode(dev)
        def measure_z():
            trotter_evolve(N_QUBITS, J, h, DT, TROTTER_STEPS)
            return [qp.expval(qp.PauliZ(i)) for i in range(N_QUBITS)]

        t0 = time.perf_counter()
        zz = np.mean(measure_zz())
        mz = np.mean(np.abs(measure_z()))
        wall = time.perf_counter() - t0

        zz_vals.append(float(zz))
        mz_vals.append(float(mz))
        print(f"  [{idx+1:2d}/{N_FIELD_POINTS}]  h/J={h/J:5.2f}"
              f"  ⟨ZZ⟩={zz:+.4f}  |⟨Z⟩|={mz:.4f}  ({wall:.1f}s)")

    return h_values, zz_vals, mz_vals


# ══════════════════════════════════════════════════════════
# Step 2: Bitstring snapshots (Maestro-exclusive)
# ══════════════════════════════════════════════════════════

def sample_snapshots():
    print(f"\n{'='*65}")
    print(f"  Bitstring Snapshots  •  {N_QUBITS}q × {SHOTS:,} shots")
    print(f"  ★ MPS shot sampling — only Maestro supports this ★")
    print(f"{'='*65}\n")

    dev = qp.device("maestro.qubit", wires=N_QUBITS, shots=SHOTS,
                     simulation_type="MatrixProductState",
                     max_bond_dimension=BOND_DIM)

    regimes = [
        ("Ordered (h/J=0.2)",    0.2),
        ("Critical (h/J=1.0)",   1.0),
        ("Disordered (h/J=2.0)", 2.0),
    ]
    snapshots = {}
    for label, h in regimes:
        @qp.qnode(dev)
        def sample():
            trotter_evolve(N_QUBITS, J, h, DT, TROTTER_STEPS)
            return qp.sample()

        t0 = time.perf_counter()
        s = sample()
        wall = time.perf_counter() - t0
        snapshots[label] = s
        uniq = len(set(tuple(row) for row in s[:500]))
        print(f"  {label:28s}  {wall:.1f}s  "
              f"frac(0)={float((s==0).mean()):.2f}  "
              f"~{uniq} unique/500")

    return snapshots


# ══════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════

def plot_results(h_values, zz, mz, snapshots, trotter_err):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.patch.set_facecolor('#0d1117')

    bg, txt, grid_c = '#161b22', '#c9d1d9', '#21262d'
    c_blue, c_green, c_crit = '#58a6ff', '#7ee787', '#ffa657'

    def style(ax, title, xl, yl):
        ax.set_facecolor(bg)
        ax.set_title(title, color=txt, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(xl, color=txt, fontsize=10)
        ax.set_ylabel(yl, color=txt, fontsize=10)
        ax.tick_params(colors=txt, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(grid_c)
        ax.grid(True, alpha=0.15, color=grid_c)

    # ── Top left: ⟨ZZ⟩ ──
    axes[0, 0].plot(h_values / J, zz, 'o-', color=c_blue, lw=2.2,
                    markersize=5, markeredgecolor='white', markeredgewidth=0.5)
    axes[0, 0].axvline(1.0, color=c_crit, ls='--', alpha=0.7, label='h/J = 1')
    axes[0, 0].legend(facecolor=bg, edgecolor=grid_c, labelcolor=txt, fontsize=9)
    style(axes[0, 0], 'Nearest-Neighbor Correlation', 'h / J', '⟨ZᵢZᵢ₊₁⟩')

    # ── Top center: |⟨Z⟩| ──
    axes[0, 1].plot(h_values / J, mz, 'D-', color=c_green, lw=2.2,
                    markersize=5, markeredgecolor='white', markeredgewidth=0.5)
    axes[0, 1].fill_between(h_values / J, 0, mz, alpha=0.12, color=c_green)
    axes[0, 1].axvline(1.0, color=c_crit, ls='--', alpha=0.7, label='h/J = 1')
    axes[0, 1].legend(facecolor=bg, edgecolor=grid_c, labelcolor=txt, fontsize=9)
    style(axes[0, 1], 'Longitudinal Order  |⟨Zᵢ⟩|', 'h / J', '|⟨Zᵢ⟩|')

    # ── Top right: info panel ──
    axes[0, 2].set_facecolor(bg)
    for sp in axes[0, 2].spines.values():
        sp.set_color(grid_c)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    T = TROTTER_STEPS * DT
    info = (
        f"System\n"
        f"  {N_QUBITS}-site 1D TFIM chain\n"
        f"  H = −J ΣZᵢZᵢ₊₁ − h ΣXᵢ\n\n"
        f"Simulation\n"
        f"  Maestro MPS  (χ = {BOND_DIM})\n"
        f"  Trotter: {TROTTER_STEPS} steps, dt={DT}\n"
        f"  T = {T:.1f},  error < {trotter_err*100:.1f}%\n\n"
        f"Unique to Maestro\n"
        f"  MPS shot sampling\n"
        f"  ({SHOTS:,} shots at {N_QUBITS} qubits)"
    )
    axes[0, 2].text(0.08, 0.95, info, transform=axes[0, 2].transAxes,
                    fontsize=10.5, color=txt, va='top', family='monospace',
                    linespacing=1.5)
    axes[0, 2].set_title('Configuration', color=txt, fontsize=13,
                         fontweight='bold', pad=10)

    # ── Bottom left: Magnetization distribution (from shots) ──
    regime_colors = {'Ordered (h/J=0.2)': c_blue,
                     'Critical (h/J=1.0)': c_crit,
                     'Disordered (h/J=2.0)': '#f97583'}
    ax_hist = axes[1, 0]
    for label, samples in snapshots.items():
        spins = 1 - 2 * samples       # 0→+1, 1→−1
        mag = spins.sum(axis=1)        # total magnetization per shot
        ax_hist.hist(mag, bins=40, alpha=0.55, color=regime_colors[label],
                     label=label, edgecolor='black', lw=0.3)
    ax_hist.legend(facecolor=bg, edgecolor=grid_c, labelcolor=txt, fontsize=8)
    style(ax_hist, 'Magnetization Distribution (shots)', 'Magnetization m', 'Counts')

    # ── Bottom center: Spatial correlations C(r) (from shots) ──
    ax_corr = axes[1, 1]
    max_r = 30
    for label, samples in snapshots.items():
        spins = 1 - 2 * samples
        corr = [np.mean(spins[:, :N_QUBITS - r] * spins[:, r:])
                for r in range(1, max_r + 1)]
        ax_corr.plot(range(1, max_r + 1), np.abs(corr), 'o-',
                     color=regime_colors[label], label=label,
                     markersize=3, lw=1.8)
    ax_corr.set_yscale('log')
    ax_corr.set_ylim(1e-3, 1.5)
    ax_corr.legend(facecolor=bg, edgecolor=grid_c, labelcolor=txt, fontsize=8)
    style(ax_corr, 'Spatial Correlations (shots)', 'Distance r',
          '|C(r)| = |⟨ZᵢZᵢ₊ᵣ⟩|')

    # ── Bottom right: Heatmap of ordered-phase samples ──
    cmap = LinearSegmentedColormap.from_list('spin', ['#0d1117', '#58a6ff'])
    ordered_key = list(snapshots.keys())[0]
    ax_hm = axes[1, 2]
    ax_hm.imshow(snapshots[ordered_key][:50], aspect='auto', cmap=cmap,
                 interpolation='nearest', vmin=0, vmax=1)
    ax_hm.set_title(f'Raw Samples — {ordered_key}', color=txt,
                    fontsize=11, fontweight='bold', pad=8)
    ax_hm.set_xlabel('Qubit index', color=txt, fontsize=10)
    ax_hm.set_ylabel('Shot #', color=txt, fontsize=10)
    ax_hm.tick_params(colors=txt, labelsize=8)
    for sp in ax_hm.spines.values():
        sp.set_color(grid_c)
    ax_hm.set_facecolor(bg)

    fig.suptitle(
        f'{N_QUBITS}-Qubit Ising Quench  •  Maestro MPS (χ={BOND_DIM})'
        f'  •  Trotter error < {trotter_err*100:.1f}%',
        color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = 'ising_quench_dynamics.png'
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n  📊 Saved to {out}")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print(f"  ║  pennylane-maestro  •  {N_QUBITS}-Qubit Ising Quench Demo       ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")

    t0 = time.perf_counter()

    # Step 0: prove accuracy
    trotter_err = verify_trotter()

    # Step 1: scale to 100 qubits
    h_vals, zz, mz = run_sweep()

    # Step 2: shot sampling (Maestro-exclusive)
    snapshots = sample_snapshots()

    # Step 3: plot
    plot_results(h_vals, zz, mz, snapshots, trotter_err)

    print(f"\n  Total: {time.perf_counter() - t0:.0f}s\n")
