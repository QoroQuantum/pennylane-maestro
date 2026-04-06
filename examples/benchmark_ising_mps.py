#!/usr/bin/env python3
"""
================================================================
  MPS Benchmark — Ising Quench across PennyLane backends
================================================================

Benchmarks Trotterized time evolution of the 1D transverse-field
Ising model across all available PennyLane backends:

  • default.qubit     (statevector, Python)
  • lightning.qubit    (statevector, C++)
  • default.tensor     (MPS via quimb)
  • maestro.qubit      (MPS via Maestro C++)

Statevector backends go OOM beyond ~26 qubits.  The benchmark
sweeps from 20 to 500 qubits, highlighting where MPS is required
and how Maestro compares to quimb.

Produces a bar chart saved to benchmark_ising_mps.png.

Requirements:
    pip install pennylane-maestro pennylane-lightning matplotlib
    pip install quimb  # for default.tensor
"""

import time
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import pennylane as qp

# ── Configuration ─────────────────────────────────────────
J              = 1.0
H_FIELD        = 1.0
DT             = 0.2
TROTTER_STEPS  = 10
BOND_DIM       = 128
QUBIT_COUNTS   = [20, 30, 50, 100, 200, 500]
SV_LIMIT       = 26       # statevector backends OOM above this


# ── Circuit ───────────────────────────────────────────────

def trotter_evolve(n, J, h, dt, steps):
    """First-order Trotter step for the 1D TFIM."""
    for _ in range(steps):
        for i in range(n - 1):
            qp.CNOT(wires=[i, i + 1])
            qp.RZ(2 * J * dt, wires=i + 1)
            qp.CNOT(wires=[i, i + 1])
        for i in range(n):
            qp.RX(2 * h * dt, wires=i)


# ── Backend definitions ──────────────────────────────────

def make_backends(n):
    """Return list of (name, device) pairs for a given qubit count."""
    backends = []

    if n <= SV_LIMIT:
        backends.append(("default.qubit",
                         qp.device("default.qubit", wires=n)))
        backends.append(("lightning.qubit",
                         qp.device("lightning.qubit", wires=n)))

    try:
        backends.append(("default.tensor",
                         qp.device("default.tensor", wires=n,
                                   max_bond_dim=BOND_DIM)))
    except Exception:
        pass

    backends.append(("maestro.qubit",
                     qp.device("maestro.qubit", wires=n,
                               simulation_type="MatrixProductState",
                               max_bond_dimension=BOND_DIM)))
    return backends


# ── Benchmark ─────────────────────────────────────────────

def run_benchmarks():
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║  MPS Benchmark — Ising Quench (TFIM 1D)              ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print(f"\n  Trotter: {TROTTER_STEPS} steps × dt={DT}  |"
          f"  χ={BOND_DIM}  |  J={J}, h={H_FIELD}\n")

    results = {}

    for n in QUBIT_COUNTS:
        results[n] = {}
        print(f"  ── {n} qubits {'─' * 40}")

        backends = make_backends(n)

        for name, dev in backends:
            @qp.qnode(dev)
            def circuit():
                trotter_evolve(n, J, H_FIELD, DT, TROTTER_STEPS)
                return [qp.expval(qp.PauliZ(i) @ qp.PauliZ(i + 1))
                        for i in range(n - 1)]

            try:
                t0 = time.perf_counter()
                zz = float(np.mean(circuit()))
                wall = time.perf_counter() - t0
                results[n][name] = wall
                print(f"     {name:20s}  {wall:8.3f}s  ⟨ZZ⟩={zz:+.4f}")
            except Exception as e:
                results[n][name] = None
                print(f"     {name:20s}  FAIL ({type(e).__name__})")

        # Mark OOM for skipped backends
        if n > SV_LIMIT:
            results[n]["default.qubit"] = None
            results[n]["lightning.qubit"] = None
            print(f"     {'default.qubit':20s}  OOM")
            print(f"     {'lightning.qubit':20s}  OOM")

        print()

    return results


# ── Plot ──────────────────────────────────────────────────

def plot_results(results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    qubits = list(results.keys())
    x = np.arange(len(qubits))
    bar_w = 0.2

    colors = {
        "default.qubit":  '#8b949e',
        "lightning.qubit": '#f0883e',
        "default.tensor": '#a371f7',
        "maestro.qubit":  '#58a6ff',
    }
    labels = {
        "default.qubit":  'default.qubit',
        "lightning.qubit": 'lightning.qubit',
        "default.tensor": 'default.tensor (quimb)',
        "maestro.qubit":  'maestro.qubit (MPS)',
    }
    offsets = {
        "default.qubit":  -1.5,
        "lightning.qubit": -0.5,
        "default.tensor":  0.5,
        "maestro.qubit":   1.5,
    }

    for name in ["default.qubit", "lightning.qubit",
                 "default.tensor", "maestro.qubit"]:
        vals = [results[q].get(name) for q in qubits]
        bar_vals = [v if v else 0 for v in vals]
        ax.bar(x + offsets[name] * bar_w, bar_vals, bar_w,
               color=colors[name], label=labels[name],
               edgecolor='white', lw=0.3, zorder=3)

        # OOM labels
        for i, v in enumerate(vals):
            if v is None:
                ax.text(x[i] + offsets[name] * bar_w, 0.15, 'OOM',
                        ha='center', va='bottom', fontsize=7,
                        color=colors[name], fontweight='bold', rotation=90)

    # Speedup labels (Maestro vs default.tensor)
    for i, q in enumerate(qubits):
        t_dt = results[q].get("default.tensor")
        t_mq = results[q].get("maestro.qubit")
        if t_dt and t_mq and t_mq > 0:
            sp = t_dt / t_mq
            ax.text(x[i] + 1.5 * bar_w, t_mq + 0.3, f'{sp:.0f}×',
                    ha='center', va='bottom', fontsize=9,
                    color=colors["maestro.qubit"], fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in qubits], fontsize=12,
                       color='#c9d1d9')
    ax.set_xlabel('Qubits', fontsize=13, color='#c9d1d9', labelpad=10)
    ax.set_ylabel('Runtime (seconds, log scale)', fontsize=13,
                  color='#c9d1d9', labelpad=10)
    ax.tick_params(colors='#c9d1d9', labelsize=10)
    for sp in ax.spines.values():
        sp.set_color('#21262d')
    ax.grid(True, alpha=0.1, color='#c9d1d9', axis='y')
    ax.set_ylim(0.05, 300)

    ax.legend(loc='upper left', fontsize=10, facecolor='#161b22',
              edgecolor='#21262d', labelcolor='#c9d1d9')
    ax.set_title(
        f'Ising Quench  •  Trotter ({TROTTER_STEPS} steps, χ={BOND_DIM})',
        color='white', fontsize=14, fontweight='bold', pad=15)

    fig.text(0.5, 0.01,
             'pennylane-maestro  •  pip install pennylane-maestro',
             ha='center', color='#8b949e', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out = 'benchmark_ising_mps.png'
    plt.savefig(out, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.show()
    print(f"  📊 Saved to {out}")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    results = run_benchmarks()
    plot_results(results)

    print(f"\n  Total: {time.perf_counter() - t0:.0f}s")
    print(f"\n  Raw data:")
    print(f"  {json.dumps(results, indent=2)}\n")
