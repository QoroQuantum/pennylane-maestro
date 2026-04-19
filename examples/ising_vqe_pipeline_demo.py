#!/usr/bin/env python3
"""
Ising VQE Pipeline: PennyLane → Qoro Cloud → IBM QPU
=====================================================

Three-stage quantum computing workflow on the 1D transverse-field
Ising model (30 qubits), demonstrating the path from research to
production with a single codebase.

  Stage 1 — Research:   PennyLane + Maestro MPS (gradient VQE)
  Stage 2 — Cloud:      Divi + Qoro Cloud (Monte Carlo VQE)
  Stage 3 — Hardware:   Divi + IBM QPU (same MC optimizer)

Requirements:
    pip install pennylane-maestro divi matplotlib numpy
"""

import time
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


# ── Configuration ─────────────────────────────────────────────────────

N_QUBITS   = 30
N_LAYERS   = 2
J, H_FIELD = 1.0, 1.0        # ZZ coupling, transverse field
BOND_DIM   = 64               # MPS truncation
SEED       = 42
SHOTS      = 10_000


# ── Hamiltonian ───────────────────────────────────────────────────────

def build_ising_hamiltonian(n, J=1.0, h=1.0):
    """H = −J Σ ZᵢZᵢ₊₁ − h Σ Xᵢ  (open boundary conditions)."""
    coeffs, obs = [], []
    for i in range(n - 1):
        coeffs.append(-J)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(n):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, obs)


def exact_ground_state_energy(n, J=1.0, h=1.0):
    """Exact E₀ via free-fermion mapping (Pfeuty 1970)."""
    M = np.diag(np.full(n, h)) + np.diag(np.full(n - 1, J), k=-1)
    return float(-np.sum(np.linalg.svd(M, compute_uv=False)))


hamiltonian  = build_ising_hamiltonian(N_QUBITS, J, H_FIELD)
exact_energy = exact_ground_state_energy(N_QUBITS, J, H_FIELD)
n_params     = N_LAYERS * N_QUBITS * 2

print(f"System: {N_QUBITS}-qubit TFIM  (J={J}, h={H_FIELD})")
print(f"Exact E₀ = {exact_energy:.6f}")
print(f"Ansatz:  HEA (RY-RZ + linear CNOT) × {N_LAYERS} layers = {n_params} params\n")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — PennyLane + Maestro MPS
# ═══════════════════════════════════════════════════════════════════════

dev = qml.device(
    "maestro.qubit",
    wires=N_QUBITS,
    simulation_type="MatrixProductState",
    max_bond_dimension=BOND_DIM,
)

@qml.qnode(dev)
def cost_fn(params):
    idx = 0
    for _ in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])
    return qml.expval(hamiltonian)


rng    = np.random.default_rng(SEED)
params = qml.numpy.array(rng.uniform(0, 2 * np.pi, n_params), requires_grad=True)
opt    = qml.MomentumOptimizer(stepsize=0.1)

print("Stage 1: PennyLane + Maestro MPS")
stage1_energies = []
t0 = time.time()
for step in range(30):
    params, energy = opt.step_and_cost(cost_fn, params)
    stage1_energies.append(float(energy))
    if (step + 1) % 10 == 0:
        print(f"  Step {step+1:3d}:  E = {energy:.6f}")
stage1_time = time.time() - t0
best_params = np.array(params.copy())
print(f"  Done — best E = {min(stage1_energies):.6f}  ({stage1_time:.1f}s)\n")

# Save checkpoint so Stage 2/3 can be re-run independently
np.savez("ising_vqe_checkpoint.npz", params=best_params, energies=stage1_energies)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — Divi + Qoro Cloud
# ═══════════════════════════════════════════════════════════════════════

from divi.backends import ExecutionConfig, JobConfig, QoroService, SimulationMethod
from divi.qprog import VQE, GenericLayerAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer

ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ],
    entangler=qml.CNOT,
    entangling_layout="linear",
)

# Backend: Maestro cloud cluster
cloud_backend = QoroService(
    job_config=JobConfig(simulator_cluster="qoro_maestro", shots=SHOTS),
    execution_config=ExecutionConfig(
        bond_dimension=BOND_DIM,
        simulation_method=SimulationMethod.MatrixProductState,
    ),
)

optimizer = MonteCarloOptimizer(population_size=20, n_best_sets=5, keep_best_params=True)

vqe = VQE(
    hamiltonian=hamiltonian, ansatz=ansatz, n_layers=N_LAYERS,
    optimizer=optimizer, max_iterations=3, backend=cloud_backend, seed=SEED,
)

# Warm-start from Stage 1
population = np.tile(best_params, (20, 1))
population[1:] += rng.normal(0, 0.5, (19, n_params))

print("Stage 2: Divi + Qoro Cloud")
t0 = time.time()
vqe.run(initial_params=population, perform_final_computation=False)
stage2_time = time.time() - t0

stage2_energies = vqe.min_losses_per_iteration
cloud_params    = np.array(vqe.best_params)
for i, e in enumerate(stage2_energies):
    print(f"  Iter {i+1}:  E = {e:.6f}")
print(f"  Done — best E = {vqe.best_loss:.6f}  ({stage2_time:.1f}s)\n")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — Divi + IBM QPU
# ═══════════════════════════════════════════════════════════════════════
# One config change: simulator_cluster → qpu_system

qpu_backend = QoroService(
    job_config=JobConfig(qpu_system="superconducting_qpus", shots=SHOTS, use_circuit_packing=True),
)

optimizer_qpu = MonteCarloOptimizer(population_size=10, n_best_sets=3, keep_best_params=True)

vqe_qpu = VQE(
    hamiltonian=hamiltonian, ansatz=ansatz, n_layers=N_LAYERS,
    optimizer=optimizer_qpu, max_iterations=4, backend=qpu_backend, seed=SEED,
)

population_qpu = np.tile(cloud_params, (10, 1))
population_qpu[1:] += rng.normal(0, 0.05, (9, n_params))

print("Stage 3: Divi + IBM QPU")
t0 = time.time()
vqe_qpu.run(initial_params=population_qpu, perform_final_computation=False)
stage3_time = time.time() - t0

stage3_energies = vqe_qpu.min_losses_per_iteration
for i, e in enumerate(stage3_energies):
    print(f"  Iter {i+1}:  E = {e:.6f}")
print(f"  Done — best E = {vqe_qpu.best_loss:.6f}  ({stage3_time:.1f}s)\n")


# ═══════════════════════════════════════════════════════════════════════
# Summary + Plot
# ═══════════════════════════════════════════════════════════════════════

print("=" * 50)
print(f"{'Stage':<20} {'Best E':>10} {'Time':>8}")
print("-" * 50)
print(f"{'1. Research':<20} {min(stage1_energies):>10.4f} {stage1_time:>7.1f}s")
print(f"{'2. Cloud':<20} {min(stage2_energies):>10.4f} {stage2_time:>7.1f}s")
valid_s3 = [e for e in stage3_energies if np.isfinite(e)]
print(f"{'3. QPU':<20} {min(valid_s3):>10.4f} {stage3_time:>7.1f}s")
print(f"{'Exact':<20} {exact_energy:>10.4f}")
print("=" * 50)

# ── Convergence plot ──
x1 = np.arange(len(stage1_energies))
x2 = np.arange(len(stage2_energies)) + len(stage1_energies)
x3 = np.arange(len(valid_s3)) + len(stage1_energies) + len(stage2_energies)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x1, stage1_energies, "o-", label="Stage 1: PennyLane + Maestro MPS")
ax.plot(x2, stage2_energies, "s-", label="Stage 2: Divi + Qoro Cloud")
ax.plot(x3, valid_s3,        "D-", label="Stage 3: Divi + IBM QPU")
ax.axhline(exact_energy, color="green", linestyle=":", label=f"Exact E₀ = {exact_energy:.4f}")
ax.set_xlabel("Optimization Step")
ax.set_ylabel("Energy ⟨H⟩")
ax.set_title(f"Ising VQE Pipeline — {N_QUBITS} qubits")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ising_vqe_pipeline.png", dpi=150)
print("\nPlot saved → ising_vqe_pipeline.png")
