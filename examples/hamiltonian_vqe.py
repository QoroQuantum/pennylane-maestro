"""VQE for a transverse-field Ising chain, with adjoint gradients.

    python examples/hamiltonian_vqe.py            # 20 qubits on this machine
    python examples/hamiltonian_vqe.py --cloud    # the same on the Qoro cloud

Maestro evaluates every Pauli term of the Hamiltonian in one call, and
``diff_method="best"`` differentiates them all with the adjoint method, so a
step costs about two simulations instead of two per parameter. The result is
checked against the exact ground-state energy, which this model has in closed
form.
"""

import argparse
import time

import numpy as np
import pennylane as qp
from pennylane import numpy as pnp

parser = argparse.ArgumentParser(description="VQE for a transverse-field Ising chain.")
parser.add_argument("--cloud", action="store_true", help="run on the Qoro cloud")
parser.add_argument("--qubits", type=int, default=20)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--steps", type=int, default=80)
args = parser.parse_args()

n, J, h = args.qubits, 1.0, 1.0

# H = -J sum Z_i Z_{i+1} - h sum X_i, open chain, at the critical point h = J.
H = qp.Hamiltonian(
    [-J] * (n - 1) + [-h] * n,
    [qp.Z(i) @ qp.Z(i + 1) for i in range(n - 1)] + [qp.X(i) for i in range(n)],
)

# Exact ground-state energy from the free-fermion solution: minus the sum of
# the singular values of the bidiagonal matrix with h on the diagonal and J above it.
exact = -np.linalg.svd(np.diag([h] * n) + np.diag([J] * (n - 1), k=1), compute_uv=False).sum()

dev = qp.device("maestro.qubit", wires=n, backend="qoro" if args.cloud else "local")


# A layer of RY rotations, then brick-wall layers of CNOTs followed by RY rotations.
pairs = [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]


@qp.qnode(dev)  # diff_method="best" -> adjoint
def energy(params):
    for i in range(n):
        qp.RY(params[0, i], wires=i)
    for layer in params[1:]:
        for pair in pairs:
            qp.CNOT(wires=pair)
        for i in range(n):
            qp.RY(layer[i], wires=i)
    return qp.expval(H)


rng = np.random.default_rng(7)
params = pnp.array(rng.normal(0, 0.1, (args.layers + 1, n)), requires_grad=True)
opt = qp.AdamOptimizer(stepsize=0.05)

print(f"{n}-qubit Ising chain, {params.size} parameters, exact E0 = {exact:.4f}")
start = time.perf_counter()
for step in range(args.steps):
    params, value = opt.step_and_cost(energy, params)
    if step % 10 == 0 or step == args.steps - 1:
        print(f"step {step:3d}: E = {value:9.4f}")
elapsed = time.perf_counter() - start

final = energy(params)
print(f"final E = {final:.4f}, {abs(final - exact) / abs(exact):.2%} above exact, "
      f"{1000 * elapsed / args.steps:.0f} ms per step")
