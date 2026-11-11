"""Run PennyLane circuits on the Qoro cloud.

Needs the cloud extra and an API key (sign up at https://dash.qoroquantum.net):

    pip install "pennylane-maestro[cloud]"
    export QORO_API_KEY=...   # or put it in a .env file

The only change from a local run is ``backend="qoro"`` on the device.
"""

import numpy as np
import pennylane as qp
from pennylane import numpy as pnp

# 1. Train a variational circuit. Each optimizer step is one cloud job, which
#    returns the energy and its adjoint gradient together.
n = 4
H = qp.Hamiltonian(
    [1.0] * (n - 1) + [0.5] * n,
    [qp.Z(i) @ qp.Z(i + 1) for i in range(n - 1)] + [qp.X(i) for i in range(n)],
)
dev = qp.device("maestro.qubit", wires=n, backend="qoro")


@qp.qnode(dev)  # diff_method="best" -> adjoint, computed in the cloud
def energy(params):
    qp.StronglyEntanglingLayers(params, wires=range(n))
    return qp.expval(H)


shape = qp.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n)
params = pnp.array(np.random.default_rng(7).uniform(0, 2 * np.pi, shape), requires_grad=True)
opt = qp.GradientDescentOptimizer(stepsize=0.1)
for step in range(10):
    params, value = opt.step_and_cost(energy, params)
    print(f"step {step}: energy {value:.4f}")

# 2. Sample a 50-qubit GHZ state with the MPS simulator in the cloud.
mps = qp.device(
    "maestro.qubit",
    wires=50,
    shots=1000,
    backend="qoro",
    simulation_type="MatrixProductState",
    max_bond_dimension=16,
)


@qp.qnode(mps)
def ghz():
    qp.Hadamard(0)
    for i in range(49):
        qp.CNOT([i, i + 1])
    return qp.counts()


counts = ghz()
print({f"{key[:4]}…{key[-4:]}": int(count) for key, count in counts.items()})  # about half all-0, half all-1

print("credits left:", dev.backend.get_credit_balance()["balance"])
