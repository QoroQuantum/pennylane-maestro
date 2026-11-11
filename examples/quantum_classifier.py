"""A data re-uploading classifier, trained on a whole batch in one engine call.

    python examples/quantum_classifier.py            # on this machine
    python examples/quantum_classifier.py --cloud    # on the Qoro cloud

The classifier learns whether a point lies inside a circle. Each layer
re-uploads the point's coordinates as rotation angles ``w * x + b``. The
batch of points is passed as one broadcast array: locally, Maestro evaluates
and differentiates every point in a single engine call.
"""

import argparse
import time

import numpy as np
import pennylane as qp
from pennylane import numpy as pnp

parser = argparse.ArgumentParser(description="A data re-uploading classifier.")
parser.add_argument("--cloud", action="store_true", help="run on the Qoro cloud")
parser.add_argument("--points", type=int, default=200)
parser.add_argument("--steps", type=int, default=40)
args = parser.parse_args()

n_qubits, n_layers = 4, 3
rng = np.random.default_rng(0)


def make_data(count):
    """Points in [-1, 1]^2, labelled +1 inside a circle of half the square's area."""
    x = rng.uniform(-1, 1, (count, 2))
    y = np.where(np.sum(x**2, axis=1) < 2 / np.pi, 1.0, -1.0)
    return pnp.array(x, requires_grad=False), y


x_train, y_train = make_data(args.points)
x_test, y_test = make_data(args.points)

dev = qp.device("maestro.qubit", wires=n_qubits, backend="qoro" if args.cloud else "local")


@qp.qnode(dev)
def classifier(weights, biases, x):
    for layer in range(n_layers):
        for q in range(n_qubits):
            feature = x[:, q % 2]  # shape (batch,): the circuit is broadcast over the batch
            angles = weights[layer, q][:, None] * feature + biases[layer, q][:, None]  # (3, batch)
            qp.Rot(*angles, wires=q)
        for q in range(n_qubits):
            qp.CNOT(wires=[q, (q + 1) % n_qubits])
    return qp.expval(qp.Z(0))


def loss(weights, biases):
    return pnp.mean((classifier(weights, biases, x_train) - y_train) ** 2)


def accuracy(weights, biases, x, y):
    return np.mean(np.sign(classifier(weights, biases, x)) == y)


weights = pnp.array(rng.normal(0, 1, (n_layers, n_qubits, 3)), requires_grad=True)
biases = pnp.array(rng.normal(0, 0.1, (n_layers, n_qubits, 3)), requires_grad=True)
opt = qp.AdamOptimizer(stepsize=0.1)

start = time.perf_counter()
for step in range(args.steps):
    (weights, biases), value = opt.step_and_cost(loss, weights, biases)
    if step % 10 == 0 or step == args.steps - 1:
        print(f"step {step:3d}: loss {value:.3f}, train accuracy {accuracy(weights, biases, x_train, y_train):.0%}")
elapsed = time.perf_counter() - start

print(f"test accuracy {accuracy(weights, biases, x_test, y_test):.0%} on {args.points} new points; "
      f"{1000 * elapsed / args.steps:.0f} ms per step for {args.points} points")
