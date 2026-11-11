"""QAOA for MaxCut on a random 3-regular graph.

    python examples/qaoa_maxcut.py            # on this machine
    python examples/qaoa_maxcut.py --cloud    # on the Qoro cloud

The QAOA angles are trained with adjoint gradients, then the trained circuit
is sampled and its most frequent bitstrings are compared with the best cut,
found by brute force.
"""

import argparse
import itertools

import networkx as nx
import numpy as np
import pennylane as qp
from pennylane import numpy as pnp

parser = argparse.ArgumentParser(description="QAOA for MaxCut on a random 3-regular graph.")
parser.add_argument("--cloud", action="store_true", help="run on the Qoro cloud")
parser.add_argument("--nodes", type=int, default=12)
parser.add_argument("--depth", type=int, default=3)
parser.add_argument("--steps", type=int, default=60)
args = parser.parse_args()

n = args.nodes
graph = nx.random_regular_graph(3, n, seed=7)
cost_h, mixer_h = qp.qaoa.maxcut(graph)  # <cost_h> is minus the expected cut size
backend = "qoro" if args.cloud else "local"


def cut_size(bits):
    return sum(bits[i] != bits[j] for i, j in graph.edges)


def qaoa(params):
    for w in range(n):
        qp.Hadamard(wires=w)
    for gamma, alpha in params:
        qp.qaoa.cost_layer(gamma, cost_h)
        qp.qaoa.mixer_layer(alpha, mixer_h)


@qp.qnode(qp.device("maestro.qubit", wires=n, backend=backend))
def cost(params):
    qaoa(params)
    return qp.expval(cost_h)


@qp.qnode(qp.device("maestro.qubit", wires=n, shots=2000, backend=backend))
def sample(params):
    qaoa(params)
    return qp.counts()


params = pnp.array([[0.1, 0.1]] * args.depth, requires_grad=True)  # (gamma, alpha) per layer
opt = qp.AdamOptimizer(stepsize=0.05)
for step in range(args.steps):
    params, value = opt.step_and_cost(cost, params)
    if step % 10 == 0:
        print(f"step {step:3d}: expected cut {-value:.3f}")

best = max(cut_size(bits) for bits in itertools.product((0, 1), repeat=n))
print(f"\n{graph.number_of_edges()} edges; trained expected cut {-cost(params):.3f}; best cut {best}")
counts = sample(params)
for bits, count in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
    size = cut_size([int(b) for b in bits])
    print(f"  {bits}  sampled {int(count):4d}x  cut {size}{'  (optimal)' if size == best else ''}")
