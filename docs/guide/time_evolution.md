# Time evolution

Time-evolution algorithms often measure expectation values at intermediate steps. Re-running from t = 0 for each step costs O(N²) over N steps; Maestro keeps one persistent simulator state across steps instead, for O(N).

When a circuit repeats a Trotter step with `qp.Snapshot`s in between, `maestro.qubit` detects the pattern and uses this automatically:

```python
dev = qp.device("maestro.qubit", wires=2)

@qp.qnode(dev)
def trotter_circuit():
    qp.PauliX(0)
    for step in range(1, 4):
        qp.IsingXX(0.2, wires=[0, 1])
        qp.Snapshot(f"step_{step}", measurement=qp.expval(qp.PauliZ(0)))
    return qp.expval(qp.PauliZ(0))

results = qp.snapshots(trotter_circuit)()
# results contains "step_1", "step_2", "step_3", and "execution_results"
```

For large workloads (e.g. 100+ steps on MPS), `incremental_evolve` skips building tapes entirely:

```python
dev = qp.device("maestro.qubit", wires=2, simulation_type="MatrixProductState")

res = dev.incremental_evolve(
    init=lambda: qp.PauliX(0),
    trotter_step=lambda: qp.IsingXX(0.2, wires=[0, 1]),
    measure_at_steps=[10, 20, 30],
    observables=[qp.PauliZ(0), qp.PauliZ(1)],
)
# res["expectation_values"] has shape (len(measure_at_steps), len(observables))
```

[The Trotter time evolution example](../examples/trotter_time_evolution.md) does it both ways.
