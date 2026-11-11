# Examples

Run each from the repository root, e.g. `python examples/hamiltonian_vqe.py`.
Scripts with a `--cloud` flag run the same code on the Qoro cloud; see the
[README](../README.md#run-on-the-qoro-cloud) to set up an API key.

| Example | What it shows |
|---|---|
| [`qoro_cloud.py`](qoro_cloud.py) | Training a circuit and sampling a 50-qubit state on the Qoro cloud |
| [`hamiltonian_vqe.py`](hamiltonian_vqe.py) | VQE for a 20-qubit Ising chain with adjoint gradients, checked against the exact energy (`--cloud`) |
| [`qaoa_maxcut.py`](qaoa_maxcut.py) | QAOA for MaxCut, trained with adjoint gradients and sampled for the best cut (`--cloud`) |
| [`quantum_classifier.py`](quantum_classifier.py) | A data re-uploading classifier trained on a broadcast batch in one engine call (`--cloud`) |
| [`ising_phase_transition.py`](ising_phase_transition.py) | A 100-qubit Ising quench with MPS shot sampling (needs `matplotlib`, `scipy`) |
| [`trotter_time_evolution.py`](trotter_time_evolution.py) | Persistent-state Trotter evolution with `qp.Snapshot` and `incremental_evolve` |
| [`benchmark_lightning_vs_maestro.py`](benchmark_lightning_vs_maestro.py) | The benchmark suite behind the [performance tables](../docs/guide.md#performance) (needs `pennylane-lightning`, `quimb`) |
