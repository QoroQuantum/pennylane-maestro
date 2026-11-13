# Examples

Seven scripts, from the repository's
[examples/](https://github.com/QoroQuantum/pennylane-maestro/tree/main/examples)
folder. Run each from the repository root, e.g.
`python examples/hamiltonian_vqe.py`. Scripts with a `--cloud` flag run the same
code on the Qoro cloud: see [Run on the Qoro cloud](../index.md#run-on-the-qoro-cloud)
to set up an API key.

| Example | What it shows |
|---|---|
| [Running on the Qoro cloud](qoro_cloud.md) | Training a circuit and sampling a 50-qubit state on the Qoro cloud |
| [VQE for an Ising chain](hamiltonian_vqe.md) | VQE for a 20-qubit Ising chain with adjoint gradients, checked against the exact energy (`--cloud`) |
| [QAOA for MaxCut](qaoa_maxcut.md) | QAOA for MaxCut, trained with adjoint gradients and sampled for the best cut (`--cloud`) |
| [A data re-uploading classifier](quantum_classifier.md) | A classifier trained on a broadcast batch in one engine call (`--cloud`) |
| [A 100-qubit Ising quench](ising_phase_transition.md) | A quench with MPS shot sampling (needs `matplotlib`, `scipy`) |
| [Trotter time evolution](trotter_time_evolution.md) | Persistent-state Trotter evolution with `qp.Snapshot` and `incremental_evolve` |
| [Benchmarks](benchmark_lightning_vs_maestro.md) | The benchmark suite behind the [performance tables](../guide/performance.md) (needs `pennylane-lightning`, `quimb`) |

```{toctree}
:hidden:

qoro_cloud
hamiltonian_vqe
qaoa_maxcut
quantum_classifier
ising_phase_transition
trotter_time_evolution
benchmark_lightning_vs_maestro
```
