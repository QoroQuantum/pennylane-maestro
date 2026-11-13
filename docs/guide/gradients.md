# Gradients

On the CPU statevector (`simulator_type="QCSim"`, `simulation_type="Statevector"`, the defaults), analytic expectation values of Pauli strings, tensor products and Hamiltonians/Sums, and `qp.probs`, are differentiated with the adjoint method. `diff_method="best"` (the QNode default) resolves to `"adjoint"`:

```python
dev = qp.device("maestro.qubit", wires=8)

@qp.qnode(dev)                       # diff_method="best" -> "adjoint"
def circuit(weights, x):
    qp.AngleEmbedding(x, wires=range(8))
    qp.StronglyEntanglingLayers(weights, wires=range(8))
    return qp.expval(qp.Z(0))
```

- With a qoro-maestro build that includes `AdjointProgram`, the adjoint sweep runs in the engine, restricted to each observable's causal cone. A broadcast batch of inputs (e.g. `x` of shape `(32, 8)`) is evaluated and differentiated in one engine call. This includes trainable broadcast parameters, as in data re-uploading (`qp.Rot(w * x + b, ...)`): `diff_method="best"` then uses a device VJP, which returns one gradient entry per batch element. With `device_vjp=False` set explicitly, with `jax.jit`, or with torch when only some trainable parameters are broadcast, such a batch is split into one circuit per element.
- `qp.probs` needs the engine path. A Jacobian costs one backward sweep per outcome, so it is used for up to 10 wires with the default `device_vjp=False` and up to 24 wires with `device_vjp=True` (one sweep in total); wider probabilities use parameter-shift.
- The inner-product fallback (`adjoint_backend="inner_product"`) is also available, evaluating one `maestro.inner_product` per gate parameter and Pauli term (expectation values only).
- Finite shots, mid-circuit measurements, other simulation types (MPS, stabilizer, ...) and other measurements (`var`, `state`, samples, `probs` beyond the wire limits above) use parameter-shift. `diff_method="adjoint"` raises for these.
- With PennyLane's default `device_vjp=False`, the Jacobian is computed together with the forward pass. This is the fastest choice for training, but plain forward calls with trainable (`requires_grad=True`) parameters also pay for the Jacobian. For inference-heavy use, pass `device_vjp=True` to the QNode: forward calls then only simulate, and the gradient is a separate VJP pass (on a 20-qubit, 32-sample VQC: forward 0.46 s instead of 1.73 s, loss+grad 2.55 s instead of 1.82 s).
