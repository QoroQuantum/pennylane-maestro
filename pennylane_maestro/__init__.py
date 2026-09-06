"""PennyLane plugin for the Maestro quantum simulator by Qoro Quantum."""

from pennylane_maestro.maestro_device import MaestroQubitDevice


def incremental_evolve(dev, init, trotter_step, measure_at_steps, observables):
    """Convenience wrapper for ``dev.incremental_evolve(...)``."""
    return dev.incremental_evolve(init, trotter_step, measure_at_steps, observables)


__version__ = "0.4.0"
__all__ = ["MaestroQubitDevice", "incremental_evolve"]
