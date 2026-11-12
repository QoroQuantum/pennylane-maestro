"""PennyLane plugin for the Maestro quantum simulator by Qoro Quantum."""

from importlib.metadata import PackageNotFoundError, version

from pennylane_maestro.maestro_device import MaestroQubitDevice


def incremental_evolve(dev, init, trotter_step, measure_at_steps, observables):
    """Convenience wrapper for ``dev.incremental_evolve(...)``."""
    return dev.incremental_evolve(init, trotter_step, measure_at_steps, observables)


# The version is set only in pyproject.toml.
try:
    __version__ = version("pennylane-maestro")
except PackageNotFoundError:  # imported from a checkout that isn't installed
    __version__ = "0.0.0"

__all__ = ["MaestroQubitDevice", "incremental_evolve"]
