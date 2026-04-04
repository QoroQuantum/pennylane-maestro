import pytest
import pennylane as qml
import numpy as np

from pennylane_maestro.maestro_device import MaestroQubitDevice

def test_device_discovery():
    """Test that the device can be instantiated via PennyLane's plugin mechanism."""
    dev = qml.device("maestro.qubit", wires=2)
    assert isinstance(dev, MaestroQubitDevice)
    assert dev.name == "maestro.qubit"

def test_wire_handling():
    """Test device initialization with different wire configurations."""
    # Integer wires
    dev_int = qml.device("maestro.qubit", wires=4)
    assert len(dev_int.wires) == 4
    
    # List of string wires
    dev_list = qml.device("maestro.qubit", wires=["a", "b", "c"])
    assert len(dev_list.wires) == 3

def test_shots_configuration():
    """Test shots parameter is correctly passed."""
    dev_analytic = qml.device("maestro.qubit", wires=1, shots=None)
    assert dev_analytic.shots.total_shots is None
    
    dev_shots = qml.device("maestro.qubit", wires=1, shots=100)
    assert dev_shots.shots.total_shots == 100

def test_invalid_simulator_type():
    """Test validation of Maestro-specific enums."""
    with pytest.raises(ValueError, match="Unknown SimulatorType"):
        qml.device("maestro.qubit", wires=1, simulator_type="InvalidSim")

    with pytest.raises(ValueError, match="Unknown SimulationType"):
        qml.device("maestro.qubit", wires=1, simulation_type="InvalidMode")

def test_mps_args():
    """Test passing MPS specific arguments."""
    dev = qml.device(
        "maestro.qubit",
        wires=2,
        simulation_type="MatrixProductState",
        max_bond_dimension=4,
        singular_value_threshold=1e-5
    )
    assert dev._max_bond_dimension == 4
    assert dev._singular_value_threshold == 1e-5
