import pytest
import pennylane as qml

@pytest.fixture
def tol():
    """Numerical tolerance for float comparisons."""
    return {"atol": 1e-6, "rtol": 1e-5}

@pytest.fixture(params=[None, 10000])
def shots(request):
    """Fixture to run tests with both analytic and finite shot execution."""
    return request.param
