import pytest
from openpyrolysis.simulator import simulate_pyrolysis

def test_simulation_outputs():
    result = simulate_pyrolysis("HDPE", 10, 450, 30)
    assert abs(result["yields_kg"]["gas"] - 3.5) < 0.01
    assert abs(result["yields_kg"]["oil"] - 5.5) < 0.01
    assert abs(result["yields_kg"]["char"] - 1.0) < 0.01
    assert result["estimated_energy_kJ"] > 0

def test_invalid_plastic():
    with pytest.raises(ValueError):
        simulate_pyrolysis("PVC", 5, 400, 20)