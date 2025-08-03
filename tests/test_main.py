import pytest
from openpyrolysis.simulator import simulate_pyrolysis, reaction_fraction

def test_reaction_fraction():
    alpha = reaction_fraction(450, 1800, 1.5e13, 200000)
    assert 0 <= alpha <= 1

def test_simulation_outputs():
    result = simulate_pyrolysis("HDPE", 10, 450, 30)
    alpha = result["reaction_fraction"]
    assert 0 <= alpha <= 1

    expected_gas = 10 * 0.35 * alpha
    expected_oil = 10 * 0.55 * alpha
    expected_char = 10 * 0.10 * (1 - alpha)

    assert abs(result["yields_kg"]["gas"] - expected_gas) < 0.01
    assert abs(result["yields_kg"]["oil"] - expected_oil) < 0.01
    assert abs(result["yields_kg"]["char"] - expected_char) < 0.01
    assert result["estimated_energy_kJ"] > 0

def test_invalid_plastic():
    with pytest.raises(ValueError):
        simulate_pyrolysis("PVC", 5, 400, 20)
