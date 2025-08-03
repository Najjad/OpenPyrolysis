import json
import os

def load_plastic_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "plastics.json")
    with open(data_path, "r") as f:
        return json.load(f)

def simulate_pyrolysis(
    plastic_type: str,
    mass_input_kg: float,
    reactor_temp_C: float,
    residence_time_min: float,
    heating_method: str = "electric"
):
    """
    Simulates pyrolysis of plastic with basic yield estimation.
    Returns dict with gas, oil, char yields and energy required.
    """

    plastic_db = load_plastic_data()

    if plastic_type not in plastic_db:
        raise ValueError(f"Unsupported plastic type: {plastic_type}")

    props = plastic_db[plastic_type]

    # Simplified energy estimate (Q = mcΔT)
    delta_T = reactor_temp_C - 25  # assuming room temp start
    Q_kJ = mass_input_kg * props["specific_heat"] * delta_T

    # Yields (fixed ratio per plastic type)
    gas_yield = mass_input_kg * props["yield_distribution"]["gas"]
    oil_yield = mass_input_kg * props["yield_distribution"]["oil"]
    char_yield = mass_input_kg * props["yield_distribution"]["char"]

    return {
        "plastic": plastic_type,
        "input_mass_kg": mass_input_kg,
        "reactor_temp_C": reactor_temp_C,
        "residence_time_min": residence_time_min,
        "heating_method": heating_method,
        "estimated_energy_kJ": Q_kJ,
        "yields_kg": {
            "gas": gas_yield,
            "oil": oil_yield,
            "char": char_yield,
        },
    }
