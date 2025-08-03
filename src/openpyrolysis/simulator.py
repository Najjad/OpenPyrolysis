import json
import os
import math

def load_plastic_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "plastics.json")
    with open(data_path, "r") as f:
        return json.load(f)

R = 8.314 

def reaction_fraction(T_C, time_s, A, Ea):
    T_K = T_C + 273.15
    k = A * math.exp(-Ea / (R * T_K))
    alpha = 1 - math.exp(-k * time_s)
    return alpha

def simulate_pyrolysis(
    plastic_type: str,
    mass_input_kg: float,
    reactor_temp_C: float,
    residence_time_min: float,
    heating_method: str = "electric"
):
    plastic_db = load_plastic_data()

    if plastic_type not in plastic_db:
        raise ValueError(f"Unsupported plastic type: {plastic_type}")

    props = plastic_db[plastic_type]

    time_s = residence_time_min * 60
    A = props.get("pre_exponential_factor", 1e13)
    Ea = props["activation_energy"]

    alpha = reaction_fraction(reactor_temp_C, time_s, A, Ea)

    delta_T = reactor_temp_C - 25
    Q_kJ = mass_input_kg * props["specific_heat"] * delta_T

    gas_yield = mass_input_kg * props["yield_distribution"]["gas"] * alpha
    oil_yield = mass_input_kg * props["yield_distribution"]["oil"] * alpha
    char_yield = mass_input_kg * props["yield_distribution"]["char"] * (1 - alpha)

    return {
        "plastic": plastic_type,
        "input_mass_kg": mass_input_kg,
        "reactor_temp_C": reactor_temp_C,
        "residence_time_min": residence_time_min,
        "heating_method": heating_method,
        "estimated_energy_kJ": Q_kJ,
        "reaction_fraction": alpha,
        "yields_kg": {
            "gas": gas_yield,
            "oil": oil_yield,
            "char": char_yield,
        },
    }
