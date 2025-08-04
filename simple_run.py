from openpyrolysis.simulation import PyrolysisSimulator
from openpyrolysis.materials import Material
from openpyrolysis.reactor import Reactor

def main():
    # Define or import a sample material and reactor
    material = Material(
        name="HDPE",
        specific_heat=2.3,  # kJ/kg·°C
        activation_energy=200000,  # J/mol
        pre_exp_factor=1e13,  # 1/s
        yield_distribution={"gas": 0.3, "oil": 0.6, "char": 0.1}
    )

    reactor = Reactor(
        temperature_c=400,  # °C
        residence_time_min=11,  # minutes
        heating_method="electrical",
        heating_power_kw=5.0,
        efficiency=0.85
    )

    sim = PyrolysisSimulator(material, reactor)
    results = sim.run(mass_input_kg=5)

    print("\nFinal Results:")
    print(f"Gas yield:  {results['final_yields']['gas']:.2f} kg")
    print(f"Oil yield:  {results['final_yields']['oil']:.2f} kg")
    print(f"Char yield: {results['final_yields']['char']:.2f} kg")
    print(f"Total energy used: {results['energy_kj']:.2f} kJ")

if __name__ == "__main__":
    main()
