from openpyrolysis.simulator import simulate_pyrolysis

def run_cli():
    print("=== OpenPyrolysis MVP Simulator ===")
    plastic = input("Enter plastic type (HDPE, LDPE, PP): ").strip().upper()
    mass = float(input("Enter input mass (kg): "))
    temp = float(input("Enter reactor temperature (°C): "))
    time = float(input("Enter residence time (minutes): "))
    method = input("Enter heating method (electric, gas, microwave): ").strip().lower()

    result = simulate_pyrolysis(plastic, mass, temp, time, method)

    print("\n--- Simulation Results ---")
    print(f"Plastic type: {result['plastic']}")
    print(f"Input mass: {result['input_mass_kg']} kg")
    print(f"Reactor temperature: {result['reactor_temp_C']} °C")
    print(f"Residence time: {result['residence_time_min']} minutes")
    print(f"Heating method: {result['heating_method']}")
    print(f"Estimated energy required: {result['estimated_energy_kJ']:.2f} kJ")
    print("Yields (kg):")
    for k, v in result['yields_kg'].items():
        print(f"  {k.capitalize()}: {v:.2f} kg")

