from .kinetics import reaction_fraction  # incremental delta alpha function

class PyrolysisSimulator:
    def __init__(self, material, reactor):
        self.material = material
        self.reactor = reactor

    def run(self, mass_input_kg, reporting_timestep_min=1, verbose=True):
        total_time_min = self.reactor.residence_time_min
        temperature_K = self.reactor.temperature_c + 273.15

        # Small internal timestep for kinetics integration (seconds)
        internal_timestep_s = 1.0  # 1 second steps for accurate kinetics

        total_time_s = total_time_min * 60
        reporting_timestep_s = reporting_timestep_min * 60

        alpha = 0.0
        alpha_values = []
        time_points_min = []

        current_time_s = 0.0
        next_report_time_s = 0.0

        if verbose:
            print("\n[ Pyrolysis Simulation Started ]")
            print(f"Total simulation time: {total_time_min} minutes @ {self.reactor.temperature_c}°C")
            print(f"Reporting timestep: {reporting_timestep_min} min, internal integration timestep: {internal_timestep_s} sec")
            print(f"{'Time (min)':>12} | {'Alpha':>8} | {'Gas (kg)':>10} | {'Oil (kg)':>10} | {'Char (kg)':>10}")
            print("-" * 60)

        while current_time_s <= total_time_s:
            # Integrate kinetics with small internal timestep increments
            delta_alpha = reaction_fraction(
                temperature_K,
                internal_timestep_s,
                self.material.pre_exp_factor,
                self.material.activation_energy,
                alpha,
            )
            alpha += delta_alpha
            if alpha > 1.0:
                alpha = 1.0

            current_time_s += internal_timestep_s

            # Report values only at the requested reporting timestep intervals
            if current_time_s >= next_report_time_s or abs(current_time_s - total_time_s) < 1e-6:
                t_min = current_time_s / 60
                alpha_values.append(alpha)
                time_points_min.append(t_min)

                gas = mass_input_kg * self.material.yield_distribution['gas'] * alpha
                oil = mass_input_kg * self.material.yield_distribution['oil'] * alpha
                char = mass_input_kg * self.material.yield_distribution['char'] * (1 - alpha)

                if verbose:
                    print(f"{t_min:12.2f} | {alpha:8.4f} | {gas:10.3f} | {oil:10.3f} | {char:10.3f}")

                next_report_time_s += reporting_timestep_s

                # Save latest yields for return
                final_gas, final_oil, final_char = gas, oil, char

        # Energy calculations (same as before)
        delta_temp = self.reactor.temperature_c - 25
        Q_kj = (mass_input_kg * self.material.specific_heat * delta_temp) / 1000

        heating_energy_kj = (
            self.reactor.heating_power_kw *
            (total_time_min / 60) * 3600 * self.reactor.efficiency
        )
        total_energy_kj = Q_kj + heating_energy_kj

        if verbose:
            print("\n[ Simulation Complete ]")
            print(f"Final Alpha: {alpha:.4f}")
            print(f"Total Energy Used: {total_energy_kj:.2f} kJ\n")

        return {
            "time_min": time_points_min,
            "alpha_over_time": alpha_values,
            "energy_kj": total_energy_kj,
            "final_yields": {
                "gas": final_gas,
                "oil": final_oil,
                "char": final_char,
            }
        }
