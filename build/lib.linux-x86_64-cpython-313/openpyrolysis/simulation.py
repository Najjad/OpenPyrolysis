from .kinetics import reaction_fraction

class PyrolysisSimulator:
    def __init__(self, material, reactor):
        self.material = material
        self.reactor = reactor

    def run(self, mass_input_kg):
        alpha = reaction_fraction(
            self.reactor.temperature_c,
            self.reactor.residence_time_min * 60,
            self.material.pre_exp_factor,
            self.material.activation_energy,
        )

        delta_t = self.reactor.temperature_c - 25
        Q_kj = mass_input_kg * self.material.specific_heat * delta_t

        extra_energy_kj = (
            self.reactor.heating_power_kw *
            (self.reactor.residence_time_min / 60) *
            3600 * self.reactor.efficiency
        )
        total_energy = Q_kj + extra_energy_kj

        return {
            "alpha": alpha,
            "energy_kj": total_energy,
            "yields": {
                "gas": mass_input_kg * self.material.yield_distribution['gas'] * alpha,
                "oil": mass_input_kg * self.material.yield_distribution['oil'] * alpha,
                "char": mass_input_kg * self.material.yield_distribution['char'] * (1 - alpha),
            }
        }
