class Material:
    def __init__(self, name, specific_heat, activation_energy, pre_exp_factor, yield_distribution):
        self.name = name
        self.specific_heat = specific_heat
        self.activation_energy = activation_energy
        self.pre_exp_factor = pre_exp_factor
        self.yield_distribution = yield_distribution

    @staticmethod
    def from_json(json_data):
        return Material(**json_data)
