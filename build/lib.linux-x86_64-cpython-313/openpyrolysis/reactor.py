class Reactor:
    def __init__(self, temperature_c, residence_time_min, heating_method, heating_power_kw, efficiency):
        self.temperature_c = temperature_c
        self.residence_time_min = residence_time_min
        self.heating_method = heating_method
        self.heating_power_kw = heating_power_kw
        self.efficiency = efficiency
