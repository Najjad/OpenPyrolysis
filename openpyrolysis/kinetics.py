import math

def reaction_fraction(T_K, delta_t_s, A, Ea, alpha_current):
    R = 8.314
    k = A * math.exp(-Ea / (R * T_K))
    delta_alpha = (1.0 - alpha_current) * (1.0 - math.exp(-k * delta_t_s))
    return delta_alpha
