import math

def reaction_fraction(T_K, delta_t_s, A, Ea, alpha_current):
    R = 8.314
    k = A * math.exp(-Ea / (R * T_K))
    delta_alpha = (1.0 - alpha_current) * (1.0 - math.exp(-k * delta_t_s))
    return delta_alpha

def pressure_modifier(P_atm, P_ref=1.0, n=0.5):
    """
    Simple empirical modifier for pressure effect on rate constant.
    P_atm: current pressure (atm)
    P_ref: reference pressure (atm)
    n: pressure exponent (empirical)
    """
    return (P_atm / P_ref) ** n

def temperature_at_time(t_s, temperature_profile):
    """
    temperature_profile: list of (time_s, T_K) tuples sorted by time.
    Returns interpolated temperature at time t_s.
    """
    if not temperature_profile:
        raise ValueError("Temperature profile is empty")

    for i, (time_point, temp) in enumerate(temperature_profile):
        if t_s < time_point:
            if i == 0:
                return temp
            t0, T0 = temperature_profile[i-1]
            t1, T1 = time_point, temp
            # Linear interpolation
            return T0 + (T1 - T0) * (t_s - t0) / (t1 - t0)
    return temperature_profile[-1][1]  # after last point

def multi_step_reaction_fraction(t_s, alpha_vec, steps, T_profile=None, P_atm=1.0):
    """
    Calculates incremental alpha changes for multi-step kinetics.

    t_s: current time in seconds
    alpha_vec: list of current alpha values for each reaction step
    steps: list of dicts for each step with keys:
        - 'A': pre-exponential factor (1/s)
        - 'Ea': activation energy (J/mol)
        - 'yield_func': function(alpha, T_K) -> yield fraction (0-1) for this step's product
    T_profile: temperature profile as list of (time_s, T_K) tuples, or None for constant T
    P_atm: pressure in atm

    Returns:
        delta_alpha_vec: list of incremental alpha changes per step
        total_alpha: combined overall conversion (0-1)
    """

    R = 8.314
    delta_alpha_vec = []
    total_alpha = 0.0

    # Get current temperature from profile or use fixed
    if T_profile:
        T_K = temperature_at_time(t_s, T_profile)
    else:
        # If no profile, use first step's T or assume constant (should be passed)
        T_K = steps[0].get('T_K', 773.15)  # default 500°C if not specified

    for i, step in enumerate(steps):
        A = step['A']
        Ea = step['Ea']

        # Apply pressure modifier on rate constant
        k = A * math.exp(-Ea / (R * T_K)) * pressure_modifier(P_atm)

        alpha_current = alpha_vec[i]

        delta_alpha = (1.0 - alpha_current) * (1.0 - math.exp(-k * t_s))
        delta_alpha_vec.append(delta_alpha)
        total_alpha += delta_alpha  # simplistic additive, adjust if needed

    # Clamp total alpha to max 1.0
    total_alpha = min(total_alpha, 1.0)

    return delta_alpha_vec, total_alpha


