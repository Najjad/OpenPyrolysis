from openpyrolysis.pykinetics import compute_alpha_cpp

def reaction_fraction(temp_c, time_s, A, Ea):
    return compute_alpha_cpp(temp_c + 273.15, time_s, A, Ea)
