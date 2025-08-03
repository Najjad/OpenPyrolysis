import numpy as np
from scipy.integrate import solve_ivp

def pyrolysis_kinetics(t, y, k):
    A = y[0]
    dA_dt = -k * A
    return [dA_dt]

def run_batch_reactor(A0=1.0, k=0.1, t_span=(0, 100)):
    sol = solve_ivp(pyrolysis_kinetics, t_span, [A0], args=(k,), dense_output=True)
    return sol
