from openpyrolysis.pykinetics import compute_alpha_cpp

def main():
    T_K = 773.15  # 500°C
    time_s = 1800  # 30 min
    A = 1e13       # Pre-exponential factor
    Ea = 200e3     # Activation energy in J/mol

    alpha = compute_alpha_cpp(T_K, time_s, A, Ea)
    print(f"Reaction fraction (alpha): {alpha:.4f}")

if __name__ == "__main__":
    main()
