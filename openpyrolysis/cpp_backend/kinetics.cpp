// kinetics.cpp

#include <cmath>

extern "C" double compute_alpha(double T_K, double time_s, double A, double Ea) {
    const double R = 8.314;
    double k = A * std::exp(-Ea / (R * T_K));
    double alpha = 1.0 - std::exp(-k * time_s);
    return alpha;
}
