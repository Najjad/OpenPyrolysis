// bindings.cpp
#include <pybind11/pybind11.h>

// Declare the C function (not define it here)
extern "C" double compute_alpha(double T_K, double time_s, double A, double Ea);

namespace py = pybind11;

PYBIND11_MODULE(pykinetics, m) {
    m.def("compute_alpha_cpp", &compute_alpha, "Calculate reaction fraction using C++");
}
