#include "simulation.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_phik_simulation_core, m) { bind_simulation(m); }
