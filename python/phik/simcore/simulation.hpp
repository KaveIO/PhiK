/* python/phik/simulation/simulation.hpp wrapper and bindings for
 * Michael Patefield,
 * Algorithm AS 159: An Efficient Method of Generating RXC Tables with Given Row and Column Totals,
 * Applied Statistics,
 * Volume 30, Number 1, 1981, pages 91-97.
 *
 * https://people.sc.fsu.edu/~jburkardt/cpp_src/asa159/asa159.html
 */

#ifndef PYTHON_PHIK_SIMCORE_SIMULATION_HPP_
#define PYTHON_PHIK_SIMCORE_SIMULATION_HPP_
#include "asa159.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct simulation_error: std::exception {
    const char* p_message;
    explicit simulation_error(const char* message) : p_message(message) {}
    const char* what() const throw() { return p_message; }
};

void _sim_2d_data_patefield(
    int nrow,
    int ncol,
    const py::array_t<int>& nrowt,
    const py::array_t<int>& ncolt,
    int seed,
    py::array_t<int>& result
) {
    bool key = false;
    int ierror = 0;
    int* nrowt_ptr = reinterpret_cast<int*>(nrowt.request().ptr);
    int* ncolt_ptr = reinterpret_cast<int*>(ncolt.request().ptr);
    int* result_ptr = reinterpret_cast<int*>(result.request().ptr);

    // constructs a random two-way contingency table with given sums,
    // the underlying memory of result is directly modified
    rcont2(nrow, ncol, nrowt_ptr, ncolt_ptr, &key, &seed, result_ptr, &ierror);
    if (ierror != 0) {
        throw simulation_error("Could not construct two-way contingency table");
    }
    return;
}

auto docstring = R"pbdoc(Construct a random two-way contingency table with given sums

Parameters
----------
nrow : int
    number of rows in the table, should be >= 2
ncol : int
    number of columns in the table, should be >= 2
nrowt : np.array[int]
    the row sums, note all values should be positive
ncolt : np.array[int]
    the col sums, note all values should be positive
seed : int
    random seed for the generation
result : np.array[int]
    initialized array where the results will be stored

Reference
---------
WM Patefield,
Algorithm AS 159:
An Efficient Method of Generating RXC Tables with
Given Row and Column Totals,
Applied Statistics,
Volume 30, Number 1, 1981, pages 91-97.
)pbdoc";

void bind_simulation(py::module &m) {
    m.def(
        "_sim_2d_data_patefield",
        &_sim_2d_data_patefield,
        docstring,
        py::arg("nrow"),
        py::arg("ncol"),
        py::arg("nrowt"),
        py::arg("ncolt"),
        py::arg("seed"),
        py::arg("result")
    );
}

#endif  // PYTHON_PHIK_SIMCORE_SIMULATION_HPP_
