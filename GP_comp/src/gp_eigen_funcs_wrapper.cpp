#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "GPlib.h"  // Include the updated header

namespace py = pybind11;

py::array_t<double> gp_eigen_funcs_fast(py::array_t<double> grids, int poly_degree = 10, double a = 0.01, double b = 1.0) {
    auto buf = grids.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int grids_size = static_cast<int>(buf.shape[0]);
    int dim = static_cast<int>(buf.shape[1]);

    int num_funcs;
    double* eigen_funcs = GP_eigen_funcs(ptr, grids_size, dim, poly_degree, a, b, num_funcs);

    py::array_t<double> result({grids_size, num_funcs});
    std::memcpy(result.mutable_data(), eigen_funcs, grids_size * num_funcs * sizeof(double));

    delete[] eigen_funcs;
    return result;
}

// Wrapper for the orthogonal GP eigenfunctions
py::array_t<double> gp_eigen_funcs_fast_orth(py::array_t<double> grids, int poly_degree = 10, double a = 0.01, double b = 1.0) {
    auto buf = grids.request();
    double* ptr = static_cast<double*>(buf.ptr);
    int grids_size = static_cast<int>(buf.shape[0]);
    int dim = static_cast<int>(buf.shape[1]);

    int num_funcs = (poly_degree + dim) / dim;  // Approximate based on poly_degree
    py::array_t<double> eigen_funcs({grids_size, num_funcs});
    
    R_GP_eigen_funcs_orth(eigen_funcs.mutable_data(), ptr, grids_size, dim, poly_degree, a, b);
    
    return eigen_funcs;
}

PYBIND11_MODULE(GPlib_py, m) {
    m.def("gp_eigen_funcs_fast", &gp_eigen_funcs_fast, "Compute Gaussian process eigenfunctions");
    m.def("gp_eigen_funcs_fast_orth", &gp_eigen_funcs_fast_orth, "Compute orthogonal Gaussian process eigenfunctions");
}
