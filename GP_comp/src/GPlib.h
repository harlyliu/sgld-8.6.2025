#ifndef GPLIB_H
#define GPLIB_H

#ifdef __cplusplus
extern "C" {
#endif

double* GP_eigen_funcs(double* grids, int grids_size, int dim, int poly_degree, double a, double b, int& num_funcs);
void R_GP_eigen_funcs_orth(double* eigen_funcs, double* grids, int grids_size, int dim, int poly_degree, double a, double b);

#ifdef __cplusplus
}
#endif

#endif // GPLIB_H
