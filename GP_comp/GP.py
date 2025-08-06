import numpy as np
from scipy.special import comb
from itertools import product
import GPlib

# https://github.com/kangjian2016/BayesGPfit/blob/master/R/fastGPfit.R


def generate_grids(dimensions=1, num_grids=50, grids_lim=(-1, 1), random=False):
    # Define the base grids based on random or sequential generation
    if random:
        base_grids = np.random.rand(num_grids) * (grids_lim[1] - grids_lim[0]) + grids_lim[0]
    else:
        base_grids = np.linspace(grids_lim[0], grids_lim[1], num_grids)

    # Create list of grids for each dimension
    grids_list = [base_grids] * dimensions

    # Generate all combinations using product and convert to tensor
    grids = np.array(list(product(*grids_list)), dtype=np.float64)

    # Reshape to match R's expand.grid output
    grids = np.flip(grids, axis=1)

    return grids


def gp_eigen_value(poly_degree=10, a=1, b=1, dimensions=2):
    # Calculate constants
    cn = np.sqrt(a ** 2 + 2 * a * b)
    A = a + b + cn
    B = b / A

    # Create index array (equivalent to R's c() and choose())
    idx = np.array([0] + [comb(i + dimensions, dimensions, exact=True) for i in range(poly_degree + 1)])

    # Generate idxlist using list comprehension (equivalent to R's sapply)
    idxlist = [list(range(idx[i] + 1, idx[i + 1] + 1)) for i in range(poly_degree + 1)]

    # Get length k from external function (assumed to be defined elsewhere)
    k = gp_num_eigen_funs(poly_degree=poly_degree, dimensions=dimensions)

    # Initialize value array with NaN
    value = np.full(k, np.nan)

    # Calculate dvalue
    dvalue = (np.sqrt(np.pi / A)) ** dimensions * B ** np.arange(1, poly_degree + 2)

    # Fill value array using idxlist
    for i in range(poly_degree + 1):
        # Convert to 0-based indexing for Python (R uses 1-based)
        indices = [x - 1 for x in idxlist[i]]
        value[indices] = dvalue[i]

    return value


def gp_num_eigen_funs(poly_degree=10, dimensions=2):
    return int(comb(poly_degree + dimensions, dimensions))


def gp_eigen_funcs_fast(grids, poly_degree=10, a=0.01, b=1.0, orth=False):
    num_funcs = gp_num_eigen_funs(poly_degree, grids.shape[1])
    eigen_funcs = np.zeros(num_funcs * grids.shape[0], dtype=np.float64)
    grids_size = grids.shape[0]
    d = grids.shape[1]
    grids = grids.reshape(-1, order="F")
    if orth:
        res = GPlib.GP_eigen_funcs_orth(eigen_funcs, grids, int(grids_size), int(d), int(poly_degree), float(a),
                                        float(b))
    else:
        res = GPlib.GP_eigen_funcs(eigen_funcs, grids, int(grids_size), int(d), int(poly_degree), float(a), float(b))
    # return res.reshape((num_funcs, grids_size), order="F")
    return res.reshape((grids_size, num_funcs), order="F")


if __name__ == '__main__':
    # ans = generate_grids(d=2, num_grids=5, grids_lim=(-1, 1), random=False)
    ans = gp_num_eigen_funs(poly_degree=10, dimensions=2)
    # ans = gp_eigen_value()
    print(type(ans), len(ans))
