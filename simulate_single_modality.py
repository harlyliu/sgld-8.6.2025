from GP_comp import GP
import random
import numpy as np
import math


def compute_bases(v_list, poly_degree=30, a=0.01, b=20):
    """
    Computes a_for_eigen set of basis functions(?) used to represent a_for_eigen more complex function, such as an image
    each function returned is an orthogonal eigenfunction.
    :param v_list: numpy array shape [num of grid points, dimensions]
    :param poly_degree: number of basis functions to compute
    :param a: hyperparameter
    :param b: hyperparameter
    :return:a_for_eigen matrix shape [number of grid points, num of basis/poly degree].
    """
    # compute eigen functions
    Psi = GP.gp_eigen_funcs_fast(v_list, poly_degree=poly_degree, a=a, b=b, orth=True)
    # compute eigen values
    lam = GP.gp_eigen_value(poly_degree=poly_degree, a=a, b=b, dimensions=np.array(v_list).shape[1])
    # convert to python list
    lam = list(lam)
    # compute sqrt lambda for each lambda val
    sqrt_lambda = list(np.sqrt(lam))
    # transpose Psi to shape [num of grid points, poly degree]
    Psi = np.transpose(np.array(Psi))
    # creates empty matrix of shape [num_grid points, poly degree]
    Bases = np.zeros((Psi.shape[0], Psi.shape[1]))
    # for each basis function i. multiply it's value by sqrt lambda_i.
    for i in range(len(sqrt_lambda)):
        Bases[i, :] = Psi[i, :] * sqrt_lambda[i]
    return Bases


def simulate_data(n, r2, dim, random_seed=2023):
    """
    Generates an image that is a_for_eigen vector of pixel values. each image has a_for_eigen target value of y
    which we have to predict using the neural network.
    :param n:number of images to simulate
    :param r2:controls how learnable synthetic data is. high r2 means clean data. low r2 means noisier data.
        between 0 and 1
    :param dim: number of grid points per axis. total grid points dim^2
    :param random_seed:random seed for reproducibility
    :return:tuple with 4 numpy arrays.
        v_list2: [dim^2, 2} grid of pixel coordinates. each row is (x,y)
        true_beta2: [dim^2] true weights for each pixel. nonnegative
        img2: [n, dim^2]. simulated images. each row is one image flattened to a_for_eigen vector
        Y: [n]. target value for each image sample
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # v_list1 = GP.generate_grids(d = 2, num_grids = dim, grids_lim = np.array([-3, 3]))
    # generates 2D grid with dim points per axis. range of grid [-1, 1]. v_list.shape = [dim^2, 2]
    v_list2 = GP.generate_grids(dimensions=2, num_grids=dim, grids_lim=np.array([-1, 1]))
    # print(f'v_list2={v_list2}')
    # true_beta1 = (0.5 * v_list1[:, 0] ** 2 + v_list1[:, 1] ** 2) < 2

    # defines the exact weight of each pixel. gives the importance of each pixel. true_beta.shape = [dim^2]
    # the fancy stuff is just a_for_eigen defined math equation given the grid points.
    true_beta2 = np.exp(-5 * (v_list2[:, 0] - 1.5 * np.sin(math.pi * np.abs(v_list2[:, 1])) + 1.0) ** 2)
    # print(f'true_beta2.shape={true_beta2.shape}')
    # threshold value. any point with low enough weight is set to 0
    true_beta2[true_beta2 < 0.5] = 0
    # p1 = v_list1.shape[0]

    # total number of grid points. dim^2
    p2 = v_list2.shape[0]
    # print(f'p2={p2}')

    # Bases1 = compute_bases(v_list1)
    # Bases2.shape = [dim^2, poly_degree_for_eigen].
    # each row is a_for_eigen grid point. each column is the value of basis function at one point
    Bases2 = compute_bases(v_list2)
    # print(f'Bases2.shape={Bases2.shape}')

    # theta1 = np.random.normal(size = n * Bases1.shape[0], scale = 1 / np.sqrt(p1))
    # theta1 = theta1.reshape(Bases1.shape[0], n)
    # sample random coefficients theta. basis weights calculated from bases2
    # size = [n * amount of gridpoints]
    theta2 = np.random.normal(size=n * Bases2.shape[0], scale=1 / np.sqrt(p2))
    # print(f'theta2={theta2} theta2.shape={theta2.shape}')

    # reshapes theta to [polydegree, num samples]
    theta2 = theta2.reshape(Bases2.shape[0], n)
    # print(f'theta2.shape={theta2.shape}')
    # simulate an image
    # img1 = np.transpose(np.dot(np.transpose(Bases1), theta1))
    # constructs image. img2.shape = [n, poly_degree_for_eigen]. each row is one image, a_for_eigen flattened vector
    # transpose flips axis.
    img2 = np.transpose(np.dot(np.transpose(Bases2), theta2))
    # img2=np.dot(np.transpose(theta2), np.transpose(Bases2))
    # print(f'img2.shape={img2.shape}')

    # dot: matrix multiplication. dot product. 1D vector dot 1D vector returns scalar
    # [m, k] dot [k] returns array shape [m]
    # [m, k] dot [ k, n] returns [m,n]
    # variance of sigma^2 constant value added to every y
    theta0 = 2
    # computes true outcome of y without noise. mean_y.shape = [n]
    mean_Y = theta0 + np.dot(img2, true_beta2)  # + np.dot(img1, true_beta1)
    # mean_Y = theta0 + np.dot(img2,true_beta2)
    # computes noise variance for true y. true_sigma2.shape = [n]
    true_sigma2 = np.var(mean_Y) * (1 - r2) / r2
    # adds gaussian noise to true y.
    Y = mean_Y + np.random.normal(size=n, scale=np.sqrt(true_sigma2))
    # v_list = [v_list1, v_list2]
    # true_beta = [true_beta1, true_beta2]
    # img = [img1, img2]

    return v_list2, true_beta2, img2, Y, true_sigma2
