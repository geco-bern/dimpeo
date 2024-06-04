# Reference: https://www.jmlr.org/papers/volume13/kim12b/kim12b.pdf
import math
import os

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import cosm1, i0e

os.environ["R_HOME"] = "/Users/davidbruggemann/miniconda3/envs/dimpeo/lib/R"
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

ks = importr("ks")


def wrapped_kernel(X, bandwidth, Y=None, type="von_mises", period_length=365):
    if type == "approx_wrapped_normal":
        # just first degree approximation
        if Y is None:
            X_dgs = X[:, 0]
            X_ndvi = X[:, 1]
            max_dist = math.ceil(period_length / bandwidth[0] / 2) ** 2
            dists_dgs = pdist(
                np.expand_dims(X_dgs / bandwidth[0], axis=1), metric="sqeuclidean"
            )
            dists_dgs[dists_dgs > max_dist] = (
                period_length / bandwidth[0] - np.sqrt(dists_dgs[dists_dgs > max_dist])
            ) ** 2
            dists_ndvi = pdist(
                np.expand_dims(X_ndvi / bandwidth[1], axis=1), metric="sqeuclidean"
            )
            dists = dists_dgs + dists_ndvi
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            X_dgs = X[:, 0]
            X_ndvi = X[:, 1]
            Y_dgs = Y[:, 0]
            Y_ndvi = Y[:, 1]
            max_dist = math.ceil(period_length / bandwidth[0] / 2) ** 2
            dists_dgs = cdist(
                np.expand_dims(X_dgs / bandwidth[0], axis=1),
                np.expand_dims(Y_dgs / bandwidth[0], axis=1),
                metric="sqeuclidean",
            )
            dists_dgs[dists_dgs > max_dist] = (
                period_length / bandwidth[0] - np.sqrt(dists_dgs[dists_dgs > max_dist])
            ) ** 2
            dists_ndvi = cdist(
                np.expand_dims(X_ndvi / bandwidth[1], axis=1),
                np.expand_dims(Y_ndvi / bandwidth[1], axis=1),
                metric="sqeuclidean",
            )
            dists = dists_dgs + dists_ndvi
            K = np.exp(-0.5 * dists).T
        K = (2 * np.pi) ** (-X.shape[1] / 2.0) * np.prod(bandwidth**2) ** (-1 / 2.0) * K
    elif type == "von_mises":
        if Y is None:
            kappa = 1 / (bandwidth[0] / period_length * 2 * np.pi) ** 2
            X_dgs = (X[:, 0] - 1) / period_length * 2 * np.pi
            X_ndvi = X[:, 1]
            dists_dgs = kappa * cosm1(
                pdist(np.expand_dims(X_dgs, axis=1), metric="minkowski", p=1)
            )
            dists_ndvi = -0.5 * pdist(
                np.expand_dims(X_ndvi / bandwidth[1], axis=1), metric="sqeuclidean"
            )
            dists = dists_dgs + dists_ndvi
            K = np.exp(dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
            K = (
                (2 * np.pi * bandwidth[1] ** 2) ** (-1 / 2.0)
                * K
                / (2 * np.pi * i0e(kappa))
            )
        else:
            kappa = 1 / (bandwidth[0] / period_length * 2 * np.pi) ** 2
            X_dgs = (X[:, 0] - 1) / period_length * 2 * np.pi
            Y_dgs = (Y[:, 0] - 1) / period_length * 2 * np.pi
            X_ndvi = X[:, 1]
            Y_ndvi = Y[:, 1]
            dists_dgs = kappa * cosm1(
                cdist(
                    np.expand_dims(X_dgs, axis=1),
                    np.expand_dims(Y_dgs, axis=1),
                    metric="minkowski",
                    p=1,
                )
            )
            dists_ndvi = -0.5 * cdist(
                np.expand_dims(X_ndvi / bandwidth[1], axis=1),
                np.expand_dims(Y_ndvi / bandwidth[1], axis=1),
                metric="sqeuclidean",
            )
            dists = dists_dgs + dists_ndvi
            K = np.exp(dists)
            K = (
                (2 * np.pi * bandwidth[1] ** 2) ** (-1 / 2.0)
                * K.T
                / (2 * np.pi * i0e(kappa))
            )
    return K


def gaussian_kernel(X, bandwidth, Y=None):
    # could use Mahalanobis distance for non-diagonal covariance matrix
    length_scale = np.expand_dims(bandwidth, axis=0)
    if Y is None:
        dists = pdist(X / length_scale, metric="sqeuclidean")
        K = np.exp(-0.5 * dists)
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)
    else:
        dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
        K = np.exp(-0.5 * dists).T
    return (2 * np.pi) ** (-X.shape[1] / 2.0) * np.prod(bandwidth**2) ** (-1 / 2.0) * K


def rho(x, typ="hampel", a=0, b=0, c=0):
    if typ == "huber":
        assert 0 < a
        dt = np.promote_types(x.dtype, "float")
        v = np.zeros(x.shape, dtype=dt)
        in1 = x <= a
        in2 = x > a
        v[in1] = x[in1] ** 2 / 2
        v[in2] = x[in2] * a - a**2 / 2
        return v
    if typ == "hampel":
        assert 0 < a <= b <= c
        dt = np.promote_types(x.dtype, "float")
        v = np.zeros(x.shape, dtype=dt)
        in1 = x <= a
        in2 = np.logical_and(a < x, x <= b)
        in3 = np.logical_and(b < x, x <= c)
        in4 = c < x
        v[in1] = x[in1] ** 2 / 2
        v[in2] = a * x[in2] - a**2 / 2
        v[in3] = (a * (x[in3] - c) ** 2 / (2 * (b - c))) + a * (b + c - a) / 2
        v[in4] = np.ones(x[in4].shape) * a * (b + c - a) / 2
        return v
    if typ == "abs":
        v = np.abs(x)
        return v


def loss(x, typ="hampel", a=0, b=0, c=0):
    return np.mean(rho(x, typ=typ, a=a, b=b, c=c))


def phi(x, typ="hampel", a=0, b=0, c=0):
    # defined as psi(x) / x
    if typ == "huber":
        assert 0 < a
        dt = np.promote_types(x.dtype, "float")
        v = np.zeros(x.shape, dtype=dt)
        in1 = x <= a
        in2 = a < x
        v[in1] = np.ones(x[in1].shape)
        v[in2] = a / x[in2]
        return v
    if typ == "hampel":
        assert 0 < a <= b <= c
        dt = np.promote_types(x.dtype, "float")
        v = np.zeros(x.shape, dtype=dt)
        in1 = x <= a
        in2 = np.logical_and(a < x, x <= b)
        in3 = np.logical_and(b < x, x <= c)
        in4 = c <= x
        v[in1] = np.ones(x[in1].shape)
        v[in2] = a / x[in2]
        v[in3] = a * (c - x[in3]) / (c - b) / x[in3]
        v[in4] = np.zeros(x[in4].shape)
        return v
    if typ == "abs":
        assert np.all(x != 0)  # undefined for x = 0
        return 1 / x


def compute_norm(Km, w):
    # as in section 4 of the paper
    t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
    t2 = -2 * np.dot(Km, w)
    t3 = np.dot(np.dot(w.T, Km), w)
    return np.sqrt(t1 + t2 + t3)


def kirwls(Km, type_rho, n, a, b, c, alpha=1e-8, max_it=100):
    """
    Kernelized iteratively re-weighted least-squares
    """
    # init weights
    w = np.ones((n, 1)) / n
    # first pass
    norm = compute_norm(Km, w)
    J = loss(norm, typ=type_rho, a=a, b=b, c=c)
    stop = 0
    count = 0
    while not stop:
        count += 1
        J_old = J
        # update w_i
        w = phi(norm, typ=type_rho, a=a, b=b, c=c)
        w = w / np.sum(w)
        # check loss
        norm = compute_norm(Km, w)
        J = loss(norm, typ=type_rho, a=a, b=b, c=c)
        if np.abs(J - J_old) < (J_old * alpha):
            stop = 1
        if count == max_it:
            print("Reached maximum of {} iterations without convergence".format(count))
            stop = 1
    return w, norm


def get_density_kde_rkde(ref, type_rho="hampel", periodic=False):

    with (ro.default_converter + numpy2ri.converter).context():

        r_ref = ro.conversion.get_conversion().py2rpy(ref)
        # only diagonal bandwidth possible / makes sense
        r_H = ks.Hpi_diag(r_ref)
        bw = ro.conversion.get_conversion().rpy2py(r_H)
        bw = np.sqrt(np.diag(bw))

    xx, yy = np.mgrid[1:366, 0:10000:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T

    # kernel matrix
    n_samples, d = ref.shape
    if periodic:
        Km = wrapped_kernel(ref, bandwidth=bw, type="von_mises")
    else:
        Km = gaussian_kernel(ref, bandwidth=bw)
    # find a, b, c via iterative reweighted least square
    a = b = c = 0
    alpha = 1e-8
    max_it = 1000
    # first it. reweighted least square with rho = absolute function
    _, norm = kirwls(Km, "abs", n_samples, a, b, c, alpha, max_it)
    if type_rho == "huber":
        a = np.median(norm)
        b = c = 0
    elif type_rho == "hampel":
        a = np.median(norm)
        b = np.percentile(norm, 95)
        c = np.max(norm)
    # find weights via second iterative reweighted least square with input rho
    w, _ = kirwls(Km, type_rho, n_samples, a, b, c, alpha, max_it)
    # kernel evaluated on plot data
    if periodic:
        K_plot = wrapped_kernel(ref, bandwidth=bw, Y=positions, type="von_mises")
    else:
        K_plot = gaussian_kernel(ref, bandwidth=bw, Y=positions)
    # final density
    z = np.dot(K_plot, w)
    z = np.reshape(z, xx.shape)

    return z
