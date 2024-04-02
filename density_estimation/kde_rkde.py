import os

import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import cdist, pdist, squareform

os.environ["R_HOME"] = "/Users/davidbruggemann/miniconda3/envs/dimpeo/lib/R"
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

ks = importr("ks")


def rbf_kernel(X, bandwidth, Y=None):
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
    return K


def rho(x, typ="hampel", a=0, b=0, c=0):
    if typ == "huber":
        in1 = x <= a
        in2 = x > a
        in1_t = x[in1] ** 2 / 2
        in2_t = x[in2] * a - a**2 / 2
        L = np.sum(in1_t) + np.sum(in2_t)
    if typ == "hampel":
        in1 = x < a
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = c <= x
        in1_t = (x[in1] ** 2) / 2
        in2_t = a * x[in2] - a**2 / 2
        in3_t = (a * (x[in3] - c) ** 2 / (2 * (b - c))) + a * (b + c - a) / 2
        in4_t = np.ones(x[in4].shape) * a * (b + c - a) / 2
        L = np.sum(in1_t) + np.sum(in2_t) + np.sum(in3_t) + np.sum(in4_t)
    if typ == "square":
        t = x**2
    if typ == "abs":
        t = np.abs(x)
        L = np.sum(t)

    return L / x.shape[0]


def loss(x, typ="hampel", a=0, b=0, c=0):
    return rho(x, typ=typ, a=a, b=b, c=c) / x.shape[0]


def psi(x, typ="hampel", a=0, b=0, c=0):
    if typ == "huber":
        return np.minimum(x, a)
    if typ == "hampel":
        in1 = x < a
        in2 = np.logical_and(a <= x, x < b)
        in3 = np.logical_and(b <= x, x < c)
        in4 = c <= x
        in1_t = x[in1]
        in2_t = np.ones(x[in2].shape) * a
        in3_t = a * (c - x[in3]) / (c - b)
        in4_t = np.zeros(x[in4].shape)
        return np.concatenate((in1_t, in2_t, in3_t, in4_t)).reshape((-1, x.shape[1]))
    if typ == "square":
        return 2 * x
    if typ == "abs":
        return 1


def phi(x, typ="hampel", a=0, b=0, c=0):
    x[x == 0] = 10e-6
    return psi(x, typ=typ, a=a, b=b, c=c) / x


def irls(Km, type_rho, n, a, b, c, alpha=10e-8, max_it=100):
    """
    Iterative reweighted least-square
    """
    # init weights
    w = np.ones((n, 1)) / n
    # first pass
    t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
    t2 = -2 * np.dot(Km, w)
    t3 = np.dot(np.dot(w.T, Km), w)
    t = t1 + t2 + t3
    norm = np.sqrt(t)
    J = loss(norm, typ=type_rho, a=a, b=b, c=c)
    stop = 0
    count = 0
    losses = [J]
    while not stop:
        count += 1
        # print("i: {}  loss: {}".format(count, J))
        J_old = J
        # update weights
        w = phi(norm, typ=type_rho, a=a, b=b, c=c)
        w = w / np.sum(w)
        t1 = np.diag(Km).reshape((-1, 1))  #  (-1, dimension)
        t2 = -2 * np.dot(Km, w)
        t3 = np.dot(np.dot(w.T, Km), w)
        t = t1 + t2 + t3
        norm = np.sqrt(t)
        # update loss
        J = loss(norm, typ=type_rho, a=a, b=b, c=c)
        losses.append(J)
        if np.abs(J - J_old) < (J_old * alpha):
            stop = 1
        if count == max_it:
            print("Reached maximum of {} iterations without convergence".format(count))
            stop = 1
    return w, norm, losses


def get_density_kde_rkde(ref, type_rho="hampel"):

    # bw estimator from sm.nonparametric.KDEMultivariate
    # bw = 1.06 * np.std(ref, axis=0) * ref.shape[0] ** (-1.0 / (4 + ref.shape[1]))

    with (ro.default_converter + numpy2ri.converter).context():

        r_ref = ro.conversion.get_conversion().py2rpy(ref)
        r_H = ks.Hpi_diag(r_ref)
        bw = ro.conversion.get_conversion().rpy2py(r_H)
        bw = np.sqrt(np.diag(bw))
        # print("bw: {}".format(bw))

    xx, yy = np.mgrid[1:366, 0:10000:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    # kernel matrix
    n_samples, d = ref.shape
    assert d == 2
    Km = (
        rbf_kernel(ref, bandwidth=bw)
        * (2 * np.pi) ** (-d / 2.0)
        * np.prod(bw**2) ** (-1 / 2.0)
    )
    # find a, b, c via iterative reweighted least square
    a = b = c = 0
    alpha = 10e-8
    max_it = 1000
    # first it. reweighted least square with rho = absolute function
    w, norm, losses = irls(Km, "abs", n_samples, a, b, c, alpha, max_it)
    a = np.median(norm)
    b = np.percentile(norm, 75)
    c = np.percentile(norm, 85)
    # find weights via second iterative reweighted least square with input rho
    w, norm, losses = irls(Km, type_rho, n_samples, a, b, c, alpha, max_it)
    # kernel evaluated on plot data
    K_plot = (
        rbf_kernel(ref, bandwidth=bw, Y=positions)
        * (2 * np.pi) ** (-d / 2.0)
        * np.prod(bw**2) ** (-1 / 2.0)
    )
    # final density
    z = np.dot(K_plot, w)
    z = np.reshape(z, xx.shape)

    return z
