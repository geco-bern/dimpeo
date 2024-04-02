import numpy as np
from sklearn.neighbors import KernelDensity


def get_density_kde_sklearn(ref):
    # NOTE: does not work well because of isotropic kernel

    xx, yy = np.mgrid[1:366, 0:10000:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    kde = KernelDensity(kernel="gaussian", bandwidth=50).fit(ref)
    pdf = np.exp(kde.score_samples(positions))
    pdf = np.reshape(pdf, xx.shape)

    return pdf
