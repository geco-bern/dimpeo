import numpy as np
import statsmodels.api as sm


def get_density_kde_statsmodels(ref):

    # TODO: try bw="cv_ml"

    xx, yy = np.mgrid[1:366, 0:10000:500j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde = sm.nonparametric.KDEMultivariate(ref, var_type="cc", bw="normal_reference")
    pdf = kde.pdf(positions)
    pdf = np.reshape(pdf, xx.shape)

    return pdf
