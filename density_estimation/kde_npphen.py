import os

os.environ["R_HOME"] = "/Users/davidbruggemann/miniconda3/envs/dimpeo/lib/R"
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

ks = importr("ks")


def get_density_kde_npphen(ref, diag_bw=False):

    with (ro.default_converter + numpy2ri.converter).context():

        r_ref = ro.conversion.get_conversion().py2rpy(ref)
        if diag_bw:
            r_H = ks.Hpi_diag(r_ref)
        else:
            r_H = ks.Hpi(r_ref)
        r_kernel = ks.kde(
            r_ref,
            H=r_H,
            xmin=ro.IntVector([1, 0]),
            xmax=ro.IntVector([365, 10000]),
            gridsize=ro.IntVector([365, 500]),
        )
        kernel_density = ro.conversion.get_conversion().rpy2py(r_kernel["estimate"])
        # kernel_eval_points_x = ro.conversion.get_conversion().rpy2py(
        #     r_kernel["eval.points"].byindex(0)[1]
        # )
        # kernel_eval_points_y = ro.conversion.get_conversion().rpy2py(
        #     r_kernel["eval.points"].byindex(1)[1]
        # )
        # anom = ano[:, 1] - kernel_eval_points_y[max_density[ano[:, 0] - 1]]

    return kernel_density
