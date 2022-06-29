import numpy as np
from scipy import linalg
from pathlib import Path
import pickle
import yaml
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY
from gevpanalysis.util import find_file
from gevpanalysis.util import read_config

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from gevpanalysis.params import params

from gevpanalysis.lambda_fitting import Fitfunction1
from gevpanalysis.lambda_fitting import Fitfunction2
from gevpanalysis.lambda_fitting import Fitfunction3
from gevpanalysis.lambda_fitting import Fitfunction6
from gevpanalysis.lambda_fitting import Fitfunction_order4

# from gevpanalysis.lambda_fitting import fit_lmb
# from gevpanalysis.lambda_fitting import fit_lambda_dep
from gevpanalysis.lambda_fitting import lambdafit_3pt
from gevpanalysis.lambda_fitting import lambdafit_4pt
from gevpanalysis.lambda_fitting import lambdafit_allpt
from gevpanalysis.lambda_fitting import lambdafit_3pt_squared
from gevpanalysis.lambda_fitting import lambdafit_4pt_squared
from gevpanalysis.lambda_fitting import lambdafit_2pt_squared_fixed
from gevpanalysis.lambda_fitting import lambdafit_3pt_squared_fixed

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
_colors = [
    "#377eb8",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#ff7f00",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]
# From the theta tuning:
m_N = 0.4179255
m_S = 0.4641829


def main():
    """Fit to the lambda dependence of the energy shift and loop over the fit windows"""
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    pars = params(0)
    nboot = 200
    nbin = 1

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
        config = read_config(sys.argv[1])
    else:
        config = read_config("qmax")
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)

    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = PROJECT_BASE_DIRECTORY / Path("data/plots") / Path(config["name"])
    datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    print("datadir: ", datadir / ("lambda_dep.pkl"))

    t_range = np.arange(config["t_range0"], config["t_range1"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    lmb_val = config["lmb_val"]

    # Read data from the pickle file
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "rb",
    ) as file_in:
        data = pickle.load(file_in)

    # Filter out data points with a high reduced chi-squared value
    chisq_tol = 1.5  # 1.7
    redchisq0 = np.array([d["red_chisq0"] for d in data])
    redchisq1 = np.array([d["red_chisq1"] for d in data])
    redchisq2 = np.array([d["red_chisq2"] for d in data])
    redchisq3 = np.array([d["red_chisq3"] for d in data])
    indices0 = np.where(redchisq0 <= chisq_tol)[0]
    indices1 = np.where(redchisq1 <= chisq_tol)[0]
    indices2 = np.where(redchisq2 <= chisq_tol)[0]
    indices3 = np.where(redchisq3 <= chisq_tol)[0]
    fitlist0 = [data[ind] for ind in indices0]
    fitlist1 = [data[ind] for ind in indices1]
    fitlist2 = [data[ind] for ind in indices2]
    fitlist3 = [data[ind] for ind in indices3]
    fitlists = [fitlist0, fitlist1, fitlist2, fitlist3]
    lambdas3 = np.array([fit[f"lambdas"] for fit in fitlist3])

    fitfunc1 = Fitfunction1()
    fitfunc2 = Fitfunction2()
    fitfunc3 = Fitfunction3()
    fitfunc4 = Fitfunction_order4()

    # lambdafit_3pt(lambdas3, fitlists, datadir, fitfunc1)
    # lambdafit_4pt(lambdas3, fitlists, datadir, fitfunc1)
    # lambdafit_allpt(lambdas3, fitlists, datadir, fitfunc1)

    # lambdafit_4pt(lambdas3, fitlists, datadir, fitfunc4)
    # # lambdafit_allpt(lambdas3, fitlists, datadir, fitfunc4)

    # lambdafit_3pt_squared(lambdas3, fitlists, datadir, fitfunc2)
    # lambdafit_4pt_squared(lambdas3, fitlists, datadir, fitfunc2)

    # delta_E_fix = data[0]["chosen_nucldivsigma_fit"]["bootfit3"]
    delta_E_fix = data[0]["chosen_nucldivsigma_fit"]["param"][:,1]
    # print([key for key in delta_E_fix])
    print(f"delta_E_fix = {np.average(delta_E_fix)}")
    # print(f"delta_E_fix = {delta_E_fix}")
    delta_E_0 = np.array(
        [fit[f"order3_fit"][:, 1] for fit in fitlists[3]][0]
    )
    print(f"delta_E_0 = {np.average(delta_E_0)}")
    
    lambdafit_2pt_squared_fixed(lambdas3, fitlists, datadir, fitfunc3, delta_E_fix)
    lambdafit_3pt_squared_fixed(lambdas3, fitlists, datadir, fitfunc3, delta_E_fix)


if __name__ == "__main__":
    main()
