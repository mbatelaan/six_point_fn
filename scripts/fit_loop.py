import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
from os.path import exists
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY
from gevpanalysis.util import find_file
from gevpanalysis.util import read_config

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from common import read_pickle
from common import fit_value
from common import fit_value3
from common import read_correlators
from common import read_correlators2
from common import read_correlators3
from common import read_correlators4
from common import read_correlators5_complex
from common import read_correlators6
from common import make_matrices
from common import normalize_matrices
from common import gevp
from common import gevp_bootstrap
from common import weighted_avg_1_2_exp
from common import weighted_avg

from params import params


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


def fit_loop_new(correlator, fitfunctions, time_limits, datadir, data_label):
    """Take a correlator with values for each bootstrap and each timeslice and a fitfunction, then loop over the fit windows given by the parameters and return a fit for each window.
    The function here is an object which has labels and initial parameters defined

    fitfunctions: A list of objects which contain fitting functions
    time_limits: A list of arrays defining the extent of the fit windows in the format [[tminmin, tminmax], [tmaxmin, tmaxmax]] for each fit function.
    """
    fitlist_list = []
    for ifunc, function in enumerate(fitfunctions):
        fitlist = stats.fit_loop(
            correlator,
            function,
            time_limits[ifunc],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        fitlist_list.append(fitlist)
        # with open(datadir / (f"time_window_loop_nucl_1exp.pkl"), "wb") as file_out:
        filename = f"time_window_loop_" + data_label + "_" + function.label + ".pkl"
        with open(datadir / filename, "wb") as file_out:
            pickle.dump(fitlist, file_out)

    return fitlist_list


def main():
    """Diagonalise correlation matrices to calculate an energy shift for various lambda values"""
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)

    # Read in the analysis data from the yaml file if one is given
    qmax_config = read_config("qmax")
    qmax_datadir = Path(qmax_config["analysis_dir"]) / Path("data")
    if len(sys.argv) == 2:
        config = read_config(sys.argv[1])
    else:
        config = read_config("qmax")

    # Set parameters to defaults defined in another YAML file
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)

    # Set the directories for reading data, saving data and saving plots
    pickledir_k1 = Path(config["pickle_dir1"])
    pickledir_k2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist" in config and config["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )
    elif "qmax" in config and config["qmax"]:
        G2_nucl, G2_sigm = read_correlators4(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )
    elif "onlytwist2" in config and config["onlytwist2"]:
        G2_nucl, G2_sigm = read_correlators5_complex(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )
    else:
        print("else")
        G2_nucl, G2_sigm = read_correlators(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )

    # Define fitting functions
    aexp_function = ff.initffncs("Aexp")
    twoexp_function = ff.initffncs("Twoexp")

    # ============================================================
    fit_loop_config = read_config("fit_loop")
    # [nucl_loop, sigma_loop, nucldivsigma_loop] = read_config("fit_loop")
    nucl_loop = fit_loop_config["nucl"]
    sigma_loop = fit_loop_config["sigma"]
    nucldivsigma_loop = fit_loop_config["nucldivsigma"]

    # if fit_loop:
    #     nucl_exist = False
    #     sigma_exist = False
    #     nucldivsigma_exist = False
    # else:
    #     nucl_exist = exists(datadir / (f"time_window_loop_nucl_Aexp.pkl")) and exists(
    #         datadir / (f"time_window_loop_nucl_Twoexp.pkl")
    #     )
    #     nucldivsigma_exist = exists(
    #         datadir / (f"time_window_loop_nucldivsigma_Aexp.pkl")
    #     ) and exists(datadir / (f"time_window_loop_nucldivsigma_Twoexp.pkl"))
    #     sigma_exist = exists(
    #         qmax_datadir / (f"time_window_loop_sigma_Aexp.pkl")
    #     ) and exists(qmax_datadir / (f"time_window_loop_sigma_Twoexp.pkl"))

    # ============================================================
    # Nucleon correlators
    if not nucl_loop:
        time_limits_nucl = np.array(
            [
                [[1, 18], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 10], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
            ]
        )

        [fitlist_nucl_1exp, fitlist_nucl_2exp] = fit_loop_new(
            np.abs(G2_nucl[0]),
            [aexp_function, twoexp_function],
            time_limits_nucl,
            datadir,
            "nucl",
        )
    else:
        with open(datadir / (f"time_window_loop_nucl_Aexp.pkl"), "rb") as file_in:
            fitlist_nucl_1exp = pickle.load(file_in)
        with open(datadir / (f"time_window_loop_nucl_Twoexp.pkl"), "rb") as file_in:
            fitlist_nucl_2exp = pickle.load(file_in)

    # ============================================================
    # Sigma correlators
    if not sigma_loop:
        time_limits_sigma = np.array(
            [
                [[1, 18], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
                [[1, 10], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
            ]
        )
        [fitlist_sigma_1exp, fitlist_sigma_2exp] = fit_loop_new(
            np.abs(G2_sigm[0]),
            [aexp_function, twoexp_function],
            time_limits_sigma,
            datadir,
            "sigma",
        )
    else:
        with open(
            "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_Aexp.pkl",
            "rb",
        ) as file_in:
            fitlist_sigma_1exp = pickle.load(file_in)
        with open(
            "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_Twoexp.pkl",
            "rb",
        ) as file_in:
            fitlist_sigma_2exp = pickle.load(file_in)

    # ============================================================
    # Nucleon divided by Sigma correlators
    if not nucldivsigma_loop:
        time_limits_nucldivsigma = np.array(
            [
                [[1, 18], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 3], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
            ]
        )
        [fitlist_nucldivsigma_1exp, fitlist_nucldivsigma_2exp] = fit_loop_new(
            np.abs(G2_nucl[0] / G2_sigm[0]),
            [aexp_function, twoexp_function],
            time_limits_nucl_div_sigma,
            datadir,
            "nucldivsigma",
        )
    else:
        with open(
            datadir / (f"time_window_loop_nucldivsigma_Aexp.pkl"), "rb"
        ) as file_in:
            fitlist_nucldivsigma_1exp = pickle.load(file_in)
        with open(
            datadir / (f"time_window_loop_nucldivsigma_Twoexp.pkl"), "rb"
        ) as file_in:
            fitlist_nucldivsigma_2exp = pickle.load(file_in)

    # =============================================================
    weighted_energy_nucl, fitweights = weighted_avg(
        fitlist_nucl_1exp,
        fitlist_nucl_2exp,
        plotdir,
        "nucl",
        tmax_choice=config["tmax_nucl"],
        tminmin_2exp=0,
        tminmax_2exp=4,
        tminmin_1exp=3,
        tminmax_1exp=16,
    )
    weighted_energy_nucldivsigma, fitweights = weighted_avg(
        fitlist_nucldivsigma_1exp,
        fitlist_nucldivsigma_2exp,
        plotdir,
        "nucldivsigma",
        tmax_choice=config["tmax_nucl"],
        tminmin_2exp=2,
        tminmax_2exp=2,
        tminmin_1exp=1,
        tminmax_1exp=15,
    )
    weighted_energy_sigma, fitweights = weighted_avg(
        fitlist_sigma_1exp,
        fitlist_sigma_2exp,
        plotdir,
        "sigma",
        tmax_choice=config["tmax_sigma"],
        tminmin_2exp=0,
        tminmax_2exp=4,
        tminmin_1exp=3,
        tminmax_1exp=16,
    )
    # =========================================

    weights_nucl = np.array([i["weight"] for i in fitlist_nucl_1exp])
    high_weight_nucl = np.argmax(weights_nucl)
    nucl_t_range = np.arange(
        fitlist_nucl_1exp[high_weight_nucl]["x"][0],
        fitlist_nucl_1exp[high_weight_nucl]["x"][-1] + 1,
    )
    print(f"nucl_t_range = {nucl_t_range}")

    weights_sigma = np.array([i["weight"] for i in fitlist_sigma_1exp])
    high_weight_sigma = np.argmax(weights_sigma)
    sigma_t_range = np.arange(
        fitlist_sigma_1exp[high_weight_sigma]["x"][0],
        fitlist_sigma_1exp[high_weight_sigma]["x"][-1] + 1,
    )
    print(f"sigma_t_range = {sigma_t_range}")

    # ===============================
    # HARD CODED RANGE!!!
    # ratio_t_range = np.arange(7, 18)
    ratio_t_range = np.arange(7, 20)
    # ===============================
    # Fit to the energy of the Nucleon and Sigma
    # Then fit to the ratio of those correlators to get the energy gap

    # bootfit_unpert_nucl, redchisq1 = fit_value3(
    #     np.abs(G2_nucl[0]), nucl_t_range, aexp_function, norm=1
    # )
    # bootfit_unpert_sigma, redchisq2 = fit_value3(
    #     np.abs(G2_sigm[0]), sigma_t_range, aexp_function, norm=1
    # )

    ratio_unpert = np.abs(G2_nucl[0] / G2_sigm[0])
    bootfit_ratio, redchisq_ratio = fit_value(ratio_unpert, ratio_t_range)
    bootfit_effratio, redchisq_effratio = fit_value3(
        ratio_unpert, ratio_t_range, aexp_function, norm=1
    )

    # ==================================================
    # Plot the effective energy of the unperturbed correlators
    # Pick out the fit determined by tmin and tmax set in the parameters file
    chosen_nucl_fit = [
        i
        for i in fitlist_nucl_1exp
        if i["x"][0] == config["tmin_nucl"] and i["x"][-1] == config["tmax_nucl"]
    ][0]
    nucl_t_range = np.arange(config["tmin_nucl"], config["tmax_nucl"] + 1)

    chosen_sigma_fit = [
        i
        for i in fitlist_sigma_1exp
        if i["x"][0] == config["tmin_sigma"] and i["x"][-1] == config["tmax_sigma"]
    ][0]
    sigma_t_range = np.arange(config["tmin_sigma"], config["tmax_sigma"] + 1)

    plotting_script_unpert(
        np.abs(G2_nucl[0]),
        np.abs(G2_sigm[0]),
        ratio_unpert,
        chosen_nucl_fit,
        chosen_sigma_fit,
        bootfit_ratio[:, 0],
        weighted_energy_nucldivsigma,
        nucl_t_range,
        sigma_t_range,
        ratio_t_range,
        plotdir,
        [redchisq1, redchisq2, redchisq_ratio],
        name="_unpert_ratio",
        show=False,
    )


if __name__ == "__main__":
    main()
