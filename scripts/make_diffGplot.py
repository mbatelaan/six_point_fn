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

from gevpanalysis.common import read_pickle
from gevpanalysis.common import fit_value
from gevpanalysis.common import fit_value3
from gevpanalysis.common import read_correlators
from gevpanalysis.common import read_correlators4
from gevpanalysis.common import read_correlators5_complex

from gevpanalysis.common import make_matrices
from gevpanalysis.common import normalize_matrices
from gevpanalysis.common import gevp
from gevpanalysis.common import gevp_bootstrap
from gevpanalysis.common import weighted_avg_1_2_exp
from gevpanalysis.common import weighted_avg
from gevpanalysis.common import fit_correlation_matrix

from gevpanalysis.params import params

import gevpanalysis.plotting_scripts as plots


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


def fit_loop_weighted(
    datadir,
    datadir_qmax,
    plotdir,
    config,
    time_limits_nucl,
    time_limits_sigma,
    time_limits_nucldivsigma,
):
    """
    Read in the unperturbed two-point function fit data
    """
    # ============================================================
    # Nucleon correlators
    with open(datadir / (f"time_window_loop_nucl_Aexp.pkl"), "rb") as file_in:
        fitlist_nucl_1exp = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_nucl_Twoexp.pkl"), "rb") as file_in:
        fitlist_nucl_2exp = pickle.load(file_in)

    # ============================================================
    # Sigma correlators
    # filename_1 = "time_window_loop_sigma_Aexp.pkl"
    # with open(datadir / filename_1, "rb") as file_in:
    with open(
        datadir_qmax / "time_window_loop_sigma_Aexp.pkl",
        "rb",
    ) as file_in:
        fitlist_sigma_1exp = pickle.load(file_in)
    with open(
        datadir_qmax / "time_window_loop_sigma_Twoexp.pkl",
        "rb",
    ) as file_in:
        fitlist_sigma_2exp = pickle.load(file_in)

    # ============================================================
    # Nucleon divided by Sigma correlators
    with open(datadir / (f"time_window_loop_nucldivsigma_Aexp.pkl"), "rb") as file_in:
        fitlist_nucldivsigma_1exp = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_nucldivsigma_Twoexp.pkl"), "rb") as file_in:
        fitlist_nucldivsigma_2exp = pickle.load(file_in)

    # =============================================================
    weighted_energy_nucl, fitweights = weighted_avg(
        fitlist_nucl_1exp,
        fitlist_nucl_2exp,
        plotdir,
        "nucl",
        tmax_choice=config["tmax_nucl"],
        tminmin_1exp=time_limits_nucl[0, 0],
        tminmax_1exp=time_limits_nucl[0, 1],
        tminmin_2exp=time_limits_nucl[1, 0],
        tminmax_2exp=time_limits_nucl[1, 1],
        plot=False,
    )
    weighted_energy_sigma, fitweights = weighted_avg(
        fitlist_sigma_1exp,
        fitlist_sigma_2exp,
        plotdir,
        "sigma",
        tmax_choice=config["tmax_sigma"],
        tminmin_1exp=time_limits_sigma[0, 0],
        tminmax_1exp=time_limits_sigma[0, 1],
        tminmin_2exp=time_limits_sigma[1, 0],
        tminmax_2exp=time_limits_sigma[1, 1],
        plot=False,
    )
    weighted_energy_nucldivsigma, fitweights = weighted_avg(
        fitlist_nucldivsigma_1exp,
        fitlist_nucldivsigma_2exp,
        plotdir,
        "nucldivsigma",
        tmax_choice=config["tmax_ratio"],
        tminmin_1exp=time_limits_nucldivsigma[0, 0],
        tminmax_1exp=time_limits_nucldivsigma[0, 1],
        tminmin_2exp=time_limits_nucldivsigma[1, 0],
        tminmax_2exp=time_limits_nucldivsigma[1, 1],
        plot=False,
    )
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

    chosen_nucldivsigma_fit = [
        i
        for i in fitlist_nucldivsigma_1exp
        if i["x"][0] == config["tmin_ratio"] and i["x"][-1] == config["tmax_ratio"]
    ][0]
    # ratio_t_range = np.arange(config["tmin_ratio"], config["tmax_ratio"] + 1)

    return (
        weighted_energy_nucl,
        weighted_energy_sigma,
        weighted_energy_nucldivsigma,
        chosen_nucl_fit,
        chosen_sigma_fit,
        chosen_nucldivsigma_fit,
    )


def main():
    """
    Diagonalise correlation matrices to calculate an energy shift for various lambda values

    Run this file for each of the following arguments:
    qmax
    theta3
    theta4
    theta5
    theta7
    theta8

    This will run the GEVP analysis for the lamdba values specified in each of the yaml configuration file in the config/ directory with the above names.
    """
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())
    plt.rcParams.update({"figure.autolayout": False})

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)
    _metadata["Keywords"] = f"{pars.__dict__}"

    # Read in the analysis data from the yaml file if one is given
    # This config file contains all of the details on the specific dataset.
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
    plotdir = PROJECT_BASE_DIRECTORY / Path("data/plots") / Path(config["name"])
    datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist2" in config and config["onlytwist2"]:
        G2_nucl, G2_sigm = read_correlators5_complex(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )
    elif "qmax" in config and config["qmax"]:
        G2_nucl, G2_sigm = read_correlators4(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )
    else:
        print("else")
        G2_nucl, G2_sigm = read_correlators(
            pars, pickledir_k1, pickledir_k2, mom_strings
        )

    # ======================================================================
    # This function will loop over the values of lambda specified in the config file and do the GEVP analysis for each value of lambda
    gevp_lambda_loop_plot(G2_nucl, G2_sigm, config, datadir, plotdir, pars)
    # ======================================================================

    return


def gevp_lambda_loop_plot(G2_nucl, G2_sigm, config, datadir, plotdir, pars):
    """
    Loop over the values of lambda, and for each value construct the correlator matrix, solve the GEVP and fit to the ratio of correlators to extract the energy shift.

    G2_nucl: array which contains all of the 2point functions which begin with a Nucleon
    G2_sigm: array which contains all of the 2point functions which begin with a sigma baryon
    config: A dictionary with details about the dataset used
    pars: Object with details on the lattice ensemble.
    datadir: Path object of the directory where the data files are saved
    plotdir: Path object of the directory where the plots are saved
    """

    # Get variables from the config file
    lambdas = np.linspace(config["lmb_i"], config["lmb_f"], config["lmb_num"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    plotting = config["plotting"]
    time_loop = config["time_loop"]
    aexp_function = ff.initffncs("Aexp")
    twoexp_function = ff.initffncs("Twoexp")
    nucl_t_range = np.arange(config["tmin_nucl"], config["tmax_nucl"] + 1)
    sigma_t_range = np.arange(config["tmin_sigma"], config["tmax_sigma"] + 1)
    ratio_t_range = np.arange(config["tmin_ratio"], config["tmax_ratio"] + 1)

    # Read data from the pickle file
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "rb",
    ) as file_in:
        data = pickle.load(file_in)

    # ============================================================

    for i, lmb_val in enumerate(lambdas):
        print(f"\n====================\nLambda = {lmb_val}\n====================")

        # Construct a correlation matrix for each order in lambda(skipping order 0)
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )
        [matrix_1, matrix_2, matrix_3, matrix_4] = normalize_matrices(
            [matrix_1, matrix_2, matrix_3, matrix_4], time_choice=6
        )
        

        # ==================================================
        # TODO: Use one evec for all bootstraps to get Gt1_0, Gt2_0 or 500 evecs?
        # ==================================================
        # O(lambda^0) fit
        print("\nO(lambda^0) fit")
        (
            Gt1_0,
            Gt2_0,
            [eval_left0, evec_left0, eval_right0, evec_right0],
        ) = gevp_bootstrap(matrix_1, time_choice, delta_t, name="_test", show=False)

        # Construct the ratio of the two projected correlators
        ratio0 = np.abs(Gt1_0 / Gt2_0)

        # ==================================================
        # O(lambda^1) fit
        print("\nO(lambda^1) fit")
        (
            Gt1_1,
            Gt2_1,
            [eval_left1, evec_left1, eval_right1, evec_right1],
        ) = gevp_bootstrap(matrix_2, time_choice, delta_t, name="_test", show=False)

        # Construct the ratio of the two projected correlators
        ratio1 = np.abs(Gt1_1 / Gt2_1)

        # ==================================================
        # O(lambda^2) fit
        print("\nO(lambda^2) fit")
        (
            Gt1_2,
            Gt2_2,
            [eval_left2, evec_left2, eval_right2, evec_right2],
        ) = gevp_bootstrap(matrix_3, time_choice, delta_t, name="_test", show=False)

        # Construct the ratio of the two projected correlators
        ratio2 = np.abs(Gt1_2 / Gt2_2)

        # ==================================================
        # O(lambda^3) fit
        print("\nO(lambda^3) fit")
        (
            Gt1_3,
            Gt2_3,
            [eval_left3, evec_left3, eval_right3, evec_right3],
        ) = gevp_bootstrap(matrix_4, time_choice, delta_t, name="_test", show=False)

        # Construct the ratio of the two projected correlators
        ratio3 = np.abs(Gt1_3 / Gt2_3)

        # ==================================================
        print("plotting")
        effmass_ratio0 = stats.bs_effmass(ratio0, time_axis=1, spacing=1)
        effmass_ratio1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
        effmass_ratio2 = stats.bs_effmass(ratio2, time_axis=1, spacing=1)
        effmass_ratio3 = stats.bs_effmass(ratio3, time_axis=1, spacing=1)
        bootfit0 = data[i]["order0_fit"]
        bootfit1 = data[i]["order1_fit"]
        bootfit2 = data[i]["order2_fit"]
        bootfit3 = data[i]["order3_fit"]
        plots.plotting_script_diff_2(
            effmass_ratio0,
            effmass_ratio1,
            effmass_ratio2,
            effmass_ratio3,
            [bootfit0[:, 1], bootfit1[:, 1], bootfit2[:, 1], bootfit3[:, 1]],
            ratio_t_range,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val) + "_all",
            show=False,
        )
        plots.plotting_script_all(
            matrix_1,
            matrix_2,
            matrix_3,
            matrix_4,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val),
            show=False,
        )
        plots.plotting_script_all_N(
            matrix_1,
            matrix_2,
            matrix_3,
            matrix_4,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val),
            show=False,
        )
        plots.plotting_script_SS_new(
            matrix_1,
            matrix_2,
            matrix_3,
            matrix_4,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val),
            show=False,
        )
        plots.plotting_script_NN_new(
            matrix_1,
            matrix_2,
            matrix_3,
            matrix_4,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val),
            show=False,
        )
        
    return


             # plots.plotting_script_all(
             #     matrix_1,
             #     matrix_2,
             #     matrix_3,
             #     matrix_4,
             #     lmb_val,
             #     plotdir,
             #     name="_l" + str(lmb_val),
             #     show=False,
             # )
             # plots.plotting_script_all_N(
             #     matrix_1,
             #     matrix_2,
             #     matrix_3,
             #     matrix_4,
             #     lmb_val,
             #     plotdir,
             #     name="_l" + str(lmb_val),
             #     show=False,
             # )
             # plots.plot_real_imag(
             #     matrix_4,
             #     lmb_val,
             #     plotdir,
             #     name="_l" + str(lmb_val),
             #     show=False,
             # )
             # effmass_ratio0 = stats.bs_effmass(ratio0, time_axis=1, spacing=1)
             # effmass_ratio1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
             # effmass_ratio2 = stats.bs_effmass(ratio2, time_axis=1, spacing=1)
             # effmass_ratio3 = stats.bs_effmass(ratio3, time_axis=1, spacing=1)
             # plots.plotting_script_diff_2(
             #     effmass_ratio0,
             #     effmass_ratio1,
             #     effmass_ratio2,
             #     effmass_ratio3,
             #     [bootfit0[:, 1], bootfit1[:, 1], bootfit2[:, 1], bootfit3[:, 1]],
             #     ratio_t_range,
             #     lmb_val,
             #     plotdir,
             #     name="_l" + str(lmb_val) + "_all",
             #     show=False,
             # )

            # plots.plotting_script_gevp_corr(
            #     Gt1_3,
            #     Gt2_3,
            #     bootfit_state1_3,
            #     bootfit_state2_3,
            #     redchisq1_3,
            #     redchisq2_3,
            #     ratio_t_range,
            #     lmb_val,
            #     plotdir,
            #     name="_l" + str(lmb_val) + "_all",
            #     show=False,
            # )
            # plots.plot_real_imag_gevp(
            #     Gt1_3,
            #     Gt2_3,
            #     lmb_val,
            #     plotdir,
            #     name="_l" + str(lmb_val),
            #     show=False,
            # )
            # print("plotted")

    # ----------------------------------------------------------------------
    # Save the fit data to a pickle file
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "wb",
    ) as file_out:
        pickle.dump(fitlist, file_out)
    print("Saved the data")
    print(_metadata)
    return


if __name__ == "__main__":
    main()
