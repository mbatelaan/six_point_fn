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
from gevpanalysis.common import read_correlators2
from gevpanalysis.common import read_correlators3
from gevpanalysis.common import read_correlators4
from gevpanalysis.common import read_correlators5_complex
from gevpanalysis.common import read_correlators6
from gevpanalysis.common import make_matrices
from gevpanalysis.common import normalize_matrices
from gevpanalysis.common import gevp
from gevpanalysis.common import gevp_bootstrap
from gevpanalysis.common import weighted_avg_1_2_exp
from gevpanalysis.common import weighted_avg

from gevpanalysis.params import params


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

def plotting_script_unpert(
    correlator1,
    correlator2,
    ratio,
    fitvals1,
    fitvals2,
    fitvals,
    fitvals_effratio,
    nucl_t_range,
    sigma_t_range,
    ratio_t_range,
    plotdir,
    redchisq,
    name="",
    show=False,
):
    spacing = 2
    xlim = 28
    time = np.arange(0, np.shape(correlator1)[1])
    efftime = time[:-spacing] + 0.5
    correlator1 = stats.bs_effmass(correlator1, time_axis=1, spacing=1)
    correlator2 = stats.bs_effmass(correlator2, time_axis=1, spacing=1)
    effratio = stats.bs_effmass(ratio, time_axis=1, spacing=1)
    yavg_1 = np.average(correlator1, axis=0)
    ystd_1 = np.std(correlator1, axis=0)
    yavg_2 = np.average(correlator2, axis=0)
    ystd_2 = np.std(correlator2, axis=0)
    yavg_ratio = np.average(ratio, axis=0)
    ystd_ratio = np.std(ratio, axis=0)
    yavg_effratio = np.average(effratio, axis=0)
    ystd_effratio = np.std(effratio, axis=0)

    fit_energy_ratio = fitvals_effratio["param"][:, 1]
    fit_redchisq_ratio = fitvals_effratio["redchisq"]
    plt.figure(figsize=(5, 5))
    plt.errorbar(
        efftime[:xlim],
        yavg_effratio[:xlim],
        ystd_effratio[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    plt.plot(
        ratio_t_range,
        len(ratio_t_range) * [np.average(fit_energy_ratio)],
        color=_colors[0],
    )
    plt.fill_between(
        ratio_t_range,
        np.average(fit_energy_ratio) - np.std(fit_energy_ratio),
        np.average(fit_energy_ratio) + np.std(fit_energy_ratio),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{p}}')$ = {err_brackets(np.average(fit_energy_ratio),np.std(fit_energy_ratio))}",
        label=rf"$\Delta E(\lambda=0)$ = {err_brackets(np.average(fit_energy_ratio),np.std(fit_energy_ratio))}",
    )

    plt.legend(fontsize="x-small")
    # plt.ylabel(r"$\textrm{eff. energy}[G_n(\mathbf{p}')/G_{\Sigma}(\mathbf{0})]$")
    plt.ylabel(r"$\textrm{eff. energy}[G_n(\mathbf{0})/G_{\Sigma}(\mathbf{0})]$")
    plt.xlabel(r"$t/a$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.ylim(-0.1, 0.1)
    plt.savefig(plotdir / ("unpert_effmass.pdf"), metadata=_metadata)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        efftime[:xlim],
        yavg_1[:xlim],
        ystd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[0]:.2f}"
    )
    plt.errorbar(
        efftime[:xlim],
        yavg_2[:xlim],
        ystd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[1]:.2f}"
    )

    fit_energy_nucl = fitvals1["param"][:, 1]
    fit_redchisq_nucl = fitvals1["redchisq"]
    plt.plot(
        nucl_t_range,
        len(nucl_t_range) * [np.average(fit_energy_nucl)],
        color=_colors[0],
    )
    plt.fill_between(
        nucl_t_range,
        np.average(fit_energy_nucl) - np.std(fit_energy_nucl),
        np.average(fit_energy_nucl) + np.std(fit_energy_nucl),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{0}}) = {err_brackets(np.average(fitvals1),np.std(fitvals1))}$; $\chi^2_{{\textrm{{dof}}}} = {redchisqs[0]:.2f}$",
        label=rf"$E_N(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_nucl),np.std(fit_energy_nucl))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_nucl:.2f}$",
    )

    fit_energy_sigma = fitvals2["param"][:, 1]
    fit_redchisq_sigma = fitvals2["redchisq"]
    plt.plot(
        sigma_t_range,
        len(sigma_t_range) * [np.average(fit_energy_sigma)],
        color=_colors[1],
    )
    plt.fill_between(
        sigma_t_range,
        np.average(fit_energy_sigma) - np.std(fit_energy_sigma),
        np.average(fit_energy_sigma) + np.std(fit_energy_sigma),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{\Sigma}}(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_sigma),np.std(fit_energy_sigma))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_sigma:.2f}$",
    )
    # plt.plot(
    #     1000,
    #     1,
    #     label=rf"$\Delta E = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}$",
    # )
    plt.legend(fontsize="x-small")
    plt.ylabel(r"$\textrm{Effective energy}$")
    plt.xlabel(r"$t/a$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(0, 2))
    plt.xlim(0, xlim)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig(plotdir / ("unpert_energies.pdf"), metadata=_metadata)

    f, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    f.tight_layout()
    axs[0].errorbar(
        efftime[:xlim],
        yavg_1[:xlim],
        ystd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        # label=r"$N$",
    )
    axs[0].errorbar(
        efftime[:xlim],
        yavg_2[:xlim],
        ystd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        # label=r"$\Sigma$",
    )
    axs[0].plot(
        nucl_t_range,
        len(nucl_t_range) * [np.average(fit_energy_nucl)],
        color=_colors[0],
    )
    axs[0].fill_between(
        nucl_t_range,
        np.average(fit_energy_nucl) - np.std(fit_energy_nucl),
        np.average(fit_energy_nucl) + np.std(fit_energy_nucl),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals1),np.std(fitvals1))}",
        label=rf"$E_N(\mathbf{{0}})$ = {err_brackets(np.average(fit_energy_nucl),np.std(fit_energy_nucl))}",
    )
    axs[0].plot(
        sigma_t_range,
        len(sigma_t_range) * [np.average(fit_energy_sigma)],
        color=_colors[1],
    )
    axs[0].fill_between(
        sigma_t_range,
        np.average(fit_energy_sigma) - np.std(fit_energy_sigma),
        np.average(fit_energy_sigma) + np.std(fit_energy_sigma),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{\Sigma}}(\mathbf{{0}})$ = {err_brackets(np.average(fit_energy_sigma),np.std(fit_energy_sigma))}",
    )
    axs[0].plot(
        1000,
        1,
        label=rf"$\Delta E$ = {err_brackets(np.average(fit_energy_sigma-fit_energy_nucl),np.std(fit_energy_sigma-fit_energy_nucl))}",
    )
    axs[0].legend(fontsize="x-small")

    axs[1].errorbar(
        time[:xlim],
        yavg_ratio[:xlim],
        ystd_ratio[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        # label=r"$G_{N}/G_{\Sigma}$",
    )
    axs[1].plot(
        ratio_t_range, len(ratio_t_range) * [np.average(fitvals)], color=_colors[0]
    )
    axs[1].fill_between(
        ratio_t_range,
        np.average(fitvals) - np.std(fitvals),
        np.average(fitvals) + np.std(fitvals),
        alpha=0.3,
        color=_colors[2],
        label=rf"Fit = ${err_brackets(np.average(fitvals),np.std(fitvals))}$; $\chi^2_{{\textrm{{dof}}}} = {redchisq:.2f}$",
    )

    # axs[0].axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim), ylim=(0, 2))
    axs[0].set_ylabel(r"$\textrm{Effective energy}$")
    # axs[1].set_ylabel(r"$G_n(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$")
    axs[1].set_ylabel(r"$G_n(\mathbf{0})/G_{\Sigma}(\mathbf{0})$")
    plt.xlabel("$t/a$")
    axs[1].legend(fontsize="x-small")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("unpert_ratio" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


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
    if nucl_loop:
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
    if sigma_loop:
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
    if nucldivsigma_loop:
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

    chosen_nucldivsigma_fit = [
        i
        for i in fitlist_nucldivsigma_1exp
        if i["x"][0] == config["tmin_ratio"] and i["x"][-1] == config["tmax_ratio"]
    ][0]
    ratio_t_range = np.arange(config["tmin_ratio"], config["tmax_ratio"] + 1)

    plotting_script_unpert(
        np.abs(G2_nucl[0]),
        np.abs(G2_sigm[0]),
        ratio_unpert,
        chosen_nucl_fit,
        chosen_sigma_fit,
        bootfit_ratio[:, 0],
        # weighted_energy_nucldivsigma,
        chosen_nucldivsigma_fit,
        nucl_t_range,
        sigma_t_range,
        ratio_t_range,
        plotdir,
        # [redchisq1, redchisq2, redchisq_ratio],
        redchisq_ratio,
        name="_unpert_ratio",
        show=False,
    )


if __name__ == "__main__":
    main()
