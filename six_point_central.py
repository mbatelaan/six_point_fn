import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as pypl
from matplotlib import rcParams

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from common import read_pickle
from common import fit_value
from common import fit_value3
from common import read_correlators
from common import read_correlators2
from common import make_matrices
from common import gevp

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


def plotting_script_all(
    corr_matrix, corr_matrix1, corr_matrix2, corr_matrix3, lmb_val, name="", show=False
):
    spacing = 2
    xlim = 16
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[1][1], axis=0)
    ystd = np.std(corr_matrix[1][1], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0)$",
    )
    yavg = np.average(corr_matrix1[1][1], axis=0)
    ystd = np.std(corr_matrix1[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix3[1][1], axis=0)
    ystd = np.std(corr_matrix3[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.4,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )

    yavg = np.average(corr_matrix[1][0], axis=0)
    ystd = np.std(corr_matrix[1][0], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{\Sigma N}(t),\ \mathcal{O}(\lambda^1)$",
    )
    yavg = np.average(corr_matrix2[1][0], axis=0)
    ystd = np.std(corr_matrix2[1][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{\Sigma N}(t),\ \mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )

    pypl.semilogy()
    pypl.legend(fontsize="x-small")
    # pypl.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    # pypl.title("$\lambda=0.04$")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_all_SS_" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_all_N(
    corr_matrix, corr_matrix1, corr_matrix2, corr_matrix3, lmb_val, name="", show=False
):
    spacing = 2
    xlim = 16
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[0][0], axis=0)
    ystd = np.std(corr_matrix[0][0], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0)$",
    )
    yavg = np.average(corr_matrix1[0][0], axis=0)
    ystd = np.std(corr_matrix1[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix3[0][0], axis=0)
    ystd = np.std(corr_matrix3[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.4,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )

    yavg = np.average(corr_matrix[0][1], axis=0)
    ystd = np.std(corr_matrix[0][1], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{N\Sigma}(t),\ \mathcal{O}(\lambda^1)$",
    )
    yavg = np.average(corr_matrix2[0][1], axis=0)
    ystd = np.std(corr_matrix2[0][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{N\Sigma}(t),\ \mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )

    pypl.semilogy()
    pypl.legend(fontsize="x-small")
    # pypl.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    # pypl.title("$\lambda=0.04$")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_all_NN_" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_diff_2(
    diffG1, diffG2, diffG3, diffG4, fitvals, t_range, lmb_val, name="", show=False
):
    spacing = 2
    xlim = 15
    time = np.arange(0, np.shape(diffG1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

    yeffavg_1 = np.average(diffG1, axis=0)
    yeffstd_1 = np.std(diffG1, axis=0)
    axs.errorbar(
        efftime[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[0])], color=_colors[0])
    axs.fill_between(
        t_range,
        np.average(fitvals[0]) - np.std(fitvals[0]),
        np.average(fitvals[0]) + np.std(fitvals[0]),
        alpha=0.3,
        color=_colors[0],
    )
    yeffavg_2 = np.average(diffG2, axis=0)
    yeffstd_2 = np.std(diffG2, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.2,
        yeffavg_2[:xlim],
        yeffstd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^2)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[1])], color=_colors[1])
    axs.fill_between(
        t_range,
        np.average(fitvals[1]) - np.std(fitvals[1]),
        np.average(fitvals[1]) + np.std(fitvals[1]),
        alpha=0.3,
        color=_colors[1],
    )
    yeffavg_3 = np.average(diffG3, axis=0)
    yeffstd_3 = np.std(diffG3, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.4,
        yeffavg_3[:xlim],
        yeffstd_3[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^3)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[2])], color=_colors[2])
    axs.fill_between(
        t_range,
        np.average(fitvals[2]) - np.std(fitvals[2]),
        np.average(fitvals[2]) + np.std(fitvals[2]),
        alpha=0.3,
        color=_colors[2],
    )
    yeffavg_4 = np.average(diffG4, axis=0)
    yeffstd_4 = np.std(diffG4, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.6,
        yeffavg_4[:xlim],
        yeffstd_4[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^4)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[3])], color=_colors[3])
    axs.fill_between(
        t_range,
        np.average(fitvals[3]) - np.std(fitvals[3]),
        np.average(fitvals[3]) + np.std(fitvals[3]),
        alpha=0.3,
        color=_colors[3],
    )

    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # pypl.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    pypl.setp(axs, xlim=(0, xlim), ylim=(-0.01, 0.4))
    pypl.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    pypl.xlabel("$t/a$")
    pypl.legend(fontsize="x-small")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    pypl.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return

def plotting_script_unpert(
    correlator1, correlator2, ratio, fitvals1, fitvals2, fitvals, t_range12, t_range, name="", show=False
):
    spacing = 2
    xlim = 20
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


    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        efftime[:xlim],
        yavg_effratio[:xlim],
        ystd_effratio[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.ylim(-0.2,0.4)
    pypl.savefig(plotdir / ("Testtttt.pdf"))


    f, axs = pypl.subplots(2, 1, figsize=(6, 6), sharex=True)
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
    axs[0].plot(t_range12, len(t_range12) * [np.average(fitvals1)], color=_colors[0])
    axs[0].fill_between(
        t_range12,
        np.average(fitvals1) - np.std(fitvals1),
        np.average(fitvals1) + np.std(fitvals1),
        alpha=0.3,
        color=_colors[0],
        label=rf"$E_N$ = {err_brackets(np.average(fitvals1),np.std(fitvals1))}",
    )
    axs[0].plot(t_range12, len(t_range12) * [np.average(fitvals2)], color=_colors[1])
    axs[0].fill_between(
        t_range12,
        np.average(fitvals2) - np.std(fitvals2),
        np.average(fitvals2) + np.std(fitvals2),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{\Sigma}}$ = {err_brackets(np.average(fitvals2),np.std(fitvals2))}",
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
    axs[1].plot(t_range, len(t_range) * [np.average(fitvals)], color=_colors[0])
    axs[1].fill_between(
        t_range,
        np.average(fitvals) - np.std(fitvals),
        np.average(fitvals) + np.std(fitvals),
        alpha=0.3,
        color=_colors[2],
        label=rf"Fit = {err_brackets(np.average(fitvals),np.std(fitvals))}",
    )

    # axs[0].axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # pypl.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    pypl.setp(axs, xlim=(0, xlim))
    axs[0].set_ylabel(r"$\textrm{Effective energy}$")
    axs[1].set_ylabel(r"$G_n(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$")
    pypl.xlabel("$t/a$")
    axs[1].legend(fontsize="x-small")
    # pypl.title("$\lambda=" + str(lmb_val) + "$")
    pypl.savefig(plotdir / ("unpert_ratio" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plot_lmb_dep(all_data):
    """Make a plot of the lambda dependence of the energy shift"""
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        all_data["lambdas"],
        np.average(all_data["order0_fit"], axis=1),
        np.std(all_data["order0_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"] +  0.0001,
        np.average(all_data["order1_fit"], axis=1),
        np.std(all_data["order1_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"] + 0.0002,
        np.average(all_data["order2_fit"], axis=1),
        np.std(all_data["order2_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"]+ 0.0003,
        np.average(all_data["order3_fit"], axis=1),
        np.std(all_data["order3_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    # pypl.ylim(-0.003, 0.035)
    # pypl.xlim(-0.01, 0.22)
    pypl.xlim(-0.01, lambdas[-1] * 1.1)
    pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)

    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))
    # pypl.show()


def plot_lmb_dep_fit(all_data, fit_data, fitfunction):
    """Make a plot of the lambda dependence of the energy shift"""
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        all_data["lambdas"],
        np.average(all_data["order0_fit"], axis=1),
        np.std(all_data["order0_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"] + 0.0001,
        np.average(all_data["order1_fit"], axis=1),
        np.std(all_data["order1_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"]+ 0.0002,
        np.average(all_data["order2_fit"], axis=1),
        np.std(all_data["order2_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas"] +0.0003,
        np.average(all_data["order3_fit"], axis=1),
        np.std(all_data["order3_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    params = np.average(fit_data, axis=0)
    fit_ydata = fitfunction(lambdas, *params)
    print(fit_ydata)
    pypl.plot(lambdas, fit_ydata, color="k")
    # axs.fill_between(
    #     t_range,
    #     np.average(fitvals[1]) - np.std(fitvals[1]),
    #     np.average(fitvals[1]) + np.std(fitvals[1]),
    #     alpha=0.3,
    #     color=_colors[1],
    # )

    pypl.legend(fontsize="x-small")
    pypl.xlim(-0.01, 0.22)
    pypl.ylim(0, 0.2)
    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep_fit.pdf"))
    # pypl.show()


def fitfunction(lmb, pars):
    deltaE = 0.5 * (pars[0] + pars[1]) - 0.5 * np.sqrt(
        (pars[0] - pars[1]) ** 2 + 4 * lmb ** 2 * pars[2] ** 2
    )
    return deltaE


def fitfunction2(lmb, pars0, pars1, pars2):
    deltaE = 0.5 * (pars0 + pars1) - 0.5 * np.sqrt(
        (pars0 - pars1) ** 2 + 4 * lmb ** 2 * pars2 ** 2
    )
    return deltaE


def fit_lmb(ydata, function, lambdas):
    """Fit the lambda dependence

    data is a correlator with tht bootstraps on the first index and the time on the second
    lambdas is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    # order0_fit[i] = bootfit1[:, 0]
    ydata = ydata.T
    data_set = ydata
    ydata_avg = np.average(data_set, axis=0)
    print(ydata_avg)
    print(lambdas)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    popt_avg, pcov_avg = curve_fit(
        function, lambdas, ydata_avg, sigma=covmat, maxfev=2000
    )
    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, np.linalg.inv(covmat))
    redchisq = chisq / len(lambdas)
    bootfit = []
    for iboot, values in enumerate(ydata):
        popt, pcov = curve_fit(function, lambdas, values, sigma=covmat, maxfev=2000)
        bootfit.append(popt)
    bootfit = np.array(bootfit)

    return bootfit, redchisq


if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)

    # Read in the directory data from the yaml file if one is given
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir.yaml"
    print("Reading directories from: ", config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # TODO: Set up a defaults.yaml file for when there is no input file
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist" in config and config["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(pars, pickledir, pickledir2, mom_strings)
    else:
        G2_nucl, G2_sigm = read_correlators(pars, pickledir, pickledir2, mom_strings)

    # lambdas = np.linspace(0.12,0.16,20)
    # lambdas = np.linspace(0,0.16,10)[1:]
    # lambdas = np.linspace(0,0.06,30)
    lambdas = np.linspace(0, 0.16, 30)  # [1:]
    # lambdas = np.linspace(0,0.16,10) #[1:]
    t_range = np.arange(config["t_range0"], config["t_range1"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    plotting = True

    order0_fit = np.zeros((len(lambdas), pars.nboot))
    order1_fit = np.zeros((len(lambdas), pars.nboot))
    order2_fit = np.zeros((len(lambdas), pars.nboot))
    order3_fit = np.zeros((len(lambdas), pars.nboot))
    red_chisq_list = np.zeros((4, len(lambdas)))
    
    aexp_function = ff.initffncs("Aexp")
    # aexp_eval = aexp_function.eval
    print(aexp_function.label)

    # Fit to the energy gap
    fit_range = np.arange(5,14)
    fit_range12 = np.arange(5,18)
    ratio_unpert = G2_nucl[0][:, :, 0] / G2_sigm[0][:,:,0]
    bootfit1, redchisq1 = fit_value3(G2_nucl[0][:,:,0], fit_range12, aexp_function, norm=1)
    bootfit2, redchisq2 = fit_value3(G2_sigm[0][:,:,0], fit_range12, aexp_function, norm=1)
    bootfit_ratio, redchisq_ratio = fit_value(ratio_unpert, fit_range)
    print(f"redchisq = {redchisq_ratio}")
    print(f"fit = {np.average(bootfit1,axis=0)}")
    print(f"fit = {np.average(bootfit2,axis=0)}")
    print(f"fit = {np.average(bootfit_ratio,axis=0)}")
    diff = bootfit1[:,1]-bootfit2[:,1]
    print(f"diff = {np.average(diff,axis=0)}")
    print(f"diff = {err_brackets(np.average(diff),np.std(diff))}")
    plotting_script_unpert(
        G2_nucl[0][:, :, 0],
        G2_sigm[0][:,:,0],
        ratio_unpert,
        bootfit1[:, 1],
        bootfit2[:, 1],
        bootfit_ratio[:, 0],
        fit_range12,
        fit_range,
        name="_unpert_ratio",
        show=False,
    )
    # order0_fit[i] = bootfit1[:, 1]
    # red_chisq_list[0, i] = redchisq1
    
    
    for i, lmb_val in enumerate(lambdas):
        print(f"Lambda = {lmb_val}\n")
        # Construct a correlation matrix for each order in lambda(skipping order 0)
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )

        Gt1_1, Gt2_1, evals = gevp(
            matrix_1, time_choice, delta_t, name="_test", show=False
        )
        ratio1 = Gt1_1 / Gt2_1
        effmass_ratio1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1) 
        bootfit1, redchisq1 = fit_value3(ratio1, t_range, aexp_function, norm=1)
        order0_fit[i] = bootfit1[:, 1]/2
        red_chisq_list[0, i] = redchisq1
        print(f"diff = {err_brackets(np.average(bootfit1[:,1]),np.std(bootfit1[:,1]))}")

        Gt1_2, Gt2_2, evals = gevp(
            matrix_2, time_choice, delta_t, name="_test", show=False
        )
        ratio2 = Gt1_2 / Gt2_2
        effmass_ratio2 = stats.bs_effmass(ratio2, time_axis=1, spacing=1) 
        bootfit2, redchisq2 = fit_value3(ratio2, t_range, aexp_function, norm=1)
        order1_fit[i] = bootfit2[:, 1]/2
        red_chisq_list[1, i] = redchisq2

        Gt1_3, Gt2_3, evals = gevp(
            matrix_3, time_choice, delta_t, name="_test", show=False
        )
        ratio3 = Gt1_3 / Gt2_3
        effmass_ratio3 = stats.bs_effmass(ratio3, time_axis=1, spacing=1) 
        bootfit3, redchisq3 = fit_value3(ratio3, t_range, aexp_function, norm=1)
        order2_fit[i] = bootfit3[:, 1]/2
        red_chisq_list[2, i] = redchisq3

        Gt1_4, Gt2_4, evals = gevp(
            matrix_4, time_choice, delta_t, name="_test", show=False
        )
        ratio4 = Gt1_4 / Gt2_4
        effmass_ratio4 = stats.bs_effmass(ratio4, time_axis=1, spacing=1) 
        bootfit4, redchisq4 = fit_value3(ratio4, t_range, aexp_function, norm=1)
        order3_fit[i] = bootfit4[:, 1]/2
        red_chisq_list[3, i] = redchisq4

        if plotting:
            plotting_script_all(
                matrix_1 / 1e39,
                matrix_2 / 1e39,
                matrix_3 / 1e39,
                matrix_4 / 1e39,
                lmb_val,
                name="_l" + str(lmb_val),
                show=False,
            )
            plotting_script_all_N(
                matrix_1 / 1e39,
                matrix_2 / 1e39,
                matrix_3 / 1e39,
                matrix_4 / 1e39,
                lmb_val,
                name="_l" + str(lmb_val),
                show=False,
            )
            plotting_script_diff_2(
                effmass_ratio1,
                effmass_ratio2,
                effmass_ratio3,
                effmass_ratio4,
                [bootfit1[:, 1], bootfit2[:, 1], bootfit3[:, 1], bootfit4[:, 1]],
                t_range,
                lmb_val,
                name="_l" + str(lmb_val) + "_all",
                show=False,
            )

    # ----------------------------------------------------------------------
    # Save the fit data to a pickle file
    all_data = {
        "lambdas": lambdas,
        "order0_fit": order0_fit,
        "order1_fit": order1_fit,
        "order2_fit": order2_fit,
        "order3_fit": order3_fit,
        "redchisq": red_chisq_list,
        "time_choice": time_choice,
        "delta_t": delta_t,
    }
    with open(
        datadir
        / (f"lambda_dep_t{time_choice}_dt{delta_t}_fit{t_range[0]}-{t_range[-1]}.pkl"),
        "wb",
    ) as file_out:
        pickle.dump(all_data, file_out)

    # bootfit, redchisq = fit_lmb(order0_fit, fitfunction2, lambdas)
    # print(bootfit)
    # print(redchisq)

    # ----------------------------------------------------------------------
    # Make a plot of the lambda dependence of the energy shift
    plot_lmb_dep(all_data)
    # plot_lmb_dep_fit(all_data, bootfit, fitfunction2)

    # pypl.figure(figsize=(6, 6))
    # pypl.errorbar(
    #     lambdas,
    #     np.average(order0_fit, axis=1),
    #     np.std(order0_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^1)$",
    #     color=_colors[0],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     lambdas+0.001,
    #     np.average(order1_fit, axis=1),
    #     np.std(order1_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^2)$",
    #     color=_colors[1],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     lambdas+0.002,
    #     np.average(order2_fit, axis=1),
    #     np.std(order2_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^3)$",
    #     color=_colors[2],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     lambdas+0.003,
    #     np.average(order3_fit, axis=1),
    #     np.std(order3_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^4)$",
    #     color=_colors[3],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    # pypl.ylim(0, 0.2)
    # pypl.xlabel("$\lambda$")
    # pypl.ylabel("$\Delta E$")
    # pypl.title(rf"$t_{{0}}={time_choice}, \Delta t={delta_t}$")
    # pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # pypl.savefig(plotdir / ("lambda_dep.pdf"))
    # # pypl.show()
