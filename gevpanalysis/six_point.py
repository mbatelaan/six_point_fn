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


def plotting_script_all(
    corr_matrix,
    corr_matrix1,
    corr_matrix2,
    corr_matrix3,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
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

    plt.semilogy()
    plt.legend(fontsize="x-small")
    # plt.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    # plt.title("$\lambda=0.04$")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    # plt.xlabel(r"$\textrm{t/a}$")
    plt.xlabel(r"$t/a$")
    plt.savefig(plotdir / ("comp_plot_all_SS_" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_all_N(
    corr_matrix,
    corr_matrix1,
    corr_matrix2,
    corr_matrix3,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
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

    plt.semilogy()
    plt.legend(fontsize="x-small")
    # plt.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    # plt.title("$\lambda=0.04$")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    # plt.xlabel(r"$\textrm{t/a}$")
    plt.xlabel(r"$t/a$")
    plt.savefig(plotdir / ("comp_plot_all_NN_" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_diff_2(
    diffG1,
    diffG2,
    diffG3,
    diffG4,
    fitvals,
    t_range,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(diffG1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

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
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim), ylim=(-0.05, 0.25))
    plt.ylabel(r"$\Delta E_{\textrm{eff}}$")
    plt.xlabel("$t/a$")
    plt.legend(fontsize="x-small")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("diff_G" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


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
    redchisqs,
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
        len(ratio_t_range) * [np.average(fitvals_effratio)],
        color=_colors[0],
    )
    plt.fill_between(
        ratio_t_range,
        np.average(fitvals_effratio) - np.std(fitvals_effratio),
        np.average(fitvals_effratio) + np.std(fitvals_effratio),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}",
        label=rf"$\Delta E(\lambda=0)$ = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}",
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
        label=rf"Fit = ${err_brackets(np.average(fitvals),np.std(fitvals))}$; $\chi^2_{{\textrm{{dof}}}} = {redchisqs[2]:.2f}$",
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


def plot_lmb_dep(all_data, plotdir):
    """Make a plot of the lambda dependence of the energy shift"""
    plt.figure(figsize=(6, 6))
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
        all_data["lambdas"] + 0.0003,
        np.average(all_data["order3_fit"], axis=1),
        np.std(all_data["order3_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    plt.xlim(-0.01, lambdas[-1] * 1.1)
    plt.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.savefig(plotdir / ("lambda_dep.pdf"), metadata=_metadata)
    # plt.show()
    plt.close()
    return


def fit_loop(
    G2_nucl,
    G2_sigm,
    aexp_function,
    twoexp_function,
    time_choice,
    delta_t,
    datadir,
    time_limits,
    time_limits2,
    which_corr=[True, True, True],
):
    # Nucleon fit loop
    # time_limits = [[5, 18], [10, 25]]
    (
        fitlist_nucl,
        fitlist_nucl_2exp,
        fitlist_sigma,
        fitlist_sigma_2exp,
        fitlist_small,
        fitlist_large,
        fitlist_nucldivsigma,
        fitlist_nucldivsigma_2exp,
    ) = ([], [], [], [], [], [], [], [])
    if which_corr[0]:
        print("\n\nNucleon fitting")
        fitlist_nucl = stats.fit_loop(
            np.abs(G2_nucl[0]),
            aexp_function,
            time_limits[0],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_nucl_1exp.pkl"), "wb") as file_out:
            pickle.dump(fitlist_nucl, file_out)
        fitlist_nucl_2exp = stats.fit_loop(
            np.abs(G2_nucl[0]),
            twoexp_function,
            time_limits2[0],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_nucl_2exp.pkl"), "wb") as file_out:
            pickle.dump(fitlist_nucl_2exp, file_out)

        ratio_unpert = np.abs(G2_nucl[0] / G2_sigm[0])
        fitlist_nucldivsigma = stats.fit_loop(
            ratio_unpert,
            aexp_function,
            time_limits[0],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(
            datadir / (f"time_window_loop_nucldivsigma_1exp.pkl"), "wb"
        ) as file_out:
            pickle.dump(fitlist_nucldivsigma, file_out)
        fitlist_nucldivsigma_2exp = stats.fit_loop(
            ratio_unpert,
            twoexp_function,
            time_limits2[0],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(
            datadir / (f"time_window_loop_nucldivsigma_2exp.pkl"), "wb"
        ) as file_out:
            pickle.dump(fitlist_nucldivsigma_2exp, file_out)

    if which_corr[1]:
        print("\n\nSigma fitting")
        # Sigma fit loop
        fitlist_sigma = stats.fit_loop(
            np.abs(G2_sigm[0]),
            aexp_function,
            time_limits[1],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_sigma_1exp.pkl"), "wb") as file_out:
            pickle.dump(fitlist_sigma, file_out)
        fitlist_sigma_2exp = stats.fit_loop(
            np.abs(G2_sigm[0]),
            twoexp_function,
            time_limits2[1],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_sigma_2exp.pkl"), "wb") as file_out:
            pickle.dump(fitlist_sigma_2exp, file_out)

    if which_corr[2]:
        print("\n\nSmall lambda fitting")
        # small lambda fit loop
        lmb_val = 0.003
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )
        Gt1_4, Gt2_4, evals = gevp(
            matrix_4, time_choice, delta_t, name="_test", show=False
        )
        ratio4 = Gt1_4 / Gt2_4
        fitlist_small = stats.fit_loop(
            ratio4,
            aexp_function,
            time_limits[2],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_lambda_small.pkl"), "wb") as file_out:
            pickle.dump(fitlist_small, file_out)

        # large lambda fit loop
        print("\n\nLarge lambda fitting")
        lmb_val = 0.05
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )
        Gt1_4, Gt2_4, evals = gevp(
            matrix_4, time_choice, delta_t, name="_test", show=False
        )
        ratio4 = Gt1_4 / Gt2_4
        fitlist_large = stats.fit_loop(
            ratio4,
            aexp_function,
            time_limits[2],
            plot=False,
            disp=True,
            time=False,
            weights_=True,
        )
        with open(datadir / (f"time_window_loop_lambda_large.pkl"), "wb") as file_out:
            pickle.dump(fitlist_large, file_out)
    return (
        fitlist_nucl,
        fitlist_nucl_2exp,
        fitlist_sigma,
        fitlist_sigma_2exp,
        fitlist_small,
        fitlist_large,
        fitlist_nucldivsigma,
        fitlist_nucldivsigma_2exp,
    )


def main():
    """Diagonalise correlation matrices to calculate an energy shift for various lambda values"""
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)

    # Read in the analysis data from the yaml file if one is given
    if len(sys.argv) == 2:
        config = read_config(sys.argv[1])
    else:
        config = read_config("data_dir_qmax")

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

    # Set the analysis parameters that will be used (from the yaml file)
    lambdas = np.linspace(config["lmb_i"], config["lmb_f"], 15)
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    plotting = config["plotting"]
    time_loop = config["time_loop"]
    aexp_function = ff.initffncs("Aexp")
    twoexp_function = ff.initffncs("Twoexp")

    if time_loop:
        which_corr = [True, False, True]
        time_limits = np.array(
            [
                [[1, 18], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 18], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
                [[5, 18], [config["tmax_ratio"] - 2, config["tmax_ratio"] + 2]],
            ]
        )
        time_limits2 = np.array(
            [
                [[1, 10], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 10], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
                [[5, 10], [config["tmax_ratio"] - 2, config["tmax_ratio"] + 2]],
            ]
        )
        (
            fitlist_nucl_1exp,
            fitlist_nucl_2exp,
            fitlist_sigma_1exp,
            fitlist_sigma_2exp,
            fitlist_small,
            fitlist_large,
            fitlist_nucldivsigma_1exp,
            fitlist_nucldivsigma_2exp,
        ) = fit_loop(
            G2_nucl,
            G2_sigm,
            aexp_function,
            twoexp_function,
            time_choice,
            delta_t,
            datadir,
            time_limits,
            time_limits2,
            which_corr,
        )
    else:
        nucl_exist = exists(datadir / (f"time_window_loop_nucl_1exp.pkl")) and exists(
            datadir / (f"time_window_loop_nucl_2exp.pkl")
        )
        sigma_exist = True  # Because we'll use the fit from qmax for all other datasets
        small_exist = exists(datadir / (f"time_window_loop_lambda_small.pkl"))
        large_exist = exists(datadir / (f"time_window_loop_lambda_large.pkl"))
        which_corr = [
            not nucl_exist,
            not sigma_exist,
            not (small_exist and large_exist),
        ]
        time_limits = np.array(
            [
                [[1, 18], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 18], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
                [[5, 18], [config["tmax_ratio"] - 2, config["tmax_ratio"] + 2]],
            ]
        )
        time_limits2 = np.array(
            [
                [[1, 10], [config["tmax_nucl"] - 2, config["tmax_nucl"] + 2]],
                [[1, 10], [config["tmax_sigma"] - 2, config["tmax_sigma"] + 2]],
                [[5, 10], [config["tmax_ratio"] - 2, config["tmax_ratio"] + 2]],
            ]
        )

        (
            fitlist_nucl_1exp,
            fitlist_nucl_2exp,
            fitlist_sigma_1exp,
            fitlist_sigma_2exp,
            fitlist_small,
            fitlist_large,
            fitlist_nucldivsigma_1exp,
            fitlist_nucldivsigma_2exp,
        ) = fit_loop(
            G2_nucl,
            G2_sigm,
            aexp_function,
            twoexp_function,
            time_choice,
            delta_t,
            datadir,
            time_limits,
            time_limits2,
            which_corr,
        )
        if nucl_exist:
            with open(datadir / (f"time_window_loop_nucl_1exp.pkl"), "rb") as file_in:
                fitlist_nucl_1exp = pickle.load(file_in)
            with open(datadir / (f"time_window_loop_nucl_2exp.pkl"), "rb") as file_in:
                fitlist_nucl_2exp = pickle.load(file_in)
            with open(
                datadir / (f"time_window_loop_nucldivsigma_1exp.pkl"), "rb"
            ) as file_in:
                fitlist_nucldivsigma_1exp = pickle.load(file_in)
            with open(
                datadir / (f"time_window_loop_nucldivsigma_2exp.pkl"), "rb"
            ) as file_in:
                fitlist_nucldivsigma_2exp = pickle.load(file_in)
        if sigma_exist:
            with open(
                "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_1exp.pkl",
                "rb",
            ) as file_in:
                fitlist_sigma_1exp = pickle.load(file_in)
            with open(
                "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_2exp.pkl",
                "rb",
            ) as file_in:
                fitlist_sigma_2exp = pickle.load(file_in)
        if small_exist and large_exist:
            with open(
                datadir / (f"time_window_loop_lambda_small.pkl"), "rb"
            ) as file_in:
                fitlist_small = pickle.load(file_in)
            with open(
                datadir / (f"time_window_loop_lambda_large.pkl"), "rb"
            ) as file_in:
                fitlist_large = pickle.load(file_in)

    # =========================================
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
    # print(fitlist_nucl_1exp[high_weight_nucl]["redchisq"])
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

    weights_small = np.array([i["weight"] for i in fitlist_small])
    high_weight_small = np.argmax(weights_small)
    weights_large = np.array([i["weight"] for i in fitlist_large])
    high_weight_large = np.argmax(weights_large)
    ratio_t_range = np.arange(
        min(
            fitlist_small[high_weight_small]["x"][0],
            fitlist_large[high_weight_large]["x"][0],
        ),
        fitlist_large[high_weight_large]["x"][-1] + 1,
    )
    print(f"ratio_t_range = {ratio_t_range}")

    # ===============================
    # HARD CODED RANGE!!!
    ratio_t_range = np.arange(7, 18)
    # ===============================
    # Fit to the energy of the Nucleon and Sigma
    # Then fit to the ratio of those correlators to get the energy gap

    bootfit_unpert_nucl, redchisq1 = fit_value3(
        np.abs(G2_nucl[0]), nucl_t_range, aexp_function, norm=1
    )
    bootfit_unpert_sigma, redchisq2 = fit_value3(
        np.abs(G2_sigm[0]), sigma_t_range, aexp_function, norm=1
    )

    ratio_unpert = np.abs(G2_nucl[0] / G2_sigm[0])
    bootfit_ratio, redchisq_ratio = fit_value(ratio_unpert, ratio_t_range)
    bootfit_effratio, redchisq_effratio = fit_value3(
        ratio_unpert, ratio_t_range, aexp_function, norm=1
    )

    # ==================================================
    # Plot the effective energy of the unperturbed correlators
    tmax_choice = config["tmax_nucl"]
    tmin_choice = config["tmin_nucl"]
    tmax_1exp = np.array([i["x"][-1] for i in fitlist_nucl_1exp])
    tmin_1exp = np.array([i["x"][0] for i in fitlist_nucl_1exp])
    indices = np.where(tmax_1exp == tmax_choice)
    index = indices[0][np.where(tmin_1exp[indices] == tmin_choice)]
    print(f"\n\nIndex = {index}\n\n")
    chosen_nucl_fit = fitlist_nucl_1exp[index[0]]
    print(chosen_nucl_fit["x"])
    nucl_t_range = np.arange(tmin_choice, tmax_choice + 1)

    tmax_choice = config["tmax_sigma"]
    tmin_choice = config["tmin_sigma"]
    tmax_1exp = np.array([i["x"][-1] for i in fitlist_sigma_1exp])
    tmin_1exp = np.array([i["x"][0] for i in fitlist_sigma_1exp])
    print(tmin_1exp)
    print(tmax_1exp)
    indices = np.where(tmax_1exp == tmax_choice)
    index = indices[0][np.where(tmin_1exp[indices] == tmin_choice)]
    print(f"\n\nIndex = {index}\n\n")
    chosen_sigma_fit = fitlist_sigma_1exp[index[0]]
    print(chosen_sigma_fit["x"])
    sigma_t_range = np.arange(tmin_choice, tmax_choice + 1)

    plotting_script_unpert(
        np.abs(G2_nucl[0]),
        np.abs(G2_sigm[0]),
        ratio_unpert,
        # bootfit_unpert_nucl[:, 1],
        # bootfit_unpert_sigma[:, 1],
        chosen_nucl_fit,
        chosen_sigma_fit,
        bootfit_ratio[:, 0],
        weighted_energy_nucldivsigma,
        # bootfit_effratio[:, 1],
        nucl_t_range,
        sigma_t_range,
        ratio_t_range,
        plotdir,
        [redchisq1, redchisq2, redchisq_ratio],
        name="_unpert_ratio",
        show=False,
    )

    fitlist = []
    for i, lmb_val in enumerate(lambdas):
        print(f"\n====================\nLambda = {lmb_val}\n====================")

        # Construct a correlation matrix for each order in lambda(skipping order 0)
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )

        [matrix_1, matrix_2, matrix_3, matrix_4] = normalize_matrices(
            [matrix_1, matrix_2, matrix_3, matrix_4], time_choice=6
        )

        # print("\n\nmatrix shape = \n", np.shape(matrix_4))
        # print("\n\nmatrix elem = \n", matrix_4[1, 1, 3, 16])
        # print("\n\nmatrix1 elem = \n", np.average(matrix_1[0, 1, :, 1:13], axis=0))
        # print("\n\nmatrix2 elem = \n", np.average(matrix_2[0, 1, :, 1:13], axis=0))
        # print("\n\nmatrix3 elem = \n", np.average(matrix_3[0, 1, :, 1:13], axis=0))
        # print("\n\nmatrix4 elem = \n", np.average(matrix_4[0, 1, :, 1:13], axis=0))

        # ==================================================
        # O(lambda^0) fit
        (
            Gt1_0,
            Gt2_0,
            [eval_left0, evec_left0, eval_right0, evec_right0],
        ) = gevp_bootstrap(matrix_1, time_choice, delta_t, name="_test", show=False)
        # Gt1_0 = np.einsum("ki,ijkl,kj->kl", evec_left0[:, :, 0], matrix_1, evec_right0[:, :, 0])
        # Gt2_0 = np.einsum("ki,ijkl,kj->kl", evec_left0[:, :, 1], matrix_1, evec_right0[:, :, 1])
        print("\n evec shape = ", np.shape(evec_left0))
        print("\n evec left avg = \n", np.average(evec_left0, axis=0))
        print("\n evec right avg = \n", np.average(evec_right0, axis=0))
        ratio0 = Gt1_0 / Gt2_0
        effmass_ratio0 = stats.bs_effmass(ratio0, time_axis=1, spacing=1)
        bootfit_state1_0, redchisq1_0 = fit_value3(
            Gt1_0, ratio_t_range, aexp_function, norm=1
        )
        bootfit_state2_0, redchisq2_0 = fit_value3(
            Gt2_0, ratio_t_range, aexp_function, norm=1
        )
        bootfit0, redchisq0 = fit_value3(ratio0, ratio_t_range, aexp_function, norm=1)
        print(redchisq0)

        # ==================================================
        # O(lambda^1) fit
        (
            Gt1_1,
            Gt2_1,
            [eval_left1, evec_left1, eval_right1, evec_right1],
        ) = gevp_bootstrap(matrix_2, time_choice, delta_t, name="_test", show=False)
        # Gt1_1 = np.einsum("ki,ijkl,kj->kl", evec_left1[:, :, 0], matrix_2, evec_right1[:, :, 0])
        # Gt2_1 = np.einsum("ki,ijkl,kj->kl", evec_left1[:, :, 1], matrix_2, evec_right1[:, :, 1])
        ratio1 = Gt1_1 / Gt2_1
        effmass_ratio1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
        bootfit_state1_1, redchisq1_1 = fit_value3(
            Gt1_1, ratio_t_range, aexp_function, norm=1
        )
        bootfit_state2_1, redchisq2_1 = fit_value3(
            Gt2_1, ratio_t_range, aexp_function, norm=1
        )
        bootfit1, redchisq1 = fit_value3(ratio1, ratio_t_range, aexp_function, norm=1)
        print(redchisq1)

        # ==================================================
        # O(lambda^2) fit
        (
            Gt1_2,
            Gt2_2,
            [eval_left2, evec_left2, eval_right2, evec_right2],
        ) = gevp_bootstrap(matrix_3, time_choice, delta_t, name="_test", show=False)
        # Gt1_2 = np.einsum("ki,ijkl,kj->kl", evec_left2[:, :, 0], matrix_3, evec_right2[:, :, 0])
        # Gt2_2 = np.einsum("ki,ijkl,kj->kl", evec_left2[:, :, 1], matrix_3, evec_right2[:, :, 1])
        ratio2 = Gt1_2 / Gt2_2
        effmass_ratio2 = stats.bs_effmass(ratio2, time_axis=1, spacing=1)
        bootfit_state1_2, redchisq1_2 = fit_value3(
            Gt1_2, ratio_t_range, aexp_function, norm=1
        )
        bootfit_state2_2, redchisq2_2 = fit_value3(
            Gt2_2, ratio_t_range, aexp_function, norm=1
        )
        bootfit2, redchisq2 = fit_value3(ratio2, ratio_t_range, aexp_function, norm=1)
        print(redchisq2)

        # ==================================================
        # O(lambda^3) fit
        (
            Gt1_3,
            Gt2_3,
            [eval_left3, evec_left3, eval_right3, evec_right3],
        ) = gevp_bootstrap(matrix_4, time_choice, delta_t, name="_test", show=False)
        # Gt1_3 = np.einsum("ki,ijkl,kj->kl", evec_left3[:, :, 0], matrix_4, evec_right3[:, :, 0])
        # Gt2_3 = np.einsum("ki,ijkl,kj->kl", evec_left3[:, :, 1], matrix_4, evec_right3[:, :, 1])
        ratio3 = Gt1_3 / Gt2_3
        effmass_ratio3 = stats.bs_effmass(ratio3, time_axis=1, spacing=1)
        bootfit_state1_3, redchisq1_3 = fit_value3(
            Gt1_3, ratio_t_range, aexp_function, norm=1
        )
        bootfit_state2_3, redchisq1_3 = fit_value3(
            Gt2_3, ratio_t_range, aexp_function, norm=1
        )
        bootfit3, redchisq3 = fit_value3(ratio3, ratio_t_range, aexp_function, norm=1)
        print(redchisq3)

        # ==================================================
        # Divide the nucleon correlator by the Sigma correlator and fit this ratio to get the energy shift.
        if lmb_val == 0:
            order3_states_fit_divsigma = np.zeros((2, pars.nboot, 2))
        else:
            sigma_ = G2_sigm[0]
            Gt1_3_divsigma = np.abs(Gt1_3 / sigma_)
            Gt2_3_divsigma = np.abs(Gt2_3 / sigma_)

            # aexp_function.initparfnc(Gt1_3_divsigma, timeslice=7)
            bootfit_state1_divsigma, redchisq_1_divsigma = fit_value3(
                Gt1_3_divsigma, ratio_t_range, aexp_function, norm=1
            )
            bootfit_state2_divsigma, redchisq_2_divsigma = fit_value3(
                Gt2_3_divsigma, ratio_t_range, aexp_function, norm=1
            )

            order3_states_fit_divsigma = np.array(
                [bootfit_state1_divsigma, bootfit_state2_divsigma]
            )

        print(redchisq0)
        print(redchisq1)
        print(redchisq2)
        print(redchisq3)

        # ==================================================
        # Save the data
        print("Save the data")
        fitparams = {
            "lambdas": lmb_val,
            "time_choice": time_choice,
            "delta_t": delta_t,
            # "corr_matrices": np.array([matrix_1, matrix_2, matrix_3, matrix_4]),
            # "order0_corrs": np.array([Gt1_0, Gt2_0]),
            "order0_states_fit": np.array([bootfit_state1_0, bootfit_state2_0]),
            "order0_fit": bootfit0,
            "order0_eval_left": eval_left0,
            "order0_eval_right": eval_right0,
            "order0_evec_left": evec_left0,
            "order0_evec_right": evec_right0,
            "red_chisq0": redchisq0,
            # "order1_corrs": np.array([Gt1_1, Gt2_1]),
            "order1_states_fit": np.array([bootfit_state1_1, bootfit_state2_1]),
            "order1_fit": bootfit1,
            "order1_eval_left": eval_left1,
            "order1_eval_right": eval_right1,
            "order1_evec_left": evec_left1,
            "order1_evec_right": evec_right1,
            "red_chisq1": redchisq1,
            # "order2_corrs": np.array([Gt1_2, Gt2_2]),
            "order2_states_fit": np.array([bootfit_state1_2, bootfit_state2_2]),
            "order2_fit": bootfit2,
            "order2_eval_left": eval_left2,
            "order2_eval_right": eval_right2,
            "order2_evec_left": evec_left2,
            "order2_evec_right": evec_right2,
            "red_chisq2": redchisq2,
            # "order3_corrs": np.array([Gt1_3, Gt2_3]),
            "order3_states_fit": np.array([bootfit_state1_3, bootfit_state2_3]),
            "order3_fit": bootfit3,
            "order3_eval_left": eval_left3,
            "order3_eval_right": eval_right3,
            "order3_evec_left": evec_left3,
            "order3_evec_right": evec_right3,
            "red_chisq3": redchisq3,
            "order3_states_fit_divsigma": order3_states_fit_divsigma,
            "weighted_energy_nucl": weighted_energy_nucl,
            "weighted_energy_nucldivsigma": weighted_energy_nucldivsigma,
            "weighted_energy_sigma": weighted_energy_sigma,
        }
        fitlist.append(fitparams)
        print("Saved the data")

        # ==================================================
        print("plotting")
        if plotting:
            plotting_script_all(
                matrix_1 / 1e39,
                matrix_2 / 1e39,
                matrix_3 / 1e39,
                matrix_4 / 1e39,
                lmb_val,
                plotdir,
                name="_l" + str(lmb_val),
                show=False,
            )
            plotting_script_all_N(
                matrix_1 / 1e39,
                matrix_2 / 1e39,
                matrix_3 / 1e39,
                matrix_4 / 1e39,
                lmb_val,
                plotdir,
                name="_l" + str(lmb_val),
                show=False,
            )
            plotting_script_diff_2(
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
        print("plotted")

    # ----------------------------------------------------------------------
    # Save the fit data to a pickle file
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "wb",
    ) as file_out:
        pickle.dump(fitlist, file_out)


if __name__ == "__main__":
    main()
