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
    xlim = 25
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


def fit_loop_weighted(
    datadir,
    plotdir,
    config,
    time_limits_nucl,
    time_limits_sigma,
    time_limits_nucldivsigma,
):
    # ============================================================
    # Nucleon correlators
    with open(datadir / (f"time_window_loop_nucl_Aexp.pkl"), "rb") as file_in:
        fitlist_nucl_1exp = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_nucl_Twoexp.pkl"), "rb") as file_in:
        fitlist_nucl_2exp = pickle.load(file_in)

    # ============================================================
    # Sigma correlators
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
        plot=True,
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
        plot=True,
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
        plot=True,
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
    # plotdir = Path(config["analysis_dir"]) / Path("plots")
    # datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir = PROJECT_BASE_DIRECTORY / Path("data/plots") / Path(config["name"])
    datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
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

    # ============================================================
    # time_limits_nucl = np.array([[11, 20], [4, 8]])
    # time_limits_sigma = np.array([[12, 20], [4, 9]])
    time_limits_nucl = np.array([[10, 15], [4, 4]])
    time_limits_sigma = np.array([[10, 15], [4, 4]])
    time_limits_nucldivsigma = np.array([[1, 15], [2, 2]])
    (
        weighted_energy_nucl,
        weighted_energy_sigma,
        weighted_energy_nucldivsigma,
        chosen_nucl_fit,
        chosen_sigma_fit,
        chosen_nucldivsigma_fit,
    ) = fit_loop_weighted(
        datadir,
        plotdir,
        config,
        time_limits_nucl,
        time_limits_sigma,
        time_limits_nucldivsigma,
    )
    # ============================================================

if __name__ == "__main__":
    main()
