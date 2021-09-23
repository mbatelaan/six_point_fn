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
from common import read_correlators
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


def plot_correlator(
    corr, plotname, plotdir, ylim=None, log=False, xlim=30, fitparam=None, ylabel=None
):
    """Plot the correlator"""
    time = np.arange(0, np.shape(corr)[1])
    yavg = np.average(corr, axis=0)
    ystd = np.std(corr, axis=0)

    pypl.figure(figsize=(8, 6))
    pypl.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    if log:
        pypl.semilogy()
    if fitparam:
        pypl.plot(
            fitparam["x"],
            np.average(fitparam["y"], axis=0),
            label=fitparam["label"],
        )
        pypl.fill_between(
            fitparam["x"],
            np.average(fitparam["y"], axis=0) - np.std(fitparam["y"], axis=0),
            np.average(fitparam["y"], axis=0) + np.std(fitparam["y"], axis=0),
            alpha=0.3,
        )
        pypl.legend()

    pypl.xlabel(r"$\textrm{t/a}$", labelpad=14, fontsize=18)
    pypl.ylabel(ylabel, labelpad=5, fontsize=18)
    pypl.xlim(0, xlim)
    pypl.ylim(ylim)
    pypl.grid(True, alpha=0.4)
    # metadata["Title"] = plotname.split("/")[-1][:-4]
    metadata["Title"] = plotname
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=metadata)
    # pypl.show()
    pypl.close()
    return




def plotting_script(corr_matrix, Gt1, Gt2, name="", show=False):
    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(Gt1)[1])
    efftime = time[:-spacing] + 0.5
    effmassdata_1 = stats.bs_effmass(Gt1, time_axis=1, spacing=spacing)
    yeffavg_1 = np.average(effmassdata_1, axis=0)
    yeffstd_1 = np.std(effmassdata_1, axis=0)
    effmassdata_2 = stats.bs_effmass(Gt2, time_axis=1, spacing=spacing)
    yeffavg_2 = np.average(effmassdata_2, axis=0)
    yeffstd_2 = np.std(effmassdata_2, axis=0)
    f, axs = pypl.subplots(3, 2, figsize=(9, 12), sharex=True, sharey=True)
    for i in range(4):
        # print(int(i / 2), i % 2)
        effmassdata = stats.bs_effmass(
            corr_matrix[int(i / 2)][i % 2], time_axis=1, spacing=spacing
        )
        yeffavg = np.average(effmassdata, axis=0)
        yeffstd = np.std(effmassdata, axis=0)

        axs[int(i / 2)][i % 2].errorbar(
            efftime[:xlim],
            yeffavg[:xlim],
            yeffstd[:xlim],
            capsize=4,
            elinewidth=1,
            color="b",
            fmt="s",
            markerfacecolor="none",
        )
    axs[2][0].errorbar(
        efftime[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    axs[2][1].errorbar(
        efftime[:xlim],
        yeffavg_2[:xlim],
        yeffstd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    pypl.setp(axs, xlim=(0, xlim), ylim=(0, 1))
    pypl.savefig(plotdir / ("corr_matrix" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script3(corr_matrix, Gt1, Gt2, name="", show=False):
    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(Gt1)[1])
    efftime = time[:-spacing] + 0.5
    effmassdata_1 = stats.bs_effmass(Gt1, time_axis=1, spacing=spacing)
    yeffavg_1 = np.average(effmassdata_1, axis=0)
    yeffstd_1 = np.std(effmassdata_1, axis=0)
    effmassdata_2 = stats.bs_effmass(Gt2, time_axis=1, spacing=spacing)
    yeffavg_2 = np.average(effmassdata_2, axis=0)
    yeffstd_2 = np.std(effmassdata_2, axis=0)
    f, axs = pypl.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    for i in range(4):
        print(int(i / 2), i % 2)
        effmassdata = stats.bs_effmass(
            corr_matrix[int(i / 2)][i % 2], time_axis=1, spacing=spacing
        )
        yeffavg = np.average(effmassdata, axis=0)
        yeffstd = np.std(effmassdata, axis=0)

        axs[int(i / 2)][i % 2].errorbar(
            efftime[:xlim],
            yeffavg[:xlim],
            yeffstd[:xlim],
            capsize=4,
            elinewidth=1,
            color="b",
            fmt="s",
            markerfacecolor="none",
        )
    # axs[2][0].errorbar(
    #     efftime[:xlim],
    #     yeffavg_1[:xlim],
    #     yeffstd_1[:xlim],
    #     capsize=4,
    #     elinewidth=1,
    #     color="b",
    #     fmt="s",
    #     markerfacecolor="none",
    # )
    # axs[2][1].errorbar(
    #     efftime[:xlim],
    #     yeffavg_2[:xlim],
    #     yeffstd_2[:xlim],
    #     capsize=4,
    #     elinewidth=1,
    #     color="b",
    #     fmt="s",
    #     markerfacecolor="none",
    # )
    pypl.setp(axs, xlim=(0, xlim), ylim=(0, 1))
    pypl.savefig(plotdir / ("corr_matrix" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_nucl(corr_matrix, corr_matrix1, corr_matrix2, name="", show=False):
    spacing = 2
    xlim = 22
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
        label=r"$\mathcal{O}(\lambda^0)$,",
    )
    yavg = np.average(corr_matrix1[0][0], axis=0)
    ystd = np.std(corr_matrix1[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.3,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix2[0][0], axis=0)
    ystd = np.std(corr_matrix2[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.6,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )
    pypl.semilogy()
    pypl.legend(fontsize="small")
    pypl.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    pypl.title("$\lambda=0.04$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_nucl_nucl" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_sigma(corr_matrix, corr_matrix1, corr_matrix2, name="", show=False):
    spacing = 2
    xlim = 22
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
        label=r"$\mathcal{O}(\lambda^0)$,",
    )
    yavg = np.average(corr_matrix1[1][1], axis=0)
    ystd = np.std(corr_matrix1[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.3,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix2[1][1], axis=0)
    ystd = np.std(corr_matrix2[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.6,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )
    pypl.semilogy()
    pypl.legend(fontsize="small")
    pypl.ylabel(r"$G_{\Sigma\Sigma}(t;\vec{p}=(0,0,0))$")
    pypl.title("$\lambda=0.04$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_sigma_sigma" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_nucl_sigma(corr_matrix, corr_matrix1, name="", show=False):
    spacing = 2
    xlim = 22
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[0][1], axis=0)
    ystd = np.std(corr_matrix[0][1], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1)$,",
    )
    yavg = np.average(corr_matrix1[0][1], axis=0)
    ystd = np.std(corr_matrix1[0][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.3,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )
    pypl.semilogy()
    pypl.legend(fontsize="small")
    pypl.ylabel(r"$G_{n\Sigma}(t;\vec{p}=(0,0,0))$")
    pypl.title("$\lambda=0.04$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_nucl_sigma" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_sigma_nucl(corr_matrix, corr_matrix1, name="", show=False):
    spacing = 2
    xlim = 22
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[1][0], axis=0)
    ystd = np.std(corr_matrix[1][0], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1)$,",
    )
    yavg = np.average(corr_matrix1[1][0], axis=0)
    ystd = np.std(corr_matrix1[1][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.3,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )
    pypl.semilogy()
    pypl.legend(fontsize="small")
    pypl.ylabel(r"$G_{\Sigma n}(t;\vec{p}=(1,0,0))$")
    pypl.title("$\lambda=0.04$")
    # pypl.xlabel(r"$\textrm{t/a}$")
    pypl.xlabel(r"$t/a$")
    pypl.savefig(plotdir / ("comp_plot_sigma_nucl" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


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


def plotting_script2(diffG, name="", show=False):
    spacing = 2
    xlim = 17
    time = np.arange(0, np.shape(diffG)[1])
    efftime = time[:-spacing] + 0.5
    yeffavg_1 = np.average(diffG, axis=0)
    yeffstd_1 = np.std(diffG, axis=0)
    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    axs.errorbar(
        efftime[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.setp(axs, xlim=(0, xlim), ylim=(-1, 4))
    pypl.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    pypl.xlabel("$t/a$")
    pypl.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def plotting_script_diff(diffG1, diffG2, diffG3, diffG4, lmb_val, name="", show=False):
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

    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.setp(axs, xlim=(0, xlim), ylim=(-1, 4))
    pypl.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    pypl.xlabel("$t/a$")
    pypl.legend(fontsize="x-small")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    pypl.savefig(plotdir / ("diff_G" + name + ".pdf"))
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
    print("Reading directories from: ",config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # TODO: Set up a defaults.yaml file for when there is no input file
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    G2_nucl, G2_sigm = read_correlators(pars, pickledir, mom_strings)

    # lambdas = np.linspace(0.12,0.16,20)
    # lambdas = np.linspace(0,0.16,10)[1:]
    # lambdas = np.linspace(0,0.04,30)[1:]
    lambdas = np.linspace(0,0.16,30) #[1:]
    t_range = np.arange(4, 9)
    time_choice = 2
    delta_t = 2
    plotting = True

    # order0_fit = []
    # order1_fit = []
    # order2_fit = []
    # order3_fit = []
    # red_chisq_list = [[],[],[],[]]

    order0_fit = np.zeros((len(lambdas), pars.nboot))
    order1_fit = np.zeros((len(lambdas), pars.nboot))
    order2_fit = np.zeros((len(lambdas), pars.nboot))
    order3_fit = np.zeros((len(lambdas), pars.nboot))
    red_chisq_list = np.zeros((4,len(lambdas)))

    for i, lmb_val in enumerate(lambdas):
        print(f"Lambda = {lmb_val}\n")
        # Construct a correlation matrix for each order in lambda(skipping order 0)
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(G2_nucl, G2_sigm, lmb_val)

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

        Gt1_1, Gt2_1 = gevp(matrix_1, time_choice, delta_t, name="_test", show=False)
        ratio1 = Gt1_1/Gt2_1
        effmassdata_1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
        # effmassdata_1 = stats.bs_effmass(Gt1_1, time_axis=1, spacing=1)
        # effmassdata_2 = stats.bs_effmass(Gt2_1, time_axis=1, spacing=1)
        # diffG1 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        diffG1 = effmassdata_1 / 2
        bootfit1, redchisq1 = fit_value(diffG1, t_range)
        order0_fit[i] = bootfit1[:, 0]
        red_chisq_list[0,i] = redchisq1
        # order0_fit[i] = bootfit1[:, 0]
        # red_chisq_list[0,i] = redchisq1

        Gt1_2, Gt2_2 = gevp(matrix_2, time_choice, delta_t, name="_test", show=False)
        effmassdata_1 = stats.bs_effmass(Gt1_2, time_axis=1, spacing=1)
        effmassdata_2 = stats.bs_effmass(Gt2_2, time_axis=1, spacing=1)
        diffG2 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        bootfit2, redchisq2 = fit_value(diffG2, t_range)
        order1_fit[i] = bootfit1[:, 0]
        red_chisq_list[1,i] = redchisq2
        # order1_fit.append(bootfit2[:, 0])
        # red_chisq_list[1].append(redchisq2)

        Gt1_3, Gt2_3 = gevp(matrix_3, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_3 = stats.bs_effmass(Gt1_3, time_axis=1, spacing=1)
        effmassdata_2_3 = stats.bs_effmass(Gt2_3, time_axis=1, spacing=1)
        diffG3 = np.abs(effmassdata_1_3 - effmassdata_2_3) / 2  # / lmb_val
        bootfit3, redchisq3 = fit_value(diffG3, t_range)
        order2_fit[i] = bootfit1[:, 0]
        red_chisq_list[2,i] = redchisq3
        # order2_fit.append(bootfit3[:, 0])
        # red_chisq_list[2].append(redchisq3)

        Gt1_4, Gt2_4 = gevp(matrix_4, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_4 = stats.bs_effmass(Gt1_4, time_axis=1, spacing=1)
        effmassdata_2_4 = stats.bs_effmass(Gt2_4, time_axis=1, spacing=1)
        diffG4 = np.abs(effmassdata_1_4 - effmassdata_2_4) / 2  # / lmb_val
        bootfit4, redchisq4 = fit_value(diffG4, t_range)
        order3_fit[i] = bootfit3[:, 0]
        red_chisq_list[3,i] = redchisq4
        # order3_fit.append(bootfit4[:, 0])
        # red_chisq_list[3].append(redchisq4)

        if plotting:
            plotting_script_diff_2(
                diffG1,
                diffG2,
                diffG3,
                diffG4,
                [bootfit1, bootfit2, bootfit3, bootfit4],
                t_range,
                lmb_val,
                name="_l" + str(lmb_val) + "_all",
                show=False,
            )

    print(f"\n\n\n END of LOOP \n\n")

    all_data = {
        "lambdas" : np.array([lmb_val]),
        "order0_fit" : order0_fit, 
        "order1_fit" : order1_fit,
        "order2_fit" : order2_fit,
        "order3_fit" : order3_fit,
        "redchisq" : red_chisq_list,
        "time_choice" : time_choice_range,
        "delta_t" : delta_t_range
    }

    # all_data = {
    #     "lambdas" : np.array(lambdas),
    #     "order0_fit" : np.array(order0_fit),
    #     "order1_fit" : np.array(order1_fit),
    #     "order2_fit" : np.array(order2_fit),
    #     "order3_fit" : np.array(order3_fit),
    #     "redchisq" : red_chisq_list,
    #     "time_choice" : np.array(time_choice),
    #     "delta_t" : np.array(delta_t)
    # }
    
    with open(datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"), "wb") as file_out:
        pickle.dump(all_data, file_out)
        # pickle.dump([lambdas,order0_fit, order1_fit,order2_fit,order3_fit, time_choice, delta_t],file_out)
        # pickle.dump(np.array([lambdas,order0_fit, order1_fit,order2_fit,order3_fit],dtype=object),file_out)
    
    # print(red_chisq_list[0])
    # print(red_chisq_list[0].ptp())
    # scaled_z0 = (red_chisq_list[0] - red_chisq_list[0].min()) / red_chisq_list[0].ptp()
    # print('scaled_z0', scaled_z0)
    # colors_0 = [[0., 0., 0., i] for i in scaled_z0]
    # print('colors_0', colors_0)

    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        lambdas,
        np.average(order0_fit, axis=1),
        np.std(order0_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.001,
        np.average(order1_fit, axis=1),
        np.std(order1_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.002,
        np.average(order2_fit, axis=1),
        np.std(order2_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.003,
        np.average(order3_fit, axis=1),
        np.std(order3_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    # pypl.errorbar(
    #     lambdas,
    #     np.average(order3_fit, axis=1),
    #     np.std(order3_fit, axis=1),
    #     fmt="s",
    #     label="order 4",
    # )
    pypl.legend(fontsize="x-small")
    pypl.xlim(-0.01, 0.22)
    pypl.ylim(0, 0.2)
    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={time_choice}, \Delta t={delta_t}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))
    # pypl.show()
