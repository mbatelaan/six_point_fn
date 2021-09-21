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
# _colors = ["r", "g", "b", "k", "y", "m", "k", "k"]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]
# sys.stdout = open("output.txt", "wt")
# From the theta tuning:
m_N = 0.4179255
m_S = 0.4641829


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


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


def gevp(corr_matrix, time_choice=10, delta_t=1, name="", show=None):
    """Solve the GEVP for a given correlation matrix

    corr_matrix has the matrix indices as the first two, then the bootstrap and then the time index
    time_choice is the timeslice on which the GEVP will be set
    delta_t is the size of the time evolution which will be used to solve the GEVP
    """
    # time_choice = 10
    # delta_t = 1
    mat_0 = np.average(corr_matrix[:, :, :, time_choice], axis=2)
    mat_1 = np.average(corr_matrix[:, :, :, time_choice + delta_t], axis=2)

    # wl, vl = np.linalg.eig(mat_0.T)
    # wr, vr = np.linalg.eig(mat_0)
    wl, vl = np.linalg.eig(np.matmul(mat_1, np.linalg.inv(mat_0)).T)
    wr, vr = np.linalg.eig(np.matmul(np.linalg.inv(mat_0), mat_1))
    # print(wl, vl)
    # print(wr, vr)

    Gt1 = np.einsum("i,ijkl,j->kl", vl[:, 0], corr_matrix, vr[:, 0])
    # print(np.shape(Gt1))
    Gt2 = np.einsum("i,ijkl,j->kl", vl[:, 1], corr_matrix, vr[:, 1])
    # print(np.shape(Gt2))

    if show:
        stats.ploteffmass(Gt1, "eig_1" + name, plotdir, show=True)
        stats.ploteffmass(Gt2, "eig_2" + name, plotdir, show=True)

    # print(f"{np.shape(mat)=}")
    # print(mat)
    # wl, vl = np.linalg.eig(mat.T)
    # wr, vr = np.linalg.eig(mat)
    # print(wl)
    # print(vl)
    # print(wr, vr)
    return Gt1, Gt2


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
    pypl.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    pypl.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    pypl.xlabel("$t/a$")
    pypl.legend(fontsize="x-small")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    pypl.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def fit_value(diffG, t_range):
    """Fit a constant function to the diffG correlator

    diffG is a correlator with tht bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    data_set = diffG[:,t_range]
    diffG_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    popt_avg, pcov_avg = curve_fit(ff.constant, t_range, diffG_avg, sigma=covmat)
    chisq = ff.chisqfn(*popt_avg, ff.constant, t_range, diffG_avg, np.linalg.inv(covmat))
    redchisq = chisq / len(t_range)
    # print("popt", popt_avg)
    # print("pcov", pcov_avg)
    bootfit = []
    for iboot, values in enumerate(diffG):
        popt, pcov = curve_fit(ff.constant, t_range, values[t_range], sigma=diag_sigma)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    # print(popt_avg)
    # print(np.average(bootfit))
    return bootfit, redchisq


if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    nboot = 200  # 700
    nbin = 1  # 10

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir.yaml"
    print(config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # TODO: Set up a defaults.yaml file for when there is no input file
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    momenta = ["mass"]
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    files = (
         pickledir
        / Path(
            "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    print(pickledir)
    print(pickledir
        / Path(
            "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
        ))

    # print("list1",[i for i in files])
    # print("list2",list(files))

    conf_num_list = np.array([int("".join(filter(str.isdigit, l.name))) for l in list(files)])
    print(conf_num_list)
    # conf_num_list = [50]
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    barspec_name = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    G2_unpert_qp100_nucl = read_pickle(unpertfile_nucleon_pos, nboot=pars.nboot, nbin=1)
    G2_unpert_q000_sigma = read_pickle(unpertfile_sigma, nboot=pars.nboot, nbin=1)
    unpert_ratio = G2_unpert_qp100_nucl/G2_unpert_q000_sigma
    t_range0 = np.arange(4, 9)
    unpert_fit, redchisq = fit_value(unpert_ratio[:,:,0], t_range0)
    # print(unpert_fit)
    # print(np.shape(unpert_fit))

    ### ----------------------------------------------------------------------
    ### SD
    filelist_SD1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_SD3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )

    G2_q100_SD_lmb = read_pickle(filelist_SD1, nboot=pars.nboot, nbin=1)
    G2_q100_SD_lmb3 = read_pickle(filelist_SD3, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### DS
    filelist_DS1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )
    filelist_DS3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )

    G2_q100_DS_lmb = read_pickle(filelist_DS1, nboot=pars.nboot, nbin=1)
    G2_q100_DS_lmb3 = read_pickle(filelist_DS3, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### DD
    filelist_DD2 = pickledir / Path(
        # "baryon-3pt_DD_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040//lp0lp0__lp0lp0/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_DD4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )

    G2_q100_DD_lmb2 = read_pickle(filelist_DD2, nboot=pars.nboot, nbin=1)
    G2_q100_DD_lmb4 = read_pickle(filelist_DD4, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### SS
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )

    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=1)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=1)
    ### ----------------------------------------------------------------------

    order0_fit = []
    order1_fit = []
    order2_fit = []
    order3_fit = []
    red_chisq_list = [[],[],[],[]]

    # lambdas = [0.005, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    # lambdas = [0.005, 0.04, 0.16]
    # lambdas = np.array([
    #     0.001,
    #     0.005,
    #     0.01,
    #     0.02,
    #     0.04,
    #     0.06,
    #     0.08,
    #     0.10,
    #     0.15,
    #     0.2,
    #     0.25,
    #     0.3,
    #     0.4,
    #     0.5,
    # ])
    # lambdas = np.array([
    #     0.001,
    #     0.004,
    #     0.008,
    #     0.016,
    #     0.032,
    #     0.064,
    #     0.128,
    #     0.256,
    #     0.512,
    # ])
    # lambdas = np.linspace(0.12,0.16,20)
    # lambdas = np.linspace(0,0.16,10)[1:]
    # lambdas = np.linspace(0,0.04,30)[1:]
    lambdas = np.linspace(0,0.16,30)[1:]
    plotting = False

    print("\n HERE0\n")
    
    for lmb_val in lambdas:
        # Construct a correlation matrix for each order in lambda (skipping order 0)
        matrix_1 = np.array(
            [
                [G2_unpert_qp100_nucl[:, :, 0], lmb_val * G2_q100_DS_lmb[:, :, 0]],
                [lmb_val * G2_q100_SD_lmb[:, :, 0], G2_unpert_q000_sigma[:, :, 0]],
            ]
        )
        matrix_2 = np.array(
            [
                [
                    G2_unpert_qp100_nucl[:, :, 0]
                    + lmb_val ** 2 * G2_q100_DD_lmb2[:, :, 0],
                    lmb_val * G2_q100_DS_lmb[:, :, 0],
                ],
                [
                    lmb_val * G2_q100_SD_lmb[:, :, 0],
                    G2_unpert_q000_sigma[:, :, 0]
                    + lmb_val ** 2 * G2_q000_SS_lmb2[:, :, 0],
                ],
            ]
        )
        matrix_3 = np.array(
            [
                [
                    G2_unpert_qp100_nucl[:, :, 0]
                    + lmb_val ** 2 * G2_q100_DD_lmb2[:, :, 0],
                    lmb_val * G2_q100_DS_lmb[:, :, 0]
                    + lmb_val ** 3 * G2_q100_DS_lmb3[:, :, 0],
                ],
                [
                    lmb_val * G2_q100_SD_lmb[:, :, 0]
                    + lmb_val ** 3 * G2_q100_SD_lmb3[:, :, 0],
                    G2_unpert_q000_sigma[:, :, 0]
                    + lmb_val ** 2 * G2_q000_SS_lmb2[:, :, 0],
                ],
            ]
        )
        matrix_4 = np.array(
            [
                [
                    G2_unpert_qp100_nucl[:, :, 0]
                    + (lmb_val ** 2) * G2_q100_DD_lmb2[:, :, 0]
                    + (lmb_val ** 4) * G2_q100_DD_lmb4[:, :, 0],
                    lmb_val * G2_q100_DS_lmb[:, :, 0]
                    + (lmb_val ** 3) * G2_q100_DS_lmb3[:, :, 0],
                ],
                [
                    lmb_val * G2_q100_SD_lmb[:, :, 0]
                    + (lmb_val ** 3) * G2_q100_SD_lmb3[:, :, 0],
                    G2_unpert_q000_sigma[:, :, 0]
                    + (lmb_val ** 2) * G2_q000_SS_lmb2[:, :, 0]
                    + (lmb_val ** 4) * G2_q000_SS_lmb4[:, :, 0],
                ],
            ]
        )
        ### ----------------------------------------------------------------------
        # Test the magnitude of the order lambda^4 correlator
        # test1 = np.average(G2_q100_DD_lmb4[:, :, 0], axis=0)
        # test2 = np.average(G2_q100_DD_lmb2[:, :, 0], axis=0)
        # test3 = np.average(G2_q000_SS_lmb4[:, :, 0], axis=0)
        # print("\n\n", test1, test2, test3, "\n\n")
        ### ----------------------------------------------------------------------
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

        print(f"\n HERE {lmb_val}\n")

        t_range = np.arange(4, 9)
        time_choice = 2
        delta_t = 2

        Gt1_1, Gt2_1 = gevp(matrix_1, time_choice, delta_t, name="_test", show=False)
        ratio1 = Gt1_1/Gt2_1
        effmassdata_1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
        # effmassdata_1 = stats.bs_effmass(Gt1_1, time_axis=1, spacing=1)
        # effmassdata_2 = stats.bs_effmass(Gt2_1, time_axis=1, spacing=1)
        # diffG1 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        diffG1 = effmassdata_1 / 2
        # diffG1_avg = np.average(diffG1, axis=0)[t_range]
        # covmat = np.diag(diffG1[t_range])
        # popt_1, pcov_1 = curve_fit(ff.constant, t_range, diffG1_avg, sigma=covmat)
        # print(popt_1)
        bootfit1, redchisq1 = fit_value(diffG1, t_range)
        order0_fit.append(bootfit1[:, 0])
        red_chisq_list[0].append(redchisq1)
        print(redchisq1)

        Gt1_2, Gt2_2 = gevp(matrix_2, time_choice, delta_t, name="_test", show=False)
        effmassdata_1 = stats.bs_effmass(Gt1_2, time_axis=1, spacing=1)
        effmassdata_2 = stats.bs_effmass(Gt2_2, time_axis=1, spacing=1)
        diffG2 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        bootfit2, redchisq2 = fit_value(diffG2, t_range)
        order1_fit.append(bootfit2[:, 0])
        red_chisq_list[1].append(redchisq2)

        Gt1_3, Gt2_3 = gevp(matrix_3, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_3 = stats.bs_effmass(Gt1_3, time_axis=1, spacing=1)
        effmassdata_2_3 = stats.bs_effmass(Gt2_3, time_axis=1, spacing=1)
        diffG3 = np.abs(effmassdata_1_3 - effmassdata_2_3) / 2  # / lmb_val
        bootfit3, redchisq3 = fit_value(diffG3, t_range)
        order2_fit.append(bootfit3[:, 0])
        red_chisq_list[2].append(redchisq3)

        Gt1_4, Gt2_4 = gevp(matrix_4, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_4 = stats.bs_effmass(Gt1_4, time_axis=1, spacing=1)
        effmassdata_2_4 = stats.bs_effmass(Gt2_4, time_axis=1, spacing=1)
        diffG4 = np.abs(effmassdata_1_4 - effmassdata_2_4) / 2  # / lmb_val
        bootfit4, redchisq4 = fit_value(diffG4, t_range)
        order3_fit.append(bootfit4[:, 0])
        red_chisq_list[3].append(redchisq4)

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

    print(red_chisq_list)
    red_chisq_list = np.array(red_chisq_list)
    print(red_chisq_list)

    all_data = {
        "lambdas" : np.array(lambdas),
        "order0_fit" : np.array(order0_fit),
        "order1_fit" : np.array(order1_fit),
        "order2_fit" : np.array(order2_fit),
        "order3_fit" : np.array(order3_fit),
        "redchisq" : red_chisq_list,
        "time_choice" : np.array(time_choice),
        "delta_t" : np.array(delta_t)
    }
    
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
