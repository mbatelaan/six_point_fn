import numpy as np
from pathlib import Path
import pickle
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as pypl
from matplotlib import rcParams

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff
from analysis.evxptreaders import evxptdata

from params import params


_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
_colors = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
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
        print(f"{np.shape(data)=}")
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
    diffG_avg = np.average(diffG, axis=0)[t_range]
    covmat = np.diag(np.std(diffG, axis=0)[t_range] ** 2)
    # covmat = np.cov(diffG[t_range])
    popt_avg, pcov_avg = curve_fit(ff.constant, t_range, diffG_avg, sigma=covmat)
    bootfit = []
    for iboot, values in enumerate(diffG):
        popt, pcov = curve_fit(ff.constant, t_range, values[t_range], sigma=covmat)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    print(popt_avg)
    print(np.average(bootfit))
    return bootfit


if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    nboot = 200  # 700
    nbin = 1  # 10
    pickledir = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn2/pickle/"
    )
    pickledir2 = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn4/pickle/"
    )
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn2/plots/")
    datadir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn2/data/")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    momenta = ["mass"]
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    files = (
        pickledir
        / Path(
            "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    conf_num_list = [int("".join(filter(str.isdigit, l.name))) for l in list(files)]
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    barspec_name = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    unpertfile_sigma = pickledir / Path(
        "baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    G2_unpert_qp100_nucl = read_pickle(unpertfile_nucleon_pos, nboot=pars.nboot, nbin=1)
    G2_unpert_q000_sigma = read_pickle(unpertfile_sigma, nboot=pars.nboot, nbin=1)
    unpert_ratio = G2_unpert_qp100_nucl/G2_unpert_q000_sigma
    t_range0 = np.arange(4, 9)
    unpert_fit = fit_value(unpert_ratio[:,:,0], t_range0)
    print(unpert_fit)
    print(np.shape(unpert_fit))

    ### ----------------------------------------------------------------------
    ### SD
    filelist_SD1 = pickledir / Path(
        "baryon-3pt_SD_lmb_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_SD3 = pickledir / Path(
        "baryon-3pt_SD_lmb3_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )

    G2_q100_SD_lmb = read_pickle(filelist_SD1, nboot=pars.nboot, nbin=1)
    G2_q100_SD_lmb3 = read_pickle(filelist_SD3, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### DS
    filelist_DS1 = pickledir / Path(
        "baryon-3pt_DS_lmb_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )
    filelist_DS3 = pickledir / Path(
        "baryon-3pt_DS_lmb3_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )

    G2_q100_DS_lmb = read_pickle(filelist_DS1, nboot=pars.nboot, nbin=1)
    G2_q100_DS_lmb3 = read_pickle(filelist_DS3, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### DD
    filelist_DD2 = pickledir / Path(
        # "baryon-3pt_DD_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        "baryon-3pt_DD_lmb2_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_DD4 = pickledir2 / Path(
        "baryon-3pt_DD_lmb4_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )

    G2_q100_DD_lmb2 = read_pickle(filelist_DD2, nboot=pars.nboot, nbin=1)
    G2_q100_DD_lmb4 = read_pickle(filelist_DD4, nboot=pars.nboot, nbin=1)

    ### ----------------------------------------------------------------------
    ### SS
    filelist_SS2 = pickledir / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    filelist_SS4 = pickledir / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
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
    lambdas = np.linspace(0,0.5)
    plotting = False
    
    for lmb_val in lambdas:
        print(f"\n\n\n {lmb_val}")
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
        print(f"{np.shape(matrix_1)=}")
        test1 = np.average(G2_q100_DD_lmb4[:, :, 0], axis=0)
        test2 = np.average(G2_q100_DD_lmb2[:, :, 0], axis=0)
        test3 = np.average(G2_q000_SS_lmb4[:, :, 0], axis=0)
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

        t_range = np.arange(4, 9)
        time_choice = 13
        delta_t = 1

        Gt1_1, Gt2_1 = gevp(matrix_1, time_choice, delta_t, name="_test", show=False)
        effmassdata_1 = stats.bs_effmass(Gt1_1, time_axis=1, spacing=1)
        effmassdata_2 = stats.bs_effmass(Gt2_1, time_axis=1, spacing=1)
        diffG1 = (effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        # diffG1_avg = np.average(diffG1, axis=0)[t_range]
        # covmat = np.diag(diffG1[t_range])
        # popt_1, pcov_1 = curve_fit(ff.constant, t_range, diffG1_avg, sigma=covmat)
        # print(popt_1)
        bootfit1 = fit_value(diffG1, t_range)
        print(f"{np.shape(bootfit1[:,0])=}")
        order0_fit.append(bootfit1[:, 0])

        Gt1_2, Gt2_2 = gevp(matrix_2, time_choice, delta_t, name="_test", show=False)
        effmassdata_1 = stats.bs_effmass(Gt1_2, time_axis=1, spacing=1)
        effmassdata_2 = stats.bs_effmass(Gt2_2, time_axis=1, spacing=1)
        diffG2 = (effmassdata_1 - effmassdata_2) / 2  # / lmb_val
        bootfit2 = fit_value(diffG2, t_range)
        order1_fit.append(bootfit2[:, 0])
        # diffG2_avg = np.average(diffG2, axis=0)[t_range]
        # covmat = np.diag(diffG2[t_range])
        # popt_2, pcov_2 = curve_fit(ff.constant, t_range, diffG2_avg, sigma=covmat)
        # print(popt_2)

        Gt1_3, Gt2_3 = gevp(matrix_3, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_3 = stats.bs_effmass(Gt1_3, time_axis=1, spacing=1)
        effmassdata_2_3 = stats.bs_effmass(Gt2_3, time_axis=1, spacing=1)
        diffG3 = (effmassdata_1_3 - effmassdata_2_3) / 2  # / lmb_val
        bootfit3 = fit_value(diffG3, t_range)
        order2_fit.append(bootfit3[:, 0])
        # diffG3_avg = np.average(diffG3, axis=0)[t_range]
        # covmat = np.diag(diffG3[t_range])
        # popt_3, pcov_3 = curve_fit(ff.constant, t_range, diffG3_avg, sigma=covmat)
        # print(popt_3)

        Gt1_4, Gt2_4 = gevp(matrix_4, time_choice, delta_t, name="_test", show=False)
        effmassdata_1_4 = stats.bs_effmass(Gt1_4, time_axis=1, spacing=1)
        effmassdata_2_4 = stats.bs_effmass(Gt2_4, time_axis=1, spacing=1)
        diffG4 = (effmassdata_1_4 - effmassdata_2_4) / 2  # / lmb_val
        bootfit4 = fit_value(diffG4, t_range)
        order3_fit.append(bootfit4[:, 0])
        # print(np.average(diffG4, axis=0)[t_range])
        # print("\n\n\n", np.average(bootfit4), np.std(bootfit4), "\n\n\n")
        # diffG4_avg = np.average(diffG4, axis=0)[t_range]
        # covmat = np.diag(diffG4[t_range])
        # popt_4, pcov_4 = curve_fit(ff.constant, t_range, diffG4_avg, sigma=covmat)
        # print(popt_4)

        # plotting_script_diff(
        #     diffG1,
        #     diffG2,
        #     diffG3,
        #     diffG3,
        #     lmb_val,
        #     name="_l" + str(lmb_val) + "_all",
        #     show=True,
        # )

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

    # print("\n\n")
    # print(np.shape(order0_fit))
    # print(np.average(order0_fit, axis=1))
    # print(np.std(order0_fit, axis=1))
    # print("\n\n")
    # print(np.average(order1_fit, axis=1))
    # print(np.average(order2_fit, axis=1))
    # print(np.average(order3_fit, axis=1))

    with open(datadir / ("lambda_dep.pkl"), "wb") as file_out:
        pickle.dump(np.array([lambdas,order0_fit, order1_fit,order2_fit,order3_fit],dtype=object),file_out)
    

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
    # pypl.plot(lambdas, np.average(order0_fit, axis=1))
    # pypl.plot(lambdas, np.average(order1_fit, axis=1))
    # pypl.plot(lambdas, np.average(order2_fit, axis=1))
    # pypl.plot(lambdas, np.average(order3_fit, axis=1))
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))
    pypl.show()
