import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY
from gevpanalysis.util import find_file
from gevpanalysis.util import read_config
from gevpanalysis.util import save_plot

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from gevpanalysis.common import read_pickle
from gevpanalysis.common import fit_value3
from gevpanalysis.common import read_correlators
from gevpanalysis.common import read_correlators2
from gevpanalysis.common import read_correlators4
from gevpanalysis.common import read_correlators5_complex
from gevpanalysis.common import make_matrices
from gevpanalysis.common import normalize_matrices
from gevpanalysis.common import gevp
from gevpanalysis.common import gevp_bootstrap

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


def plot_matrix(fitlist, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])

    # print(max(weights))
    # argument_w = np.argmax(weights)
    # Find the unique values of tmin and tmax to make a grid showing the reduced chi-squared values.
    unique_x = np.unique(t_0)
    unique_y = np.unique(delta_t)
    min_x = np.min(t_0)
    min_y = np.min(delta_t)
    plot_x = np.append(unique_x, unique_x[-1] + 1)
    plot_y = np.append(unique_y, unique_y[-1] + 1)

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = np.average(
            fitlist[i]["order3_fit"][:, 1]
        )
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"energy_matrix_gevp_" + name + ".pdf"))
    save_plot(
        fig,
        f"energy_matrix_gevp_{name}.pdf",
        subdir=plotdir,
    )
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = (
            -np.log(fitlist[i]["order3_eval_left"][0]) / delta_t[i]
        )
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"eval1_matrix_gevp_" + name + ".pdf"))
    save_plot(
        fig,
        f"eval1_matrix_gevp_{name}.pdf",
        subdir=plotdir,
    )
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = (
            -np.log(fitlist[i]["order3_eval_left"][1]) / delta_t[i]
        )
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"eval2_matrix_gevp_" + name +".pdf"))
    save_plot(fig, f"eval2_matrix_gevp_{name}.pdf", subdir=plotdir)
    plt.close()
    return


def plot_matrix_bs(fitlist, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])

    # print(max(weights))
    # argument_w = np.argmax(weights)
    # Find the unique values of tmin and tmax to make a grid showing the reduced chi-squared values.
    unique_x = np.unique(t_0)
    unique_y = np.unique(delta_t)
    min_x = np.min(t_0)
    min_y = np.min(delta_t)
    plot_x = np.append(unique_x, unique_x[-1] + 1)
    plot_y = np.append(unique_y, unique_y[-1] + 1)

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = np.average(
            fitlist[i]["order3_fit"][:, 1]
        )
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"energy_matrix_gevp_" + name + ".pdf"))
    save_plot(fig, f"energy_matrix_gevp_{name}.pdf", subdir=plotdir)
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = fitlist[i]["order3_eval_left"][0]
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"eval1_matrix_gevp_" + name + ".pdf"))
    save_plot(fig, f"eval1_matrix_gevp_{name}.pdf", subdir=plotdir)
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = fitlist[i]["order3_eval_left"][1]
    fig = plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu")  # , vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    # plt.savefig(plotdir / (f"eval2_matrix_gevp_" + name +".pdf"))
    save_plot(fig, f"eval2_matrix_gevp_{name}.pdf", subdir=plotdir)
    plt.close()
    return


def delta_t_slice(fitlist, delta_t_fix, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])
    indices = np.where(delta_t == delta_t_fix)
    evec_num = 0
    x = t_0[indices]

    # product = np.sum(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices] * np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices], axis=1)
    # a = np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices]
    # b = np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices]
    # print('a*b = ',a*b)
    # print('sum a*b = ',np.sum(a*b,axis=1))
    # # product2 = np.sum(np.array([ fit["order3_evec_right"][:,0] for fit in fitlist ])[indices] * np.array([ fit["order3_evec_right"][:,1] for fit in fitlist ])[indices], axis=1)
    # product2 = np.sum(np.array([ fit["order3_evec_right"][:,0] for fit in fitlist ])[indices] ** 2, axis=1)
    # print(np.shape(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices]))
    # # print(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices][0], np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices][0])
    # print('\n\nproduct = ', product)
    # print('\n\nproduct2 = ', product2)
    # # print('\n\nproduct = ', np.shape(product))

    # print(np.array([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ])[indices][0])
    # evecs = np.array([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ])[indices][0]
    # print(np.shape(evecs))
    # print(np.average(evecs,axis=0))

    # product = np.sum(np.average([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ], axis=0)[indices] * np.average([ fit["order3_evec_left_bs"][:,:,1] for fit in fitlist ],axis=0)[indices], axis=1)
    # # product2 = np.sum(np.average([ fit["order3_evec_right_bs"][:,:,0] for fit in fitlist ])[indices] * np.average([ fit["order3_evec_right_bs"][:,1] for fit in fitlist ])[indices], axis=1)
    # print('\n\nproduct = ', product)
    # # print('\n\nproduct2 = ', product2)

    energy_shifts0_bs = np.array([fit["order0_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts1_bs = np.array([fit["order1_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts2_bs = np.array([fit["order2_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts3_bs = np.array([fit["order3_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]

    energy_shifts0 = np.array([fit["order0_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts1 = np.array([fit["order1_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts2 = np.array([fit["order2_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts3 = np.array([fit["order3_fit"][:, 1] for fit in fitlist])[indices]

    eval_energy1 = np.array(
        [-np.log(fit["order3_eval_left"][0]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy2 = np.array(
        [-np.log(fit["order3_eval_left"][1]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy1_bs = np.array(
        [-np.log(fit["order3_eval_left_bs"][:, 0]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy2_bs = np.array(
        [-np.log(fit["order3_eval_left_bs"][:, 1]) / delta_t_fix for fit in fitlist]
    )[indices]
    evec_val1 = np.array([fit["order3_evec_left"][0, 0] ** 2 for fit in fitlist])[
        indices
    ]
    evec_val2 = np.array([fit["order3_evec_left"][1, 0] ** 2 for fit in fitlist])[
        indices
    ]
    evec_val1_2 = np.array([fit["order3_evec_left"][0, 1] ** 2 for fit in fitlist])[
        indices
    ]
    evec_val2_2 = np.array([fit["order3_evec_left"][1, 1] ** 2 for fit in fitlist])[
        indices
    ]
    evec_val1_bs = np.array(
        [fit["order3_evec_left_bs"][:, 0, evec_num] ** 2 for fit in fitlist]
    )[indices]
    evec_val2_bs = np.array(
        [fit["order3_evec_left_bs"][:, 1, evec_num] ** 2 for fit in fitlist]
    )[indices]
    evec_val1_2bs = np.array(
        [fit["order3_evec_left_bs"][:, 0, 1] ** 2 for fit in fitlist]
    )[indices]
    evec_val2_2bs = np.array(
        [fit["order3_evec_left_bs"][:, 1, 1] ** 2 for fit in fitlist]
    )[indices]
    # eval_energy1 = np.array([ fit["order3_eval_left"][0] for fit in fitlist ])[indices]
    # eval_energy2 = np.array([ fit["order3_eval_left"][1] for fit in fitlist ])[indices]

    eval_energy1_right = np.array(
        [-np.log(fit["order3_eval_right"][0]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy2_right = np.array(
        [-np.log(fit["order3_eval_right"][1]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy1_bs_right = np.array(
        [-np.log(fit["order3_eval_right_bs"][:, 0]) / delta_t_fix for fit in fitlist]
    )[indices]
    eval_energy2_bs_right = np.array(
        [-np.log(fit["order3_eval_right_bs"][:, 1]) / delta_t_fix for fit in fitlist]
    )[indices]
    evec_val1_right = np.array(
        [fit["order3_evec_right"][0, 0] ** 2 for fit in fitlist]
    )[indices]
    evec_val2_right = np.array(
        [fit["order3_evec_right"][1, 0] ** 2 for fit in fitlist]
    )[indices]
    evec_val1_2_right = np.array(
        [fit["order3_evec_right"][0, 1] ** 2 for fit in fitlist]
    )[indices]
    evec_val2_2_right = np.array(
        [fit["order3_evec_right"][1, 1] ** 2 for fit in fitlist]
    )[indices]
    print(len(x))

    # plt.figure(figsize=(5, 4))
    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(
        x,
        np.average(energy_shifts0_bs, axis=1),
        np.std(energy_shifts0_bs, axis=1),
        fmt=_markers[0],
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.05,
        np.average(energy_shifts1_bs, axis=1),
        np.std(energy_shifts1_bs, axis=1),
        fmt=_markers[1],
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.1,
        np.average(energy_shifts2_bs, axis=1),
        np.std(energy_shifts2_bs, axis=1),
        fmt=_markers[2],
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.15,
        np.average(energy_shifts3_bs, axis=1),
        np.std(energy_shifts3_bs, axis=1),
        fmt=_markers[3],
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.xticks(x)
    # plt.xlim(0.7, 6.2)
    plt.ylim(0.02, 0.05)
    # plt.ylim(0,1)
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Energy}$")
    plt.legend(fontsize="xx-small", framealpha=0.3)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_energyshift_bs_" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_energyshift_bs_{name}.pdf", subdir=plotdir)
    plt.close()

    # plt.figure(figsize=(5, 4))
    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(
        x,
        np.average(energy_shifts0, axis=1),
        np.std(energy_shifts0, axis=1),
        fmt=_markers[0],
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.05,
        np.average(energy_shifts1, axis=1),
        np.std(energy_shifts1, axis=1),
        fmt=_markers[1],
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.1,
        np.average(energy_shifts2, axis=1),
        np.std(energy_shifts2, axis=1),
        fmt=_markers[2],
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.15,
        np.average(energy_shifts3, axis=1),
        np.std(energy_shifts3, axis=1),
        fmt=_markers[3],
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.xticks(x)
    # plt.xlim(0.7, 6.2)
    plt.ylim(0.02, 0.05)
    # plt.ylim(0,1)
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Energy}$")
    plt.legend(fontsize="xx-small", framealpha=0.3)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_energyshift_" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_energyshift_{name}.pdf", subdir=plotdir)
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, evec_val1, color=_colors[0], label="left evec 1 first element")
    plt.plot(
        x,
        evec_val2,
        color=_colors[0],
        linestyle="--",
        label="left evec 1 second element",
    )
    plt.plot(x, evec_val1_2, color=_colors[1], label="left evec 2 first element")
    plt.plot(
        x,
        evec_val2_2,
        color=_colors[1],
        linestyle="--",
        label="left evec 2 second element",
    )
    plt.plot(x, evec_val1_right, color=_colors[2], label="right evec 1 first element")
    plt.plot(
        x,
        evec_val2_right,
        color=_colors[2],
        linestyle="--",
        label="right evec 1 second element",
    )
    plt.plot(x, evec_val1_2_right, color=_colors[3], label="right evec 2 first element")
    plt.plot(
        x,
        evec_val2_2_right,
        color=_colors[3],
        linestyle="--",
        label="right evec 2 second element",
    )
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector elements squared}$")
    fig.legend(fontsize="xx-small", framealpha=0.3)
    plt.ylim(0, 1)
    plt.xticks(x)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_evec_vals_{name}.pdf", subdir=plotdir)
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(
        x,
        evec_val1 + evec_val1_right,
        color=_colors[0],
        label="left + right evec 1 first element",
    )
    plt.plot(
        x,
        evec_val2 + evec_val2_right,
        color=_colors[0],
        linestyle="--",
        label="left + right evec 1 second element",
    )
    plt.plot(
        x,
        evec_val1_2 + evec_val1_2_right,
        color=_colors[1],
        label="left + right evec 2 first element",
    )
    plt.plot(
        x,
        evec_val2_2 + evec_val2_2_right,
        color=_colors[1],
        linestyle="--",
        label="left + right evec 2 second element",
    )
    # plt.plot(x, evec_val1_right, color=_colors[2], label='right evec 1 first element')
    # plt.plot(x, evec_val2_right, color=_colors[2], linestyle='--', label='right evec 1 second element')
    # plt.plot(x, evec_val1_2_right, color=_colors[3], label='right evec 2 first element')
    # plt.plot(x, evec_val2_2_right, color=_colors[3], linestyle='--', label='right evec 2 second element')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector value squared}$")
    fig.legend(fontsize="xx-small", framealpha=0.3)
    plt.ylim(0, 2)
    plt.xticks(x)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_summed" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_evec_vals_summed_{name}.pdf", subdir=plotdir)
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, eval_energy1, color=_colors[0], label="eval 1 left")
    plt.plot(x, eval_energy2, color=_colors[1], label="eval 2 left")
    plt.plot(x, eval_energy1_right, color=_colors[2], label="eval 1 right")
    plt.plot(x, eval_energy2_right, color=_colors[3], label="eval 2 right")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    fig.legend(fontsize="xx-small", framealpha=0.3)
    plt.ylim(0.3, 1)
    plt.xticks(x)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_eval_energy_{name}.pdf", subdir=plotdir)
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(
        x,
        np.average(eval_energy1_bs, axis=1),
        np.std(eval_energy1_bs, axis=1),
        color=_colors[0],
        fmt="s",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="eval 1 left",
    )
    plt.errorbar(
        x + 0.08,
        np.average(eval_energy2_bs, axis=1),
        np.std(eval_energy2_bs, axis=1),
        color=_colors[1],
        fmt="^",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="eval 2 left",
    )
    plt.errorbar(
        x,
        np.average(eval_energy1_bs_right, axis=1),
        np.std(eval_energy1_bs_right, axis=1),
        color=_colors[0],
        fmt="s",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="eval 1 right",
    )
    plt.errorbar(
        x + 0.08,
        np.average(eval_energy2_bs_right, axis=1),
        np.std(eval_energy2_bs_right, axis=1),
        color=_colors[1],
        fmt="^",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="eval 2 right",
    )
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    plt.ylim(0.3, 1)
    plt.xticks(x)
    fig.legend(fontsize="xx-small", framealpha=0.3)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_bs" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_eval_energy_bs{name}.pdf", subdir=plotdir)
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(
        x,
        np.average(evec_val1_bs, axis=1),
        np.std(evec_val1_bs, axis=1),
        color=_colors[0],
        fmt="s--",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="left evec 1 first element",
    )
    plt.errorbar(
        x,
        np.average(evec_val2_bs, axis=1),
        np.std(evec_val2_bs, axis=1),
        color=_colors[0],
        fmt="^-",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="left evec 1 second element",
    )
    plt.errorbar(
        x + 0.08,
        np.average(evec_val1_2bs, axis=1),
        np.std(evec_val1_2bs, axis=1),
        color=_colors[1],
        fmt="s--",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="left evec 2 first element",
    )
    plt.errorbar(
        x + 0.08,
        np.average(evec_val2_2bs, axis=1),
        np.std(evec_val2_2bs, axis=1),
        color=_colors[1],
        fmt="^-",
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
        label="left evec 2 second element",
    )
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector elements squared}$")
    plt.legend(fontsize="xx-small", framealpha=0.3)
    plt.ylim(0, 1)
    plt.xticks(x)
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_bs" + name + ".pdf"))
    save_plot(fig, f"delta_t{delta_t_fix}_evec_vals_bs{name}.pdf", subdir=plotdir)
    plt.close()

    return


def t_0_slice(fitlist, t_0_fix, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])
    indices = np.where(t_0 == t_0_fix)
    evec_num = 0
    x = delta_t[indices]

    energy_shifts0_bs = np.array([fit["order0_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts1_bs = np.array([fit["order1_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts2_bs = np.array([fit["order2_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]
    energy_shifts3_bs = np.array([fit["order3_fit_bs"][:, 1] for fit in fitlist])[
        indices
    ]

    energy_shifts0 = np.array([fit["order0_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts1 = np.array([fit["order1_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts2 = np.array([fit["order2_fit"][:, 1] for fit in fitlist])[indices]
    energy_shifts3 = np.array([fit["order3_fit"][:, 1] for fit in fitlist])[indices]

    fig = plt.figure(figsize=(5, 4))
    plt.errorbar(
        x,
        np.average(energy_shifts0_bs, axis=1),
        np.std(energy_shifts0_bs, axis=1),
        fmt=_markers[0],
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.05,
        np.average(energy_shifts1_bs, axis=1),
        np.std(energy_shifts1_bs, axis=1),
        fmt=_markers[1],
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.1,
        np.average(energy_shifts2_bs, axis=1),
        np.std(energy_shifts2_bs, axis=1),
        fmt=_markers[2],
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.15,
        np.average(energy_shifts3_bs, axis=1),
        np.std(energy_shifts3_bs, axis=1),
        fmt=_markers[3],
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.xticks(x)
    plt.xlim(0.7, 6.2)
    plt.ylim(0.02, 0.05)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\textrm{Energy}$")
    plt.legend(fontsize="xx-small", framealpha=0.3)
    plt.savefig(plotdir / (f"t_0{t_0_fix}_energyshift_bs_" + name + ".pdf"))
    plt.close()

    # plt.figure(figsize=(5, 4))
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        x,
        np.average(energy_shifts0, axis=1),
        np.std(energy_shifts0, axis=1),
        fmt=_markers[0],
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.05,
        np.average(energy_shifts1, axis=1),
        np.std(energy_shifts1, axis=1),
        fmt=_markers[1],
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.1,
        np.average(energy_shifts2, axis=1),
        np.std(energy_shifts2, axis=1),
        fmt=_markers[2],
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        x + 0.15,
        np.average(energy_shifts3, axis=1),
        np.std(energy_shifts3, axis=1),
        fmt=_markers[3],
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.xticks(x)
    plt.xlim(0.7, 6.2)
    plt.ylim(0.02, 0.05)
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\textrm{Energy}$")
    plt.legend(fontsize="xx-small", framealpha=0.3)
    plt.savefig(plotdir / (f"t_0{t_0_fix}_energyshift_" + name + ".pdf"))
    plt.close()

    return


def gevp_slice(fitlist, t_0_range, delta_t_range, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])

    # ==================================================
    # Energies from the fit
    fig, ax = plt.subplots(figsize=(9, 5))
    for delta_t_ in delta_t_range:
        indices = np.where(delta_t == delta_t_)
        print(indices)
        indices = np.where(
            [
                delta_t[i] == delta_t_ and t_0[i] in t_0_range
                for i, j in enumerate(delta_t)
            ]
        )
        print(indices)
        t_0_values = t_0[indices]
        # delta_t_values = delta_t_values[indices]
        # indices_ = np.where(t_0_values

        delta_t_x = (
            delta_t_
            + 0.7 * ((t_0_values - t_0_values[0]) / (t_0_values[-1] - t_0_values[0]))
            - 0.35
        )
        energy_shifts3_bs = np.array([fit["order3_fit_bs"][:, 1] for fit in fitlist])[
            indices
        ]
        energy_shifts3 = np.array([fit["order3_fit"][:, 1] for fit in fitlist])[indices]
        plt.errorbar(
            delta_t_x,
            np.average(energy_shifts3, axis=1),
            np.std(energy_shifts3, axis=1),
            fmt=_markers[0],
            # label=r"$\mathcal{O}(\lambda^4)$",
            color="k",
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )

    for i in range(len(delta_t_range)):
        plt.axvline(i + 0.5, linestyle="--", linewidth=0.5, color="k")
    plt.xticks(delta_t_range)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.tick_params(
        axis="y", which="major", direction="in", right=True, length=3, width=1
    )
    ax.tick_params(axis="x", which="major", direction="in", length=0, width=0)
    ax.tick_params(axis="x", which="minor", direction="in", top=True, length=3, width=1)
    plt.xlim(delta_t_range[0] - 0.5, delta_t_range[-1] + 0.5)
    plt.ylim(0.02, 0.075)
    plt.xlabel(r"$\Delta t_0$")
    plt.ylabel(r"$\Delta E$")
    # plt.savefig(plotdir / (f"gevp_slice_energyshift_" + name + ".pdf"))
    save_plot(fig, f"gevp_slice_energyshift_{name}.pdf", subdir=plotdir)
    plt.close()

    # ==================================================
    # Energies from the evals
    fig, ax = plt.subplots(figsize=(7, 5))
    for delta_t_ in delta_t_range:
        indices = np.where(delta_t == delta_t_)
        indices = np.where(
            [
                delta_t[i] == delta_t_ and t_0[i] in t_0_range
                for i, j in enumerate(delta_t)
            ]
        )
        t_0_values = t_0[indices]

        delta_t_x = (
            delta_t_
            + 0.7 * ((t_0_values - t_0_values[0]) / (t_0_values[-1] - t_0_values[0]))
            - 0.35
        )
        # energy_shifts3_bs = np.array([ fit["order3_fit_bs"][:,1] for fit in fitlist ])[indices]
        # energy_shifts3 = np.array([ fit["order3_fit"][:,1] for fit in fitlist ])[indices]

        eval_energy1 = np.array(
            [-np.log(fit["order3_eval_left_bs"][:, 0]) / delta_t_ for fit in fitlist]
        )[indices]
        eval_energy2 = np.array(
            [-np.log(fit["order3_eval_left_bs"][:, 1]) / delta_t_ for fit in fitlist]
        )[indices]
        eval_energy_diff = np.abs(eval_energy1 - eval_energy2)
        # eval_energy_diff = np.array([ np.abs(np.log(fit["order3_eval_left_bs"][0]/fit["order3_eval_left_bs"][1]))/delta_t_ for fit in fitlist ])[indices]
        print(np.shape(eval_energy_diff))
        plt.errorbar(
            delta_t_x,
            np.average(eval_energy_diff, axis=1),
            np.std(eval_energy_diff, axis=1),
            fmt=_markers[0],
            color="k",
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )
        # plt.errorbar(
        #     delta_t_x,
        #     np.average(eval_energy1, axis=1),
        #     np.std(eval_energy1, axis=1),
        #     fmt=_markers[0],
        #     color='k',
        #     capsize=4,
        #     elinewidth=1,
        #     markerfacecolor="none",
        # )
        # plt.errorbar(
        #     delta_t_x,
        #     np.average(eval_energy2, axis=1),
        #     np.std(eval_energy2, axis=1),
        #     fmt=_markers[1],
        #     color='b',
        #     capsize=4,
        #     elinewidth=1,
        #     markerfacecolor="none",
        # )

    for i in range(len(delta_t_range)):
        plt.axvline(i + 0.5, linestyle="--", linewidth=0.5, color="k")
    plt.xticks(delta_t_range)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.tick_params(
        axis="y", which="major", direction="in", right=True, length=3, width=1
    )
    ax.tick_params(axis="x", which="major", direction="in", length=0, width=0)
    ax.tick_params(axis="x", which="minor", direction="in", top=True, length=3, width=1)
    plt.xlim(delta_t_range[0] - 0.5, delta_t_range[-1] + 0.5)
    plt.ylim(0.02, 0.075)
    # plt.ylim(0.3,1)
    plt.xlabel(r"$\Delta t_0$")
    plt.ylabel(r"$\Delta E(c^+, c^-)$")
    # plt.savefig(plotdir / (f"gevp_slice_eval_energyshift_" + name + ".pdf"))
    save_plot(fig, f"gevp_slice_eval_energyshift_{name}.pdf", subdir=plotdir)
    plt.close()

    # ==================================================
    # Difference in energies
    fig, ax = plt.subplots(figsize=(9, 5))
    for delta_t_ in delta_t_range:
        indices = np.where(delta_t == delta_t_)
        indices = np.where(
            [
                delta_t[i] == delta_t_ and t_0[i] in t_0_range
                for i, j in enumerate(delta_t)
            ]
        )
        t_0_values = t_0[indices]

        delta_t_x = (
            delta_t_
            + 0.7 * ((t_0_values - t_0_values[0]) / (t_0_values[-1] - t_0_values[0]))
            - 0.35
        )
        # energy_shifts3_bs = np.array([ fit["order3_fit_bs"][:,1] for fit in fitlist ])[indices]
        energy_shifts3 = np.array([fit["order3_fit"][:, 1] for fit in fitlist])[indices]

        eval_energy1 = np.array(
            [-np.log(fit["order3_eval_left_bs"][:, 0]) / delta_t_ for fit in fitlist]
        )[indices]
        eval_energy2 = np.array(
            [-np.log(fit["order3_eval_left_bs"][:, 1]) / delta_t_ for fit in fitlist]
        )[indices]
        eval_energy_diff = np.abs(eval_energy1 - eval_energy2)
        # eval_energy_diff = np.array([ np.abs(np.log(fit["order3_eval_left_bs"][0]/fit["order3_eval_left_bs"][1]))/delta_t_ for fit in fitlist ])[indices]
        energy_diff = energy_shifts3 - eval_energy_diff

        print(np.shape(eval_energy_diff))
        plt.errorbar(
            delta_t_x,
            np.average(energy_diff, axis=1),
            np.std(energy_diff, axis=1),
            fmt=_markers[0],
            color="k",
            capsize=4,
            elinewidth=1,
            markerfacecolor="none",
        )

    for i in range(len(delta_t_range)):
        plt.axvline(i + 0.5, linestyle="--", linewidth=0.5, color="k")
    plt.axhline(0, linestyle="solid", linewidth=0.8, color="k")
    plt.xticks(delta_t_range)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.tick_params(
        axis="y", which="major", direction="in", right=True, length=3, width=1
    )
    ax.tick_params(axis="x", which="major", direction="in", length=0, width=0)
    ax.tick_params(axis="x", which="minor", direction="in", top=True, length=3, width=1)
    plt.xlim(delta_t_range[0] - 0.5, delta_t_range[-1] + 0.5)
    # plt.ylim(0.02,0.075)
    # plt.ylim(0.3,1)
    plt.xlabel(r"$\Delta t_0$")
    plt.ylabel(r"$\Delta E_{\lambda} - \Delta E_{\lambda}(c^+, c^-)$")
    # plt.savefig(plotdir / (f"gevp_slice_energydiff_eval_fit_" + name + ".pdf"))
    save_plot(fig, f"gevp_slice_energydiff_eval_fit_{name}.pdf", subdir=plotdir)
    plt.close()

    return


def plotting_script_diff_2(
    diffG1, diffG2, diffG3, diffG4, fitvals, t_range, lmb_val, name="", show=False
):
    spacing = 2
    xlim = 15
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
    plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    plt.xlabel("$t/a$")
    plt.legend(fontsize="x-small")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        plt.show()
    plt.close()
    return


def plot_eigenstates(state1, state2, t_range, lmb_val, name="", show=False):
    spacing = 2
    xlim = 15
    time = np.arange(0, np.shape(state1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

    yeffavg_1 = np.average(state1, axis=0)
    yeffstd_1 = np.std(state1, axis=0)
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
    # axs.plot(t_range, len(t_range) * [np.average(fitvals[0])], color=_colors[0])
    # axs.fill_between(
    #     t_range,
    #     np.average(fitvals[0]) - np.std(fitvals[0]),
    #     np.average(fitvals[0]) + np.std(fitvals[0]),
    #     alpha=0.3,
    #     color=_colors[0],
    # )
    yeffavg_2 = np.average(state2, axis=0)
    yeffstd_2 = np.std(state2, axis=0)
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
    # axs.plot(t_range, len(t_range) * [np.average(fitvals[1])], color=_colors[1])
    # axs.fill_between(
    #     t_range,
    #     np.average(fitvals[1]) - np.std(fitvals[1]),
    #     np.average(fitvals[1]) + np.std(fitvals[1]),
    #     alpha=0.3,
    #     color=_colors[1],
    # )
    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    plt.xlabel("$t/a$")
    plt.legend(fontsize="x-small")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("eigenstates" + name + ".pdf"))
    if show:
        plt.show()
    plt.close()
    return


# def normalize_matrices(matrices): #_1, matrix_2, matrix_3, matrix_4):
#     matrix_list = []
#     time_choice = 6
#     for matrix in matrices:
#         matrix_copy = matrix.copy()
#         # matrix = matrix.copy()/1e36
#         for i, elemi in enumerate(matrix):
#             for j, elemj in enumerate(elemi):
#                 matrix[i,j] = np.einsum('kl,k->kl', matrix_copy[i,j], np.sqrt(np.abs(matrix_copy[i,i,:,time_choice]*matrix_copy[j,j,:,time_choice]))**(-1))
#         matrix_list.append(matrix)
#     return matrix_list


def gevp_loop(G2_nucl, G2_sigm, lmb_val, datadir):
    time_choice_range = np.arange(1, 13)
    delta_t_range = np.arange(1, 8)
    lambdas = np.linspace(0, 0.05, 15)[1:]
    t_range = np.arange(7, 18)
    aexp_function = ff.initffncs("Aexp")
    fitlist = []

    matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(G2_nucl, G2_sigm, lmb_val)

    # print(np.average(matrix_4[1,1]))
    print(np.average(matrix_4, axis=2)[:, :, 6])
    [matrix_1, matrix_2, matrix_3, matrix_4] = normalize_matrices(
        [matrix_1, matrix_2, matrix_3, matrix_4], time_choice=6
    )
    print(np.average(matrix_4, axis=2)[:, :, 6])
    # print(np.average(matrix_4[1,1]))

    for i, time_choice in enumerate(time_choice_range):
        for j, delta_t in enumerate(delta_t_range):
            print(f"t_0 =  {time_choice}\tDelta t = {delta_t}\n")
            # GEVP on ensemble average
            Gt1_1, Gt2_1, gevp_data_1 = gevp(
                matrix_1, time_choice, delta_t, name="_test", show=False
            )
            Gt1_2, Gt2_2, gevp_data_2 = gevp(
                matrix_2, time_choice, delta_t, name="_test", show=False
            )
            Gt1_3, Gt2_3, gevp_data_3 = gevp(
                matrix_3, time_choice, delta_t, name="_test", show=False
            )
            Gt1_4, Gt2_4, gevp_data_4 = gevp(
                matrix_4, time_choice, delta_t, name="_test", show=False
            )

            # GEVP full bootstrap
            Gt1_1_bs, Gt2_1_bs, gevp_data_bs_1 = gevp_bootstrap(
                matrix_1, time_choice, delta_t, name="_test", show=False
            )
            Gt1_2_bs, Gt2_2_bs, gevp_data_bs_2 = gevp_bootstrap(
                matrix_2, time_choice, delta_t, name="_test", show=False
            )
            Gt1_3_bs, Gt2_3_bs, gevp_data_bs_3 = gevp_bootstrap(
                matrix_3, time_choice, delta_t, name="_test", show=False
            )
            Gt1_4_bs, Gt2_4_bs, gevp_data_bs_4 = gevp_bootstrap(
                matrix_4, time_choice, delta_t, name="_test", show=False
            )

            ratio1 = np.abs(Gt1_1 / Gt2_1)
            ratio2 = np.abs(Gt1_2 / Gt2_2)
            ratio3 = np.abs(Gt1_3 / Gt2_3)
            ratio4 = np.abs(Gt1_4 / Gt2_4)
            ratio1_bs = np.abs(Gt1_1_bs / Gt2_1_bs)
            ratio2_bs = np.abs(Gt1_2_bs / Gt2_2_bs)
            ratio3_bs = np.abs(Gt1_3_bs / Gt2_3_bs)
            ratio4_bs = np.abs(Gt1_4_bs / Gt2_4_bs)
            try:
                bootfit0, redchisq0 = fit_value3(ratio1, t_range, aexp_function)
                bootfit1, redchisq1 = fit_value3(ratio2, t_range, aexp_function)
                bootfit2, redchisq2 = fit_value3(ratio3, t_range, aexp_function)
                bootfit3, redchisq3 = fit_value3(ratio4, t_range, aexp_function)
            except:
                bootfit0, redchisq0 = np.nan, np.nan
                bootfit1, redchisq1 = np.nan, np.nan
                bootfit2, redchisq2 = np.nan, np.nan
                bootfit3, redchisq3 = np.nan, np.nan

            # Fit bootstrap GEVP
            try:
                bootfit0_bs, redchisq0_bs = fit_value3(
                    ratio1_bs, t_range, aexp_function
                )
                bootfit1_bs, redchisq1_bs = fit_value3(
                    ratio2_bs, t_range, aexp_function
                )
                bootfit2_bs, redchisq2_bs = fit_value3(
                    ratio3_bs, t_range, aexp_function
                )
                bootfit3_bs, redchisq3_bs = fit_value3(
                    ratio4_bs, t_range, aexp_function
                )
            except:
                bootfit0_bs, redchisq0_bs = np.nan, np.nan
                bootfit1_bs, redchisq1_bs = np.nan, np.nan
                bootfit2_bs, redchisq2_bs = np.nan, np.nan
                bootfit3_bs, redchisq3_bs = np.nan, np.nan

            fitparams = {
                "t_0": time_choice,
                "delta_t": delta_t,
                "order3_eval_left": gevp_data_4[0],
                "order3_evec_left": gevp_data_4[1],
                "order3_eval_right": gevp_data_4[2],
                "order3_evec_right": gevp_data_4[3],
                "order3_eval_left_bs": gevp_data_bs_4[0],
                "order3_evec_left_bs": gevp_data_bs_4[1],
                "order3_eval_right_bs": gevp_data_bs_4[2],
                "order3_evec_right_bs": gevp_data_bs_4[3],
                "order0_fit": bootfit0,
                "order1_fit": bootfit1,
                "order2_fit": bootfit2,
                "order3_fit": bootfit3,
                "order0_fit_bs": bootfit0_bs,
                "order1_fit_bs": bootfit1_bs,
                "order2_fit_bs": bootfit2_bs,
                "order3_fit_bs": bootfit3_bs,
                "red_chisq": redchisq3,
                "red_chisq_bs": redchisq3_bs,
            }
            fitlist.append(fitparams)

    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "wb") as file_out:
        pickle.dump(fitlist, file_out)

    return fitlist


if __name__ == "__main__":
    # Load the style file for matplotlib
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    pars = params(0)
    # Read in the directory data from the yaml file
    if len(sys.argv) >= 2:
        config = read_config(sys.argv[1])
    else:
        config = read_config("qmax")

    # Set parameters to defaults defined in another YAML file
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)

    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = PROJECT_BASE_DIRECTORY / Path("data/plots") / Path(config["name"])
    datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    lmb_val = config["lmb_val"]

    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    if "onlytwist" in config and config["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(pars, pickledir, pickledir2, mom_strings)
    elif "onlytwist2" in config and config["onlytwist2"]:
        G2_nucl, G2_sigm = read_correlators5_complex(
            pars, pickledir, pickledir2, mom_strings
        )
    elif "qmax" in config and config["qmax"]:
        G2_nucl, G2_sigm = read_correlators4(pars, pickledir, pickledir2, mom_strings)
    else:
        G2_nucl, G2_sigm = read_correlators(pars, pickledir, pickledir2, mom_strings)
    # lambdas = np.linspace(0, 0.05, 30)[1:]
    # t_range = np.arange(7, 18)
    # time_choice_range = np.arange(1, 12)
    # delta_t_range = np.arange(1, 7)

    name = ""

    # If the script is run with 2 arguments, then the fit results are loaded from the pickle file.
    # Otherwise the data is fit again.
    if len(sys.argv) > 2:
        print("argv > 2")
        with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "rb") as file_in:
            fitlist = pickle.load(file_in)
    else:
        print("argv = 2")
        fitlist = gevp_loop(G2_nucl, G2_sigm, lmb_val, datadir)

    # plot_matrix(fitlist, plotdir, name)
    # delta_t_slice(fitlist, 1, plotdir, name)
    # delta_t_slice(fitlist, 2, plotdir, name)
    # delta_t_slice(fitlist, 3, plotdir, name)
    # delta_t_slice(fitlist, 4, plotdir, name)
    # delta_t_slice(fitlist, 5, plotdir, name)
    # delta_t_slice(fitlist, 6, plotdir, name)
    # t_0_slice(fitlist, 3, plotdir, name)
    # t_0_slice(fitlist, 4, plotdir, name)
    # t_0_slice(fitlist, 5, plotdir, name)
    # t_0_slice(fitlist, 6, plotdir, name)
    gevp_slice(fitlist, np.arange(1, 13), np.arange(1, 6), plotdir, "t0_1_13_dt_1_6")
    gevp_slice(fitlist, np.arange(1, 9), np.arange(1, 7), plotdir, "t0_1_9_dt_1_7")
