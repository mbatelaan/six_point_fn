import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from common import read_pickle
from common import fit_value3
from common import read_correlators
from common import read_correlators2
from common import read_correlators4
from common import read_correlators5
from common import make_matrices
from common import gevp
from common import gevp_bootstrap

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
        matrix[x - min_x, delta_t[i] - min_y] = np.average(fitlist[i]["order3_fit"][:,1])
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"energy_matrix_gevp_" + name + ".pdf"))
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = - np.log(fitlist[i]["order3_eval_left"][0])/delta_t[i]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"eval1_matrix_gevp_" + name + ".pdf"))
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = - np.log(fitlist[i]["order3_eval_left"][1])/delta_t[i]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"eval2_matrix_gevp_" + name +".pdf"))
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
        matrix[x - min_x, delta_t[i] - min_y] = np.average(fitlist[i]["order3_fit"][:,1])
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"energy_matrix_gevp_" + name + ".pdf"))
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = fitlist[i]["order3_eval_left"][0]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"eval1_matrix_gevp_" + name + ".pdf"))
    plt.close()

    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(t_0):
        matrix[x - min_x, delta_t[i] - min_y] = fitlist[i]["order3_eval_left"][1]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu") #, vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\textrm{Energy}$")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\Delta t$")
    plt.savefig(plotdir / (f"eval2_matrix_gevp_" + name +".pdf"))
    plt.close()
    return

def delta_t_slice(fitlist, delta_t_fix, plotdir, name):
    t_0 = np.array([i["t_0"] for i in fitlist])
    delta_t = np.array([i["delta_t"] for i in fitlist])
    indices = np.where(delta_t == delta_t_fix)
    evec_num = 0
    x = t_0[indices]

    product = np.sum(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices] * np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices], axis=1)
    a = np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices]
    b = np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices]
    print('a*b = ',a*b)
    print('sum a*b = ',np.sum(a*b,axis=1))
    # product2 = np.sum(np.array([ fit["order3_evec_right"][:,0] for fit in fitlist ])[indices] * np.array([ fit["order3_evec_right"][:,1] for fit in fitlist ])[indices], axis=1)
    product2 = np.sum(np.array([ fit["order3_evec_right"][:,0] for fit in fitlist ])[indices] ** 2, axis=1)
    print(np.shape(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices]))
    # print(np.array([ fit["order3_evec_left"][:,0] for fit in fitlist ])[indices][0], np.array([ fit["order3_evec_left"][:,1] for fit in fitlist ])[indices][0])
    print('\n\nproduct = ', product)
    print('\n\nproduct2 = ', product2)
    # print('\n\nproduct = ', np.shape(product))

    # print(np.array([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ])[indices][0])
    # evecs = np.array([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ])[indices][0]
    # print(np.shape(evecs))
    # print(np.average(evecs,axis=0))

    # product = np.sum(np.average([ fit["order3_evec_left_bs"][:,:,0] for fit in fitlist ], axis=0)[indices] * np.average([ fit["order3_evec_left_bs"][:,:,1] for fit in fitlist ],axis=0)[indices], axis=1)
    # # product2 = np.sum(np.average([ fit["order3_evec_right_bs"][:,:,0] for fit in fitlist ])[indices] * np.average([ fit["order3_evec_right_bs"][:,1] for fit in fitlist ])[indices], axis=1)
    # print('\n\nproduct = ', product)
    # # print('\n\nproduct2 = ', product2)


    # energy_shifts = np.array([ fit["order3_fit"][:,1] for fit in fitlist ])[indices]
    eval_energy1 = np.array([ -np.log(fit["order3_eval_left"][0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2 = np.array([ -np.log(fit["order3_eval_left"][1])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy1_bs = np.array([ -np.log(fit["order3_eval_left_bs"][:, 0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2_bs = np.array([ -np.log(fit["order3_eval_left_bs"][:, 1])/delta_t_fix for fit in fitlist ])[indices]
    evec_val1 = np.array([ fit["order3_evec_left"][0,0]**2 for fit in fitlist ])[indices]
    evec_val2 = np.array([ fit["order3_evec_left"][1,0]**2 for fit in fitlist ])[indices]
    evec_val1_2 = np.array([ fit["order3_evec_left"][0,1]**2 for fit in fitlist ])[indices]
    evec_val2_2 = np.array([ fit["order3_evec_left"][1,1]**2 for fit in fitlist ])[indices]
    evec_val1_bs = np.array([ fit["order3_evec_left_bs"][:, 0, evec_num]**2 for fit in fitlist ])[indices]
    evec_val2_bs = np.array([ fit["order3_evec_left_bs"][:, 1, evec_num]**2 for fit in fitlist ])[indices]
    evec_val1_2bs = np.array([ fit["order3_evec_left_bs"][:, 0, 1]**2 for fit in fitlist ])[indices]
    evec_val2_2bs = np.array([ fit["order3_evec_left_bs"][:, 1, 1]**2 for fit in fitlist ])[indices]
    # eval_energy1 = np.array([ fit["order3_eval_left"][0] for fit in fitlist ])[indices]
    # eval_energy2 = np.array([ fit["order3_eval_left"][1] for fit in fitlist ])[indices]

    eval_energy1_right = np.array([ -np.log(fit["order3_eval_right"][0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2_right = np.array([ -np.log(fit["order3_eval_right"][1])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy1_bs_right = np.array([ -np.log(fit["order3_eval_right_bs"][:, 0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2_bs_right = np.array([ -np.log(fit["order3_eval_right_bs"][:, 1])/delta_t_fix for fit in fitlist ])[indices]
    evec_val1_right = np.array([ fit["order3_evec_right"][0,0]**2 for fit in fitlist ])[indices]
    evec_val2_right = np.array([ fit["order3_evec_right"][1,0]**2 for fit in fitlist ])[indices]
    evec_val1_2_right = np.array([ fit["order3_evec_right"][0,1]**2 for fit in fitlist ])[indices]
    evec_val2_2_right = np.array([ fit["order3_evec_right"][1,1]**2 for fit in fitlist ])[indices]
    print(len(x))

    # plt.figure(figsize=(5, 4))
    # plt.errorbar(
    #     x,
    #     np.average(energy_shifts, axis=1),
    #     np.std(energy_shifts, axis=1),
    #     fmt="s",
    #     # label=r"$\mathcal{O}(\lambda^1)$",
    #     color=_colors[0],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # plt.xlabel(r"$t_{0}$")
    # plt.ylabel(r"$\textrm{Energy}$")
    # plt.savefig(plotdir / (f"delta_t{delta_t_fix}_energyshift_" + name + ".pdf"))
    # plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, evec_val1, color=_colors[0],label='left evec 1 first element')
    plt.plot(x, evec_val2, color=_colors[0], linestyle='--', label='left evec 1 second element')
    plt.plot(x, evec_val1_2, color=_colors[1],label='left evec 2 first element')
    plt.plot(x, evec_val2_2, color=_colors[1], linestyle='--',label='left evec 2 second element')
    plt.plot(x, evec_val1_right, color=_colors[2], label='right evec 1 first element')
    plt.plot(x, evec_val2_right, color=_colors[2], linestyle='--', label='right evec 1 second element')
    plt.plot(x, evec_val1_2_right, color=_colors[3], label='right evec 2 first element')
    plt.plot(x, evec_val2_2_right, color=_colors[3], linestyle='--', label='right evec 2 second element')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector elements squared}$")
    fig.legend(fontsize='xx-small', framealpha=0.3)
    plt.ylim(0,1)
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_" + name + ".pdf"))
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, evec_val1 + evec_val1_right, color=_colors[0],label='left + right evec 1 first element')
    plt.plot(x, evec_val2 + evec_val2_right, color=_colors[0], linestyle='--', label='left + right evec 1 second element')
    plt.plot(x, evec_val1_2 + evec_val1_2_right, color=_colors[1],label='left + right evec 2 first element')
    plt.plot(x, evec_val2_2 + evec_val2_2_right, color=_colors[1], linestyle='--',label='left + right evec 2 second element')
    # plt.plot(x, evec_val1_right, color=_colors[2], label='right evec 1 first element')
    # plt.plot(x, evec_val2_right, color=_colors[2], linestyle='--', label='right evec 1 second element')
    # plt.plot(x, evec_val1_2_right, color=_colors[3], label='right evec 2 first element')
    # plt.plot(x, evec_val2_2_right, color=_colors[3], linestyle='--', label='right evec 2 second element')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector value squared}$")
    fig.legend(fontsize='xx-small', framealpha=0.3)
    plt.ylim(0,2)
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_summed" + name + ".pdf"))
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, eval_energy1, color=_colors[0], label='eval 1 left')
    plt.plot(x, eval_energy2, color=_colors[1], label='eval 2 left')
    plt.plot(x, eval_energy1_right, color=_colors[2], label='eval 1 right')
    plt.plot(x, eval_energy2_right, color=_colors[3], label='eval 2 right')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    fig.legend(fontsize='xx-small', framealpha=0.3)
    plt.ylim(0.3,1)
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_" + name + ".pdf"))
    plt.close()

    fig = plt.figure(figsize=(9, 6))
    plt.errorbar(x, np.average(eval_energy1_bs, axis=1), np.std(eval_energy1_bs, axis=1), color=_colors[0], fmt="s", capsize=4, elinewidth=1, markerfacecolor="none", label='eval 1 left')
    plt.errorbar(x+0.08, np.average(eval_energy2_bs, axis=1), np.std(eval_energy2_bs, axis=1), color=_colors[1], fmt="^", capsize=4, elinewidth=1, markerfacecolor="none", label='eval 2 left')
    plt.errorbar(x, np.average(eval_energy1_bs_right, axis=1), np.std(eval_energy1_bs_right, axis=1), color=_colors[0], fmt="s", capsize=4, elinewidth=1, markerfacecolor="none", label='eval 1 right')
    plt.errorbar(x+0.08, np.average(eval_energy2_bs_right, axis=1), np.std(eval_energy2_bs_right, axis=1), color=_colors[1], fmt="^", capsize=4, elinewidth=1, markerfacecolor="none", label='eval 2 right')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    plt.ylim(0.3,1)
    fig.legend(fontsize='xx-small', framealpha=0.3)
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_bs" + name + ".pdf"))
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.errorbar(x, np.average(evec_val1_bs, axis=1), np.std(evec_val1_bs, axis=1), color=_colors[0], fmt="s--", capsize=4, elinewidth=1, markerfacecolor="none",label='left evec 1 first element')
    plt.errorbar(x, np.average(evec_val2_bs, axis=1), np.std(evec_val2_bs, axis=1), color=_colors[0], fmt="^-", capsize=4, elinewidth=1, markerfacecolor="none",label='left evec 1 second element')
    plt.errorbar(x+0.08, np.average(evec_val1_2bs, axis=1), np.std(evec_val1_2bs, axis=1), color=_colors[1], fmt="s--", capsize=4, elinewidth=1, markerfacecolor="none",label='left evec 2 first element')
    plt.errorbar(x+0.08, np.average(evec_val2_2bs, axis=1), np.std(evec_val2_2bs, axis=1), color=_colors[1], fmt="^-", capsize=4, elinewidth=1, markerfacecolor="none",label='left evec 2 second element')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector elements squared}$")
    plt.legend(fontsize='xx-small', framealpha=0.3)
    plt.ylim(0,1)
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_bs" + name + ".pdf"))
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

def plot_eigenstates(
    state1, state2, t_range, lmb_val, name="", show=False
):
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

def normalize_matrices(matrices): #_1, matrix_2, matrix_3, matrix_4):
    matrix_list = []
    time_choice = 6
    for matrix in matrices:
        matrix_copy = matrix.copy()
        # matrix = matrix/1e38
        for i, elemi in enumerate(matrix):
            for j, elemj in enumerate(elemi):
                # print(np.shape(matrix[i,i,:,1]))
                # print(np.shape(matrix[i,j]))
                # print(np.shape(np.sqrt(np.abs(matrix[i,i,:,1]*matrix[j,j,:,1]))[0]**(-1)))
                # print('\n',np.average(matrix[i,j],axis=0)[time_choice])
                # print(np.average(matrix[i,i,:,time_choice]))
                # print(np.average(matrix[j,j,:,time_choice]))
                # print(np.average(np.sqrt(np.abs(matrix[i,i,:,time_choice]*matrix[j,j,:,time_choice]))))
                # print(np.average(np.sqrt(np.abs(matrix[i,i,:,time_choice]*matrix[j,j,:,time_choice]))**(-1)))
                matrix[i,j] = np.einsum('kl,k->kl', matrix_copy[i,j], np.sqrt(np.abs(matrix_copy[i,i,:,time_choice]*matrix_copy[j,j,:,time_choice]))**(-1))
        matrix_list.append(matrix)
    return matrix_list


def gevp_loop(G2_nucl, G2_sigm, lmb_val, datadir):
    time_choice_range = np.arange(1, 18)
    delta_t_range = np.arange(1, 7)
    lambdas = np.linspace(0, 0.05, 30)[1:]
    t_range = np.arange(7, 18)
    aexp_function = ff.initffncs("Aexp")
    fitlist = []

    matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
        G2_nucl, G2_sigm, lmb_val
    )
    
    # print(np.average(matrix_4[1,1]))
    print(np.average(matrix_4,axis=2)[:,:,6])
    [matrix_1, matrix_2, matrix_3, matrix_4] = normalize_matrices(
        [matrix_1, matrix_2, matrix_3, matrix_4]
    )
    print(np.average(matrix_4,axis=2)[:,:,6])
    # print(np.average(matrix_4[1,1]))
    
    for i, time_choice in enumerate(time_choice_range):
        for j, delta_t in enumerate(delta_t_range):
            print(f"t_0 =  {time_choice}\tDelta t = {delta_t}\n")
            # order 4
            Gt1_4, Gt2_4, gevp_data = gevp(
                matrix_4, time_choice, delta_t, name="_test", show=False
            )
            Gt1_4_bs, Gt2_4_bs, gevp_data_bs = gevp_bootstrap(
                matrix_4, time_choice, delta_t, name="_test", show=False
            )

            ratio3 = Gt1_4 / Gt2_4
            try:
                bootfit3, redchisq3 = fit_value3(ratio3, t_range, aexp_function)
            except:
                bootfit3, redchisq3 = np.nan, np.nan
            ratio3_bs = Gt1_4_bs / Gt2_4_bs
            try:
                bootfit3_bs, redchisq3_bs = fit_value3(ratio3_bs, t_range, aexp_function)
            except:
                bootfit3_bs, redchisq3_bs = np.nan, np.nan

            fitparams = {
                "t_0": time_choice,
                "delta_t": delta_t,
                "order3_eval_left": gevp_data[0],
                "order3_evec_left": gevp_data[1],
                "order3_eval_right": gevp_data[2],
                "order3_evec_right": gevp_data[3],
                "order3_eval_left_bs": gevp_data_bs[0],
                "order3_evec_left_bs": gevp_data_bs[1],
                "order3_eval_right_bs": gevp_data_bs[2],
                "order3_evec_right_bs": gevp_data_bs[3],
                "order3_fit": bootfit3,
                "order3_fit_bs": bootfit3_bs,
                "red_chisq": redchisq3,
                "red_chisq_bs": redchisq3_bs,
            }
            fitlist.append(fitparams)

    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "wb") as file_out:
        pickle.dump(fitlist, file_out)

    return fitlist


if __name__ == "__main__":
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    # Read in the directory data from the yaml file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # TODO: Set up a defaults.yaml file for when there is no input file
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    lmb_val = config["lmb_val"]

    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    if "onlytwist" in config and config["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(pars, pickledir, pickledir2, mom_strings)
    elif "onlytwist2" in config and config["onlytwist2"]:
        G2_nucl, G2_sigm = read_correlators5(
            pars, pickledir, pickledir2, mom_strings
        )
    elif "qmax" in config and config["qmax"]:
        G2_nucl, G2_sigm = read_correlators4(
            pars, pickledir, pickledir2, mom_strings
        )
    else:
        G2_nucl, G2_sigm = read_correlators(pars, pickledir, pickledir2, mom_strings)

    # corr_11_avg_real = np.average(G2_nucl[0][:,:,0], axis=0)
    # print('\n')
    # print(corr_11_avg_real)

    # corr_11_avg_imag = np.average(G2_nucl[0][:,:,1], axis=0)
    # print('\n')
    # print(corr_11_avg_imag)

    # print('\n')
    # print(corr_11_avg_real - corr_11_avg_imag)

    # corr_11_avg = np.average(G2_nucl[0][:,:,0], axis=0) + 1j * np.average(G2_nucl[0][:,:,1], axis=0)
    # print('\n')
    # print(corr_11_avg)

    for icorr, corr in enumerate(G2_nucl):
        G2_nucl[icorr] = corr[:, :, 0] + 1j * corr[:, :, 1]
    for icorr, corr in enumerate(G2_sigm):
        G2_sigm[icorr] = corr[:, :, 0] + 1j * corr[:, :, 1]

    corr_11_avg_real = np.average(G2_nucl[0][:,:], axis=0)
    print('\n')
    print(corr_11_avg_real)

    corr_12_avg_real = np.average(G2_nucl[1][:,:], axis=0)
    print('\n')
    print(corr_12_avg_real)


    matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
        G2_nucl, G2_sigm, lmb_val
    )

 
