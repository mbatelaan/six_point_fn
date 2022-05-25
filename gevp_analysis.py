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

    # energy_shifts = np.array([ fit["order3_fit"][:,1] for fit in fitlist ])[indices]
    eval_energy1 = np.array([ -np.log(fit["order3_eval_left"][0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2 = np.array([ -np.log(fit["order3_eval_left"][1])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy1_bs = np.array([ -np.log(fit["order3_eval_left_bs"][:, 0])/delta_t_fix for fit in fitlist ])[indices]
    eval_energy2_bs = np.array([ -np.log(fit["order3_eval_left_bs"][:, 1])/delta_t_fix for fit in fitlist ])[indices]
    evec_val1_bs = np.array([ fit["order3_evec_left_bs"][:, 0, evec_num]**2 for fit in fitlist ])[indices]
    evec_val2_bs = np.array([ fit["order3_evec_left_bs"][:, 1, evec_num]**2 for fit in fitlist ])[indices]
    evec_val1_2bs = np.array([ fit["order3_evec_left_bs"][:, 0, 1]**2 for fit in fitlist ])[indices]
    evec_val2_2bs = np.array([ fit["order3_evec_left_bs"][:, 1, 1]**2 for fit in fitlist ])[indices]
    # eval_energy1 = np.array([ fit["order3_eval_left"][0] for fit in fitlist ])[indices]
    # eval_energy2 = np.array([ fit["order3_eval_left"][1] for fit in fitlist ])[indices]
    evec_val1 = np.array([ fit["order3_evec_right"][0,0]**2 for fit in fitlist ])[indices]
    evec_val2 = np.array([ fit["order3_evec_right"][1,0]**2 for fit in fitlist ])[indices]
    evec_val1_2 = np.array([ fit["order3_evec_right"][0,1]**2 for fit in fitlist ])[indices]
    evec_val2_2 = np.array([ fit["order3_evec_right"][1,1]**2 for fit in fitlist ])[indices]
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

    plt.figure(figsize=(5, 4))
    plt.plot(x, evec_val1, color=_colors[0])
    plt.plot(x, evec_val2, color=_colors[0], linestyle='--')
    plt.plot(x, evec_val1_2, color=_colors[1])
    plt.plot(x, evec_val2_2, color=_colors[1], linestyle='--')
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{Eigenvector value squared}$")
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_evec_vals_" + name + ".pdf"))
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(x, eval_energy1, color=_colors[0])
    plt.plot(x, eval_energy2, color=_colors[1])
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_" + name + ".pdf"))
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.errorbar(x, np.average(eval_energy1_bs, axis=1), np.std(eval_energy1_bs, axis=1), color=_colors[0], fmt="s", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.errorbar(x+0.08, np.average(eval_energy2_bs, axis=1), np.std(eval_energy2_bs, axis=1), color=_colors[1], fmt="^", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
    plt.savefig(plotdir / (f"delta_t{delta_t_fix}_eval_energy_bs" + name + ".pdf"))
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.errorbar(x, np.average(evec_val1_bs, axis=1), np.std(evec_val1_bs, axis=1), color=_colors[0], fmt="s", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.errorbar(x+0.08, np.average(evec_val2_bs, axis=1), np.std(evec_val2_bs, axis=1), color=_colors[0], fmt="^", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.errorbar(x, np.average(evec_val1_2bs, axis=1), np.std(evec_val1_2bs, axis=1), color=_colors[1], fmt="s", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.errorbar(x+0.08, np.average(evec_val2_2bs, axis=1), np.std(evec_val2_2bs, axis=1), color=_colors[1], fmt="^", capsize=4, elinewidth=1, markerfacecolor="none")
    plt.xlabel(r"$t_{0}$")
    plt.ylabel(r"$\textrm{energy from eigenvalue}$")
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

def gevp_loop(G2_nucl, G2_sigm, lmb_val, datadir):
    time_choice_range = np.arange(1, 18)
    delta_t_range = np.arange(1, 7)
    lambdas = np.linspace(0, 0.05, 30)[1:]
    t_range = np.arange(7, 18)
    aexp_function = ff.initffncs("Aexp")
    fitlist = []

    for i, time_choice in enumerate(time_choice_range):
        for j, delta_t in enumerate(delta_t_range):
            print(f"t_0 =  {time_choice}\tDelta t = {delta_t}\n")
            # Construct a correlation matrix for each order in lambda (skipping order 0)
            matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
                G2_nucl, G2_sigm, lmb_val
            )
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
    # lambdas = np.linspace(0, 0.05, 30)[1:]
    # t_range = np.arange(7, 18)
    # time_choice_range = np.arange(1, 12)
    # delta_t_range = np.arange(1, 7)

    name = ""
    if len(sys.argv) > 2:
        print('argv > 2')
        with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "rb") as file_in:
            fitlist = pickle.load(file_in)
        # plot_matrix(fitlist, plotdir, name)
        delta_t_slice(fitlist, 1, plotdir, name)
        delta_t_slice(fitlist, 2, plotdir, name)
        delta_t_slice(fitlist, 3, plotdir, name)
        delta_t_slice(fitlist, 4, plotdir, name)
        # plot_matrix(fitlist, plotdir, name, bs="_bs")
    else:
        print('argv = 2')
        fitlist = gevp_loop(G2_nucl, G2_sigm, lmb_val, datadir)
        # plot_matrix(fitlist, plotdir, name)
        delta_t_slice(fitlist, 1, plotdir, name)
        delta_t_slice(fitlist, 2, plotdir, name)
        delta_t_slice(fitlist, 3, plotdir, name)
        delta_t_slice(fitlist, 4, plotdir, name)
        # plot_matrix(fitlist, plotdir, name, bs="bs")
    exit()

    # # order0_fit = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot))
    # # order1_fit = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot))
    # # order2_fit = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot))
    # order3_fit = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # order3_fit_bs = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # red_chisq_list = np.zeros((4, len(time_choice_range), len(delta_t_range)))
    # red_chisq_list_bs = np.zeros((4, len(time_choice_range), len(delta_t_range)))
    # # order0_evals = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # # order0_evecs = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2, 2))
    # # order1_evals = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # # order1_evecs = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2, 2))
    # # order2_evals = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # # order2_evecs = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2, 2))
    # # order3_evals = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2))
    # # order3_evecs = np.zeros((len(time_choice_range), len(delta_t_range), pars.nboot, 2, 2))
    # order3_eval_left = np.zeros((len(time_choice_range), len(delta_t_range),2))
    # order3_eval_right = np.zeros((len(time_choice_range), len(delta_t_range),2))
    # order3_evec_left = np.zeros((len(time_choice_range), len(delta_t_range),2,2))
    # order3_evec_right = np.zeros((len(time_choice_range), len(delta_t_range),2,2))
    # order3_eval_left_bs = np.zeros((len(time_choice_range), len(delta_t_range),pars.nboot,2))
    # order3_eval_right_bs = np.zeros((len(time_choice_range), len(delta_t_range),pars.nboot,2))
    # order3_evec_left_bs = np.zeros((len(time_choice_range), len(delta_t_range),pars.nboot,2,2))
    # order3_evec_right_bs = np.zeros((len(time_choice_range), len(delta_t_range),pars.nboot,2,2))

    # aexp_function = ff.initffncs("Aexp")
    # print(aexp_function.label)

    # for i, time_choice in enumerate(time_choice_range):
    #     for j, delta_t in enumerate(delta_t_range):
    #         print(f"t_0 =  {time_choice}\tDelta t = {delta_t}\n")
    #         # Construct a correlation matrix for each order in lambda (skipping order 0)
    #         ### ----------------------------------------------------------------------
    #         matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
    #             G2_nucl, G2_sigm, lmb_val
    #         )

    #         # # Order 0
    #         # Gt1_1, Gt2_1, gevp_data = gevp(
    #         #     matrix_1, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # Gt1_1_bs, Gt2_1_bs, gevp_data_bs = gevp_bootstrap(
    #         #     matrix_1, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # order0_gevp_data[i, j] = gevp_data
    #         # order0_gevp_data_bs[i, j] = gevp_data_bs

    #         # ratio0 = Gt1_1 / Gt2_1
    #         # bootfit0, redchisq1 = fit_value3(ratio0, t_range, aexp_function)
    #         # order0_fit[i, j] = bootfit0
    #         # ratio0_bs = Gt1_1_bs / Gt2_1_bs
    #         # bootfit0_bs, redchisq1_bs = fit_value3(ratio0_bs, t_range, aexp_function)
    #         # order0_fit_bs[i, j] = bootfit0_bs
    #         # red_chisq_list[0, i, j] = redchisq1
    #         # red_chisq_list_bs[0, i, j] = redchisq1_bs

    #         # Gt1_2, Gt2_2, evals = gevp(
    #         #     matrix_2, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # ratio2 = Gt1_2 / Gt2_2
    #         # bootfit2, redchisq2 = fit_value3(ratio2, t_range, aexp_function)
    #         # order1_fit[i, j] = bootfit2[:, 1]
    #         # red_chisq_list[1, i, j] = redchisq2

    #         # Gt1_3, Gt2_3, evals = gevp(
    #         #     matrix_3, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # ratio3 = Gt1_3 / Gt2_3
    #         # bootfit3, redchisq3 = fit_value3(ratio3, t_range, aexp_function)
    #         # order2_fit[i, j] = bootfit3[:, 1]
    #         # red_chisq_list[2, i, j] = redchisq3

    #         # Gt1_4, Gt2_4, evals = gevp(
    #         #     matrix_4, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # Gt1_4, Gt2_4, [eval_left, evec_left, eval_right, evec_right] = gevp_bootstrap(
    #         #     matrix_4, time_choice, delta_t, name="_test", show=False
    #         # )
    #         # order3_evals[i,j] = eval_left
    #         # order3_evecs[i,j] = evec_left
    #         # ratio4 = Gt1_4 / Gt2_4
    #         # bootfit4, redchisq4 = fit_value3(ratio4, t_range, aexp_function)
    #         # order3_fit[i, j] = bootfit4[:, 1]
    #         # red_chisq_list[3, i, j] = redchisq4

    #         # order 4
    #         Gt1_4, Gt2_4, gevp_data = gevp(
    #             matrix_4, time_choice, delta_t, name="_test", show=False
    #         )
    #         Gt1_4_bs, Gt2_4_bs, gevp_data_bs = gevp_bootstrap(
    #             matrix_4, time_choice, delta_t, name="_test", show=False
    #         )
    #         # print(len(gevp_data[0]))
    #         # print(len(gevp_data[1]))
    #         # print(len(gevp_data_bs[0]))
    #         # print(len(gevp_data_bs[1]))
    #         # print(np.shape(gevp_data[0]))
    #         # print(np.shape(gevp_data[1]))
    #         # print(np.shape(gevp_data_bs[0]))
    #         # print(np.shape(gevp_data_bs[1]))
    #         order3_eval_left[i, j] = gevp_data[0]
    #         order3_evec_left[i, j] = gevp_data[1]
    #         order3_eval_right[i, j] = gevp_data[2]
    #         order3_evec_right[i, j] = gevp_data[3]

    #         order3_eval_left_bs[i, j] = gevp_data_bs[0]
    #         order3_evec_left_bs[i, j] = gevp_data_bs[1]
    #         order3_eval_right_bs[i, j] = gevp_data_bs[2]
    #         order3_evec_right_bs[i, j] = gevp_data_bs[3]

    #         # order3_gevp_data_bs[i, j] = gevp_data_bs
    #         ratio3 = Gt1_4 / Gt2_4
    #         bootfit3, redchisq3 = fit_value3(ratio3, t_range, aexp_function)
    #         order3_fit[i, j] = bootfit3
    #         ratio3_bs = Gt1_4_bs / Gt2_4_bs
    #         bootfit3_bs, redchisq3_bs = fit_value3(ratio3_bs, t_range, aexp_function)
    #         order3_fit_bs[i, j] = bootfit3_bs
    #         red_chisq_list[3, i, j] = redchisq3
    #         red_chisq_list_bs[3, i, j] = redchisq3_bs

    #         if False:
    #             plotting_script_diff_2(
    #                 effmass_ratio0,
    #                 effmass_ratio1,
    #                 effmass_ratio2,
    #                 effmass_ratio3,
    #                 [bootfit1[:, 1], bootfit2[:, 1], bootfit3[:, 1], bootfit4[:, 1]],
    #                 # [bootfit1, bootfit2, bootfit3, bootfit4],
    #                 t_range,
    #                 lmb_val,
    #                 name="_l" + str(lmb_val) + "_time_choice" + str(time_choice),
    #                 show=False,
    #             )
    #             plot_eigenstates(effmass_1, effmass_2, t_range, lmb_val,
    #                 name="_l" + str(lmb_val) + "_time_choice" + str(time_choice),
    #                 show=False)


    # # ----------------------------------------------------------------------
    # # Save the fit data to a pickle file
    # all_data = {
    #     "lambdas": np.array([lmb_val]),
    #     # "order0_fit": order0_fit,
    #     # "order1_fit": order1_fit,
    #     # "order2_fit": order2_fit,
    #     "order3_fit": order3_fit,
    #     "order3_eval_left": order3_eval_left,
    #     "order3_eval_right": order3_eval_right,
    #     "order3_evec_left": order3_evec_left,
    #     "order3_evec_right": order3_evec_right,
    #     "order3_eval_left_bs": order3_eval_left_bs,
    #     "order3_eval_right_bs": order3_eval_right_bs,
    #     "order3_evec_left_bs": order3_evec_left_bs,
    #     "order3_evec_right_bs": order3_evec_right_bs,
    #     "redchisq": red_chisq_list,
    #     "redchisq_bs": red_chisq_list_bs,
    #     "time_choice": time_choice_range,
    #     "delta_t": delta_t_range,
    # }
    # with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "wb") as file_out:
    #     pickle.dump(all_data, file_out)

    # ----------------------------------------------------------------------
    # Make a plot of the dependence of the energy shift on the time parameters in the GEVP
    plt.figure(figsize=(6, 6))
    plt.errorbar(
        time_choice_range,
        np.average(order0_fit[:, 0, :], axis=1),
        np.std(order0_fit[:, 0, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        time_choice_range + 0.001,
        np.average(order1_fit[:, 0, :], axis=1),
        np.std(order1_fit[:, 0, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        time_choice_range + 0.002,
        np.average(order2_fit[:, 0, :], axis=1),
        np.std(order2_fit[:, 0, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        time_choice_range + 0.003,
        np.average(order3_fit[:, 0, :], axis=1),
        np.std(order3_fit[:, 0, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")
    # plt.xlim(-0.01, 0.22)
    # plt.ylim(0, 0.2)
    plt.xlabel("time choice")
    plt.ylabel("$\Delta E$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    # plt.show()
