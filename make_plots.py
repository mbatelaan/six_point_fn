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
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]
# From the theta tuning:
m_N = 0.4179255
m_S = 0.4641829



if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    nboot = 200
    nbin = 1

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
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

    print(datadir / ("lambda_dep.pkl"))
    # with open(datadir / ("lambda_dep_t4_dt2.pkl"), "rb") as file_in:
    time_choice = 2
    delta_t = 2
    with open(datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    # [lambdas, order0_fit, order1_fit, order2_fit, order3_fit] = data
    # [lambdas, order0_fit, order1_fit, order2_fit, order3_fit] = data
    lambdas = np.array(data["lambdas"])
    order0_fit = np.array(data["order0_fit"])
    order1_fit = np.array(data["order1_fit"])
    order2_fit = np.array(data["order2_fit"])
    order3_fit = np.array(data["order3_fit"])
    redchisq = data["redchisq"]
    time_choice = data["time_choice"]
    delta_t = data["delta_t"]

    print(lambdas)
    # print(np.array(order0_fit))
    print(np.shape(order0_fit))
    print(np.shape(order0_fit[0]))
    print(np.shape(lambdas))

    # order0_fit = np.einsum("ij,i->ij", order0_fit,lambdas**(-1))
    # order1_fit = np.einsum("ij,i->ij", order1_fit,lambdas**(-1))
    # order2_fit = np.einsum("ij,i->ij", order2_fit,lambdas**(-1))
    # order3_fit = np.einsum("ij,i->ij", order3_fit,lambdas**(-1))

    order0_fit = order0_fit[np.where(redchisq[0]<=1.5)]
    lambdas0 = lambdas[np.where(redchisq[0]<=1.5)]
    order1_fit = order1_fit[np.where(redchisq[1]<=1.5)]
    lambdas1 = lambdas[np.where(redchisq[1]<=1.5)]
    order2_fit = order2_fit[np.where(redchisq[2]<=1.5)]
    lambdas2 = lambdas[np.where(redchisq[2]<=1.5)]
    order3_fit = order3_fit[np.where(redchisq[3]<=1.5)]
    lambdas3 = lambdas[np.where(redchisq[3]<=1.5)]

    # scaled_z0 = (redchisq[0] - redchisq[0].min()) / redchisq[0].ptp()
    # colors_0 = [[0., 0., 0., i] for i in scaled_z0]

    pypl.figure(figsize=(6, 6))
    # pypl.figure(figsize=(6, 6))
    # pypl.errorbar(
    #     lambdas0,
    #     np.average(order0_fit, axis=1),
    #     np.std(order0_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^1)$",
    #     color=_colors[0],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # # pypl.scatter(lambdas, np.average(order0_fit, axis=1),label=r"$\mathcal{O}(\lambda^1)$", edgecolors=colors_0, s=150, marker='x', linewidths=4)
    # pypl.errorbar(
    #     lambdas1+0.0001,
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
    #     lambdas+0.0002,
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
    #     lambdas+0.0003,
    #     np.average(order3_fit, axis=1),
    #     np.std(order3_fit, axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^4)$",
    #     color=_colors[3],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )

    pypl.plot(
        lambdas0,
        np.average(order0_fit, axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=1,
    )
    pypl.plot(
        lambdas1,
        np.average(order1_fit, axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=1,
    )
    pypl.plot(
        lambdas2,
        np.average(order2_fit, axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=1,
    )
    pypl.plot(
        lambdas3,
        np.average(order3_fit, axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=1,
    )


    pypl.fill_between(
        lambdas0,
        np.average(order0_fit, axis=1) - np.std(order0_fit, axis=1),
        np.average(order0_fit, axis=1) + np.std(order0_fit, axis=1),
        # label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        lambdas1,
        np.average(order1_fit, axis=1) - np.std(order1_fit, axis=1),
        np.average(order1_fit, axis=1) + np.std(order1_fit, axis=1),
        # label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        lambdas2,
        np.average(order2_fit, axis=1) - np.std(order2_fit, axis=1),
        np.average(order2_fit, axis=1) + np.std(order2_fit, axis=1),
        # label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        lambdas3,
        np.average(order3_fit, axis=1) - np.std(order3_fit, axis=1),
        np.average(order3_fit, axis=1) + np.std(order3_fit, axis=1),
        # label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    pypl.xlim(-0.001, 0.045)
    pypl.ylim(-0.001, 0.035)
    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={time_choice}, \Delta t={delta_t}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # pypl.savefig(plotdir / ("Energy_over_lambda.pdf"))
    pypl.savefig(plotdir / ("lambda_dep.pdf"))
    # pypl.show()
    
    ### ----------------------------------------------------------------------
    lmb_val = 0.06 #0.16
    time_choice_range = np.arange(5,10)
    delta_t_range = np.arange(1,4)
    t_range = np.arange(4, 9)

    # with open(datadir / ("fit_data_time_choice"+str(time_choice_range[0])+"-"+str(time_choice_range[-1])+".pkl"), "rb") as file_in:
    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    lambdas = data["lambdas"]
    order0_fit = data["order0_fit"]
    order1_fit = data["order1_fit"]
    order2_fit = data["order2_fit"]
    order3_fit = data["order3_fit"]
    time_choice_range = data["time_choice"]
    delta_t_range = data["delta_t"]

    # [time_choice_range, delta_t_range, order0_fit, order1_fit,order2_fit,order3_fit] = data


    delta_t_choice = 0
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        time_choice_range,
        np.average(order0_fit[:,delta_t_choice,:], axis=1),
        np.std(order0_fit[:,delta_t_choice,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.001,
        np.average(order1_fit[:,delta_t_choice,:], axis=1),
        np.std(order1_fit[:,delta_t_choice,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.002,
        np.average(order2_fit[:,delta_t_choice,:], axis=1),
        np.std(order2_fit[:,delta_t_choice,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.003,
        np.average(order3_fit[:,delta_t_choice,:], axis=1),
        np.std(order3_fit[:,delta_t_choice,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    pypl.ylim(0, 0.2)
    pypl.xlabel("$t_{0}$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$\Delta t = {delta_t_range[delta_t_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    # pypl.show()


    t0_choice = 0
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        delta_t_range,
        np.average(order0_fit[t0_choice,:,:], axis=1),
        np.std(order0_fit[t0_choice,:,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range+0.001,
        np.average(order1_fit[t0_choice,:,:], axis=1),
        np.std(order1_fit[t0_choice,:,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range+0.002,
        np.average(order2_fit[t0_choice,:,:], axis=1),
        np.std(order2_fit[t0_choice,:,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range+0.003,
        np.average(order3_fit[t0_choice,:,:], axis=1),
        np.std(order3_fit[t0_choice,:,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    pypl.xlabel("$\Delta t$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}} = {time_choice_range[t0_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"delta_t_dep_l{lmb_val}.pdf"))
    # pypl.show()
