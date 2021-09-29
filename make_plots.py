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

pars1 = 0

def fitfunction2(lmb, pars0, pars1, pars2):
    deltaE = 0.5*(pars0+pars1) + 0.5*np.sqrt((pars0-pars1)**2 + 4*lmb**2*pars2**2)
    return deltaE

def fitfunction3(lmb, pars0, pars2):
    deltaE = 0.5*(pars0+pars1) + 0.5*np.sqrt((pars0-pars1)**2 + 4*lmb**2*pars2**2)
    return deltaE

def fit_lmb(ydata, function, lambdas, p0=None):
    """Fit the lambda dependence

    data is a correlator with tht bootstraps on the first index and the time on the second
    lambdas is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    # order0_fit[i] = bootfit1[:, 0]
    ydata = ydata.T
    print(np.shape(ydata))
    data_set = ydata
    ydata_avg = np.average(data_set, axis=0)
    print(ydata_avg)
    print(lambdas)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    popt_avg, pcov_avg = curve_fit(function, lambdas, ydata_avg, sigma=diag_sigma, p0=p0)
    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, np.linalg.inv(diag_sigma))
    print('popt_avg', popt_avg)
    redchisq = chisq / len(lambdas)
    bootfit = []
    for iboot, values in enumerate(ydata):
        popt, pcov = curve_fit(function, lambdas, values, sigma=diag_sigma)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    print(np.average(bootfit,axis=0))
    return bootfit, redchisq


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
    # time_choice = 2
    # delta_t = 2
    # t_range = np.arange(4, 10)
    t_range = np.arange(config["t_range0"], config["t_range1"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]


    with open(datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}_fit{t_range[0]}-{t_range[-1]}.pkl"), "rb") as file_in:
        # with open(datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    lambdas = data["lambdas"]
    order0_fit = data["order0_fit"]
    order1_fit = data["order1_fit"]
    order2_fit = data["order2_fit"]
    order3_fit = data["order3_fit"]
    redchisq = data["redchisq"]
    time_choice = data["time_choice"]
    delta_t = data["delta_t"]

    chisq_tol = 1.7
    order0_fit = order0_fit[np.where(redchisq[0]<=chisq_tol)]
    lambdas0 = lambdas[np.where(redchisq[0]<=chisq_tol)]
    order1_fit = order1_fit[np.where(redchisq[1]<=chisq_tol)]
    lambdas1 = lambdas[np.where(redchisq[1]<=chisq_tol)]
    order2_fit = order2_fit[np.where(redchisq[2]<=chisq_tol)]
    lambdas2 = lambdas[np.where(redchisq[2]<=chisq_tol)]
    order3_fit = order3_fit[np.where(redchisq[3]<=chisq_tol)]
    lambdas3 = lambdas[np.where(redchisq[3]<=chisq_tol)]

    print(np.shape(order0_fit))
    print(np.shape(order1_fit))
    print(np.shape(order2_fit))
    print(np.shape(order3_fit))

    print('\n')
    # Fit the quadratic behaviourin lambda
    # pars1 = np.average(order1_fit, axis=1)[0]
    # print(pars1)
    # print(np.average(order1_fit, axis=1))
    p0 = (0.01, 0.01, 0.006)
    fitlim = 18
    bootfit0, redchisq0 = fit_lmb(order0_fit, fitfunction2, lambdas0, p0=p0)
    print("redchisq",redchisq0,'\n')
    bootfit1, redchisq1 = fit_lmb(order1_fit[:fitlim], fitfunction2, lambdas1[:fitlim], p0=p0)
    print("redchisq",redchisq1,'\n')
    bootfit2, redchisq2 = fit_lmb(order2_fit[:fitlim], fitfunction2, lambdas2[:fitlim], p0=p0)
    print("redchisq",redchisq2,'\n')
    bootfit3, redchisq3 = fit_lmb(order3_fit[:fitlim], fitfunction2, lambdas3[:fitlim], p0=p0)
    print("redchisq",redchisq3,'\n')


    # scaled_z0 = (redchisq[0] - redchisq[0].min()) / redchisq[0].ptp()
    # colors_0 = [[0., 0., 0., i] for i in scaled_z0]

    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        lambdas0,
        np.average(order0_fit, axis=1),
        np.std(order0_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    # pypl.scatter(lambdas, np.average(order0_fit, axis=1),label=r"$\mathcal{O}(\lambda^1)$", edgecolors=colors_0, s=150, marker='x', linewidths=4)
    pypl.errorbar(
        lambdas1+0.0001,
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
        lambdas2+0.0002,
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
        lambdas3+0.0003,
        np.average(order3_fit, axis=1),
        np.std(order3_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    # pypl.ylim(0, 0.15)
    pypl.xlim(-0.001, 0.045)
    pypl.ylim(-0.002, 0.035)
    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={time_choice}, \Delta t={delta_t}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))

    # params0 = np.average(bootfit0,axis=0)
    # pypl.plot(lambdas0, fitfunction2(lambdas0, *params0), color=_colors[0])
    # params1 = np.average(bootfit1,axis=0)
    # pypl.plot(lambdas1, fitfunction2(lambdas1, *params1), color=_colors[1])
    # params2 = np.average(bootfit2,axis=0)
    # pypl.plot(lambdas2, fitfunction2(lambdas2, *params2), color=_colors[2])
    # params3 = np.average(bootfit3,axis=0)
    # pypl.plot(lambdas3, fitfunction2(lambdas3, *params3), color=_colors[3])

    fitBS0 = np.array([fitfunction2(lambdas0, *bf) for bf in bootfit0])
    print(np.std(fitBS0,axis=0))
    pypl.fill_between(
        lambdas0,
        np.average(fitBS0,axis=0) - np.std(fitBS0,axis=0),
        np.average(fitBS0,axis=0) + np.std(fitBS0,axis=0),
        alpha=0.3,
        color=_colors[0],
        label = f"{redchisq0:0.2}",
    )
    fitBS1 = np.array([fitfunction2(lambdas1[:fitlim], *bf) for bf in bootfit1])
    print(np.std(fitBS1,axis=0))
    pypl.fill_between(
        lambdas1[:fitlim],
        np.average(fitBS1,axis=0) - np.std(fitBS1,axis=0),
        np.average(fitBS1,axis=0) + np.std(fitBS1,axis=0),
        alpha=0.3,
        color=_colors[1],
        label = f"{redchisq1:0.2}",
    )
    fitBS2 = np.array([fitfunction2(lambdas2[:fitlim], *bf) for bf in bootfit2])
    print(np.std(fitBS2,axis=0))
    pypl.fill_between(
        lambdas2[:fitlim],
        np.average(fitBS2,axis=0) - np.std(fitBS2,axis=0),
        np.average(fitBS2,axis=0) + np.std(fitBS2,axis=0),
        alpha=0.3,
        color=_colors[2],
        label = f"{redchisq2:0.2}",
    )
    fitBS3 = np.array([fitfunction2(lambdas3, *bf) for bf in bootfit3])
    print(np.std(fitBS3,axis=0))
    pypl.fill_between(
        lambdas3,
        np.average(fitBS3,axis=0) - np.std(fitBS3,axis=0),
        np.average(fitBS3,axis=0) + np.std(fitBS3,axis=0),
        alpha=0.3,
        color=_colors[3],
        label = f"{redchisq3:0.2}",
    )

    pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    # pypl.ylim(0, 0.15)
    pypl.xlim(-0.01, 0.22)
    pypl.ylim(-0.01, 0.1)
    pypl.savefig(plotdir / ("lambda_dep_fit.pdf"))

    # pypl.xlim(-0.005, 0.08)
    # pypl.ylim(0.015, 0.065)
    pypl.xlim(-0.005, 0.08)
    pypl.ylim(-0.01, 0.065)
    pypl.savefig(plotdir / ("lambda_dep_zoom.pdf"))

    pypl.close()
    
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
        time_choice_range+0.03,
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
        time_choice_range+0.06,
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
        time_choice_range+0.09,
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
    # pypl.ylim(0, 0.06)
    pypl.ylim(0.03, 0.055)
    pypl.xlabel("$t_{0}$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$\Delta t = {delta_t_range[delta_t_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    # pypl.show()

    # --------------------------------------------------------------------------------
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
        delta_t_range+0.03,
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
        delta_t_range+0.06,
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
        delta_t_range+0.09,
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
    pypl.ylim(0.03, 0.055)
    pypl.xlabel("$\Delta t$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}} = {time_choice_range[t0_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"delta_t_dep_l{lmb_val}.pdf"))
    # pypl.show()
