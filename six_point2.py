import numpy as np
from pathlib import Path
import pickle
import sys
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


def plotratio(ratio, lmb, plotname, plotdir, ylim=None, fitparam=None, ylabel=None):
    """Plot the effective mass of the ratio of correlators for both lambdas and plot their fits"""
    spacing = 2
    time = np.arange(0, np.shape(ratio)[1])
    efftime = time[:-spacing] + 0.5

    effmassdata = stats.bs_effmass(ratio, time_axis=1, spacing=spacing) / lmb
    # effmassdata = ratio[:, :-1]
    yeffavg = np.average(effmassdata, axis=0)
    yeffstd = np.std(effmassdata, axis=0)

    xlim = 30
    pypl.figure(figsize=(7, 6))
    pypl.errorbar(
        efftime[:xlim],
        yeffavg[:xlim],
        yeffstd[:xlim],
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    if fitparam:
        pypl.plot(fitparam[0], np.average(fitparam[1], axis=0))
        pypl.fill_between(
            fitparam[0],
            np.average(fitparam[1], axis=0) - np.std(fitparam[1], axis=0),
            np.average(fitparam[1], axis=0) + np.std(fitparam[1], axis=0),
            alpha=0.3,
        )

    pypl.xlabel(r"$\textrm{t/a}$", labelpad=14, fontsize=18)
    pypl.ylabel(ylabel, labelpad=5, fontsize=18)
    # pypl.ylabel(r'$\Delta E/\lambda$',labelpad=5,fontsize=18)
    # pypl.title(r'Energy shift '+pars.momfold[pars.momentum][:-1]+r', $\gamma_{'+op[1:]+r'}$')

    pypl.ylim(ylim)
    pypl.xlim(0, 28)
    pypl.grid(True, alpha=0.4)
    # metadata["Title"] = plotname.split("/")[-1][:-4]
    metadata["Title"] = plotname
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=metadata)
    # pypl.show()
    pypl.close()
    return


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


def correlator_fitting(correlator, timerange, function, p0=[1, 1]):
    time = np.arange(0, np.shape(correlator)[1])[timerange]
    yaverage = np.average(correlator, axis=0)[timerange]
    ystd = np.std(correlator, axis=0)[timerange]
    yvalues = correlator
    covmat = np.diag(ystd)
    popt, pcov = curve_fit(function, time, yaverage, sigma=covmat)
    paramboots = []
    for bootval in correlator:
        popt1, pcov1 = curve_fit(function, time, bootval[timerange], sigma=covmat)
        paramboots.append(popt1)
    return popt, np.array(paramboots)


def fit1(ratio_unpert, datadir):
    """Fit to the 2pt. function ratio with a loop over fit windows"""
    func_const = ff.initffncs("Constant")
    time_limits = [[8, 19], [16, 25]]
    fitlist = stats.fit_loop(ratio_unpert, func_const, time_limits)

    # Write the fitresults to a file
    filename = datadir / "unpert_twopt_q000_fit.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    # Calculate the weights of each fit
    doflist = np.array([i["dof"] for i in fitlist])
    chisqlist = np.array([i["redchisq"] for i in fitlist]) * doflist
    errorlist = np.array([np.std(i["param"], axis=0)[0] for i in fitlist])
    weightlist = stats.weights(doflist, chisqlist, errorlist)
    for i, elem in enumerate(fitlist):
        elem["weight"] = weightlist[i]

    # pypl.figure()
    # pypl.plot(np.arange(len(weightlist)), weightlist)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / ("unpert_ratio_Beane_weights.pdf"), metadata=metadata)
    # # pypl.show()

    print(f"{weightlist=}")
    bestweight = np.argmax(weightlist)
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")

    ratio_z_factors = np.sqrt(fitlist[bestweight]["param"][:, 0] * 2 / (1 + m_N / m_S))
    print(
        "\nratio of z factors =",
        err_brackets(np.average(ratio_z_factors), np.std(ratio_z_factors)),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            ff.constant(fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0),
                np.std(fitlist[bestweight]["param"], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        ratio_unpert,
        "unpert_ratio_TBC16_loop",
        plotdir,
        ylim=(-0.2, 0.4),
        fitparam=fitparam,
        ylabel=r"$G_{N}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    )


def fit2(ratio_unpert, datadir):
    func_const = ff.initffncs("Constant")
    # time_limits = [[10, 19], [16, 23]]
    time_limits = [[8, 19], [16, 25]]
    fitlist = stats.fit_loop_bayes(ratio_unpert, func_const, time_limits)

    # Write the fitresults to a file
    filename = datadir / "unpert_twopt_q000_fit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    print(f"{np.shape(ratio_unpert)[1]=}")

    # Calculate the weights of each fit
    chisq_list = np.array([i["chisq"] for i in fitlist])
    val_list = np.array([np.average(i["param"], axis=0)[0] for i in fitlist])
    error_list = np.array([np.std(i["param"], axis=0)[0] for i in fitlist])
    Ncut = np.array([np.shape(ratio_unpert)[1] - len(i["x"]) for i in fitlist])
    parnum = np.array([len(i["paramavg"]) for i in fitlist])
    AIC1 = (
        -2 * np.log(1 / (4 * len(fitlist)))
        + np.array(chisq_list)
        + 2 * parnum
        + 2 * Ncut
    )
    prMD = np.exp(-2 * AIC1)
    # Normalize probabilities
    prMDsum = np.sum(prMD)  # +np.sum(prMD2)+np.sum(prMD3)
    prMD = prMD / prMDsum

    # pypl.figure()
    # pypl.plot(np.arange(len(prMD)), prMD)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / ("unpert_ratio_Bayesian_weights.pdf"), metadata=metadata)
    # # pypl.show()

    bestweight = np.argmax(prMD)
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
    print(f"{fitlist[bestweight]['x']=}")
    print(f"{fitlist[bestweight]['chisq']=}")
    print(f"{fitlist[bestweight]['redchisq']=}")

    ratio_z_factors = np.sqrt(fitlist[bestweight]["param"][:, 0] * 2 / (1 + m_N / m_S))
    print(
        "\nratio of z factors =",
        err_brackets(np.average(ratio_z_factors), np.std(ratio_z_factors)),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            ff.constant(fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0),
                np.std(fitlist[bestweight]["param"], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        ratio_unpert,
        "unpert_ratio_TBC16_loop_BAYES",
        plotdir,
        ylim=(-0.2, 0.4),
        fitparam=fitparam,
        ylabel=r"$G_{N}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    )
    return ratio_z_factors


def fit3(correlator, datadir, time_limits, filename):
    """Fit an axponential to the nucleon correlator"""
    func_aexp = ff.initffncs("Aexp")
    fitlist = stats.fit_loop_bayes(correlator, func_aexp, time_limits)

    # Write the fitresults to a file
    # filename = datadir / "nucl_twopt_q100_fit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    # Calculate the weights of each fit
    chisq_list = np.array([i["chisq"] for i in fitlist])
    val_list = np.array([np.average(i["param"], axis=0)[1] for i in fitlist])
    error_list = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
    Ncut = np.array([np.shape(correlator)[1] - len(i["x"]) for i in fitlist])
    parnum = np.array([len(i["paramavg"]) for i in fitlist])
    AIC1 = (
        -2 * np.log(1 / (4 * len(fitlist)))
        + np.array(chisq_list)
        + 2 * parnum
        + 2 * Ncut
    )
    prMD = np.exp(-2 * AIC1)
    # Normalize probabilities
    prMDsum = np.sum(prMD)  # +np.sum(prMD2)+np.sum(prMD3)
    prMD = prMD / prMDsum

    # pypl.figure()
    # pypl.plot(np.arange(len(prMD)), prMD)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / ("nucl_Bayesian_weights.pdf"), metadata=metadata)
    # # pypl.show()

    bestweight = np.argmax(prMD)
    # bestweight = 0
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
    print(f"{fitlist[bestweight]['x']=}")
    print(f"{fitlist[bestweight]['chisq']=}")
    print(f"{fitlist[bestweight]['redchisq']=}")

    z_factor_nucl_best = np.sqrt(
        np.abs(fitlist[bestweight]["param"][:, 0] * m_S / (m_S + m_N))
    )
    print(
        "z-factor nucleon =",
        err_brackets(np.average(z_factor_nucl_best), np.std(z_factor_nucl_best)),
    )

    fitparam_plot = {
        "x": fitlist[bestweight]["x"],
        "y": np.array(
            [
                stats.effmass(fitlist[bestweight]["fitfunction"](np.arange(64), i))
                for i in fitlist[bestweight]["param"]
            ]
        )[:, fitlist[bestweight]["x"]],
        "label": r"$\chi^2_{\textrm{dof}} = $"
        + f"{fitlist[bestweight]['redchisq']:0.2f}",
        "redchisq": fitlist[bestweight]["redchisq"],
    }
    stats.ploteffmass(
        fitlist[bestweight]["y"],
        "nucl_TBC16_loop_BAYES",
        plotdir,
        ylim=(0, 1.8),
        # ylim=None,
        fitparam=fitparam_plot,
        xlim=30,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    return z_factor_nucl_best


def fit4(correlator, datadir, time_limits, filename):
    """Fit an axponential to the sigma correlator"""
    func_aexp = ff.initffncs("Aexp")
    fitlist = stats.fit_loop_bayes(correlator, func_aexp, time_limits)
    print(f"{len(fitlist)=}")

    # Write the fitresults to a file
    # filename = datadir / "sigma_twopt_q000_fit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    chisq_list = np.array([i["chisq"] for i in fitlist])
    val_list = np.array([np.average(i["param"], axis=0)[1] for i in fitlist])
    error_list = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
    Ncut = np.array([np.shape(correlator)[1] - len(i["x"]) for i in fitlist])
    parnum = np.array([len(i["paramavg"]) for i in fitlist])
    AIC1 = (
        -2 * np.log(1 / (4 * len(fitlist)))
        + np.array(chisq_list)
        + 2 * parnum
        + 2 * Ncut
    )
    prMD = np.exp(-2 * AIC1)
    # Normalize probabilities
    prMDsum = np.sum(prMD)  # +np.sum(prMD2)+np.sum(prMD3)
    prMD = prMD / prMDsum

    # pypl.figure()
    # pypl.plot(np.arange(len(prMD)), prMD)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / ("sigma_Bayesian_weights.pdf"), metadata=metadata)
    # # pypl.show()
    # pypl.close()

    bestweight = np.argmax(prMD)
    # bestweight = 0
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
    print(f"{fitlist[bestweight]['x']=}")
    print(f"{fitlist[bestweight]['chisq']=}")
    print(f"{fitlist[bestweight]['redchisq']=}")

    z_factor_sigma_best = np.sqrt(np.abs(fitlist[bestweight]["param"][:, 0] / 2))
    print(
        "z-factor sigma =",
        err_brackets(np.average(z_factor_sigma_best), np.std(z_factor_sigma_best)),
    )

    fitparam_plot = {
        "x": fitlist[bestweight]["x"],
        "y": np.array(
            [
                stats.effmass(fitlist[bestweight]["fitfunction"](np.arange(64), i))
                for i in fitlist[bestweight]["param"]
            ]
        )[:, fitlist[bestweight]["x"]],
        "label": r"$\chi^2_{\textrm{dof}} = $"
        + f"{fitlist[bestweight]['redchisq']:0.2f}",
        "redchisq": fitlist[bestweight]["redchisq"],
    }
    stats.ploteffmass(
        fitlist[bestweight]["y"],
        "sigma_TBC16_loop_BAYES",
        plotdir,
        ylim=(0, 1.8),
        xlim=30,
        # ylim=None,
        fitparam=fitparam_plot,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    return z_factor_sigma_best


def plot3(correlator, datadir):
    """Fit an axponential to the nucleon correlator"""

    func_aexp = ff.initffncs("Aexp")
    filename = datadir / "nucl_twopt_q000_fit_bayes.pkl"
    with open(filename, "rb") as file_in:
        fitlist = pickle.load(file_in)

    # Calculate the weights of each fit
    chisq_list = np.array([i["chisq"] for i in fitlist])
    val_list = np.array([np.average(i["param"], axis=0)[1] for i in fitlist])
    error_list = np.array([np.std(i["param"], axis=0)[1] for i in fitlist])
    Ncut = np.array([np.shape(correlator)[1] - len(i["x"]) for i in fitlist])
    parnum = np.array([len(i["paramavg"]) for i in fitlist])
    AIC1 = (
        -2 * np.log(1 / (4 * len(fitlist)))
        + np.array(chisq_list)
        + 2 * parnum
        + 2 * Ncut
    )
    prMD = np.exp(-2 * AIC1)
    # Normalize probabilities
    prMDsum = np.sum(prMD)  # +np.sum(prMD2)+np.sum(prMD3)
    prMD = prMD / prMDsum

    # pypl.figure()
    # pypl.plot(np.arange(len(prMD)), prMD)
    # pypl.grid(True, alpha=0.4)
    # pypl.savefig(plotdir / ("nucl_Bayesian_weights.pdf"), metadata=metadata)
    # # pypl.show()
    # pypl.close()

    bestweight = np.argmax(prMD)
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
    print(f"{fitlist[bestweight]['x']=}")
    print(f"{fitlist[bestweight]['chisq']=}")
    print(f"{fitlist[bestweight]['redchisq']=}")

    z_factor_nucl_best = np.sqrt(
        np.abs(fitlist[bestweight]["param"][:, 0] * m_S / (m_S + m_N))
    )
    print(
        "z-factor nucleon =",
        err_brackets(np.average(z_factor_nucl_best), np.std(z_factor_nucl_best)),
    )

    fitparam_plot = {
        "x": fitlist[bestweight]["x"],
        "y": np.array(
            [
                stats.effmass(fitlist[bestweight]["fitfunction"](np.arange(64), i))
                for i in fitlist[bestweight]["param"]
            ]
        )[:, fitlist[bestweight]["x"]],
        "label": r"$\chi^2_{\textrm{dof}} = $"
        + f"{fitlist[bestweight]['redchisq']:0.2f}",
        "redchisq": fitlist[bestweight]["redchisq"],
    }
    stats.ploteffmass(
        fitlist[bestweight]["y"],
        "nucl_TBC16_loop_BAYES",
        plotdir,
        ylim=(0, 1.8),
        # ylim=None,
        fitparam=fitparam_plot,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    return z_factor_nucl_best


def fit_unpert_1(ratio_unpert, datadir, time_limits):
    func_const = ff.initffncs("Constant")
    fitlist = stats.fit_loop_bayes(ratio_unpert, func_const, time_limits)

    # Write the fitresults to a file
    filename = datadir / "unpert_ratio_onefit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    ratio_z_factors = np.sqrt(fitlist[bestweight]["param"][:, 0] * 2 / (1 + m_N / m_S))
    print(
        "\nratio of z factors =",
        err_brackets(np.average(ratio_z_factors), np.std(ratio_z_factors)),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            ff.constant(fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0),
                np.std(fitlist[bestweight]["param"], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        ratio_unpert,
        "unpert_ratio_BAYES",
        plotdir,
        ylim=(-0.2, 0.4),
        fitparam=fitparam,
        ylabel=r"$G_{N}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    )
    stats.ploteffmass(
        fitlist[bestweight]["y"],
        "unpert_effmass_ratio_BAYES",
        plotdir,
        ylim=(-0.4, 0.4),
        # ylim=None,
        xlim=30,
        fitparam=None,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    return ratio_z_factors


def fit_slope_nucl(correlator, datadir, time_limits, ratio_z_factors):
    fit_func = ff.initffncs("Linear")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits)

    # Write the fitresults to a file
    filename = datadir / "nucl_neg_ratio_onefit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    matrix_element_nucl_neg = fitlist[bestweight]["param"][:, 0] * ratio_z_factors
    print(
        "matrix element = ",
        err_brackets(
            np.average(matrix_element_nucl_neg), np.std(matrix_element_nucl_neg)
        ),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "slope = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        correlator,
        "nucl_neg_ratio_BAYES",
        plotdir,
        ylim=(-0.5, 200),
        fitparam=fitparam,
        ylabel=r"$G_{N}^{3}(\mathbf{p}')/G_{N}(\mathbf{p}')$",
    )
    return matrix_element_nucl_neg


def fit_slope_nucl_avg(correlator, datadir, time_limits, ratio_z_factors):
    fit_func = ff.initffncs("Linear")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits)

    # Write the fitresults to a file
    filename = datadir / "nucl_avg_ratio_onefit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    matrix_element_nucl_avg = fitlist[bestweight]["param"][:, 0] * ratio_z_factors
    print(
        "matrix element = ",
        err_brackets(
            np.average(matrix_element_nucl_avg), np.std(matrix_element_nucl_avg)
        ),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "slope = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        correlator,
        "nucl_avg_ratio_BAYES",
        plotdir,
        ylim=(-0.5, 200),
        fitparam=fitparam,
        ylabel=r"$2G_{N}^{3}(\mathbf{p}')/(G_{N}(\mathbf{p}')+G_{N}(-\mathbf{p}'))$",
    )
    return matrix_element_nucl_avg


def fit_slope_sigma(correlator, datadir, time_limits, ratio_z_factors):
    fit_func = ff.initffncs("Linear")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits)

    # Write the fitresults to a file
    filename = datadir / "sigma_ratio_onefit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    matrix_element_sigma = fitlist[bestweight]["param"][:, 0] / (
        ratio_z_factors * 0.5 * (1 + m_N / m_S)
    )
    print(
        "matrix element = ",
        err_brackets(np.average(matrix_element_sigma), np.std(matrix_element_sigma)),
    )

    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "slope = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }
    plot_correlator(
        correlator,
        "sigma_ratio_BAYES",
        plotdir,
        ylim=(-0.5, 25),
        fitparam=fitparam,
        ylabel=r"$G_{N}^{3}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    )
    return matrix_element_sigma


def fit_ratio_summed(
    correlator, datadir, time_limits, z_factor_nucl_best, z_factor_sigma_best
):
    fit_func = ff.initffncs("Constant")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits, disp=True)

    # Write the fitresults to a file
    filename = datadir / "summed_ratio_onefit_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0

    print(f"{np.shape(fitlist[bestweight]['param'][:, 0])=}")
    matrix_element_summed = (
        fitlist[bestweight]["param"][:, 0]
        * z_factor_nucl_best
        * z_factor_sigma_best
        * 2
    )
    print(
        "\nmatrix element summed =",
        err_brackets(np.average(matrix_element_summed), np.std(matrix_element_summed)),
    )
    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }

    plot_correlator(
        correlator,
        "summed_ratio_BAYES",
        plotdir,
        ylim=(-6e-39, 2e-39),
        fitparam=fitparam,
        ylabel=r"$G_{N}^3(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{N}(\tau;\mathbf{p}') G_{\Sigma}(t-\tau;\mathbf{0})\right)$",
    )
    return matrix_element_summed


def fit_ratio_summed_sigma(
    correlator, datadir, time_limits, z_factor_nucl_best, z_factor_sigma_best
):
    fit_func = ff.initffncs("Constant")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits)

    # Write the fitresults to a file
    filename = datadir / "summed_ratio_sigma_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    print(f"{np.shape(fitlist[bestweight]['param'][:, 0])=}")
    print(f"{np.shape(z_factor_nucl_best)=}")
    matrix_element_summed_sigma = (
        fitlist[bestweight]["param"][:, 0]
        * z_factor_sigma_best ** 3
        * 4
        / z_factor_nucl_best  # * m_S / m_N
        / (1 + m_N / m_S)
    )
    print(
        "\nmatrix element summed =",
        err_brackets(
            np.average(matrix_element_summed_sigma), np.std(matrix_element_summed_sigma)
        ),
    )
    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }

    plot_correlator(
        correlator,
        "summed_ratio_sigma_BAYES",
        plotdir,
        ylim=(-0.8e-39, 0.2e-39),
        fitparam=fitparam,
        ylabel=r"$G_{N}^3(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{N}(\tau;\mathbf{p}') G_{\Sigma}(t-\tau;\mathbf{0})\right)$",
    )
    return matrix_element_summed_sigma


def fit_ratio_summed_nucl(
    correlator, datadir, time_limits, z_factor_nucl_best, z_factor_sigma_best
):
    fit_func = ff.initffncs("Constant")
    fitlist = stats.fit_loop_bayes(correlator, fit_func, time_limits)

    # Write the fitresults to a file
    filename = datadir / "summed_ratio_nucl_bayes.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)

    bestweight = 0
    print(f"{np.shape(fitlist[bestweight]['param'][:, 0])=}")
    print(f"{np.shape(z_factor_nucl_best)=}")
    matrix_element_summed_nucl = (
        fitlist[bestweight]["param"][:, 0]
        * z_factor_nucl_best ** 3
        * (1 + m_N / m_S)
        / z_factor_sigma_best  # * m_S / m_N
    )
    print(
        "\nmatrix element summed =",
        err_brackets(
            np.average(matrix_element_summed_nucl), np.std(matrix_element_summed_nucl)
        ),
    )
    fitparam = {
        "x": fitlist[bestweight]["x"],
        "y": [
            fitlist[bestweight]["fitfunction"](fitlist[bestweight]["x"], i)
            for i in fitlist[bestweight]["param"]
        ],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitlist[bestweight]["param"], axis=0)[0],
                np.std(fitlist[bestweight]["param"], axis=0)[0],
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {fitlist[bestweight]['redchisq']:0.2}$",
    }

    plot_correlator(
        correlator,
        "summed_ratio_nucl_BAYES",
        plotdir,
        ylim=(-7e-38, 2e-39),
        fitparam=fitparam,
        ylabel=r"$G_{N}^3(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{N}(\tau;\mathbf{p}') G_{N}(t-\tau;\mathbf{0})\right)$",
    )
    return matrix_element_summed_nucl


def gevp(corr_matrix, time_choice=10, delta_t=1, name="", show=None):
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
    print(np.shape(Gt1))
    Gt2 = np.einsum("i,ijkl,j->kl", vl[:, 1], corr_matrix, vr[:, 1])
    print(np.shape(Gt2))

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
    corr_matrix, corr_matrix1, corr_matrix2, lmb_val, name="", show=False
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
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0)$",
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
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
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
        time[:xlim] + 0.3,
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
    pypl.savefig(plotdir / ("comp_plot_all" + name + ".pdf"))
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
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn2/plots/")
    datadir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn2/data/")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    momenta = ["mass"]
    lambdas = [0.005, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    # quarks = ["quark2"]
    conf_num = 43
    lmb_val = lambdas[6]

    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = list(
        (
            pickledir
            / Path(
                "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                # + mom_strings[2]
                + "p+1+0+0/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    print(unpertfile_nucleon_pos)
    for filename in unpertfile_nucleon_pos:
        G2_unpert_qp100_nucl = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # print(f"{np.shape(G2_unpert_qp100_nucl)=}")
        # stats.ploteffmass(
        #     G2_unpert_qp100_nucl[:, :, 0], "neutron_unpert", plotdir, show=False
        # )
    ### ----------------------------------------------------------------------
    unpertfile_sigma = list(
        (
            pickledir
            / Path(
                "baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    for filename in unpertfile_sigma:
        G2_unpert_q000_sigma = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # print(f"{np.shape(G2_unpert_q000_sigma)=}")
        # stats.ploteffmass(
        #     G2_unpert_q000_sigma[:, :, 0], "sigma_unpert", plotdir, show=False
        # )

    ratio = G2_unpert_qp100_nucl[:, :, 0] / G2_unpert_q000_sigma[:, :, 0]
    # stats.plot_correlator(ratio, "ratio", plotdir, show=False, ylim=(-0.2, 0.3))

    ### ----------------------------------------------------------------------
    # Perturbed correlators
    ### ----------------------------------------------------------------------

    ### ----------------------------------------------------------------------
    ### SD
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SD_lmb_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # + "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_SD_lmb = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(G2_q100_SD_lmb[:, :, 0], "SD_lmb", plotdir, show=False)
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SD_lmb3_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_SD_lmb3 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q100_SD_lmb_lmb3[:, :, 0], "SD_lmb+lmb3", plotdir, show=False
        # )

    ### ----------------------------------------------------------------------
    ### DS
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DS_lmb_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[0]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DS_lmb = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(G2_q100_DS_lmb[:, :, 0], "DS_lmb", plotdir, show=False)
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DS_lmb3_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[0]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DS_lmb3 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q100_DS_lmb_lmb3[:, :, 0], "DS_lmb+lmb3", plotdir, show=False
        # )

    ### ----------------------------------------------------------------------
    ### DD
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_lmb2_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DD_lmb2 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q100_DD_lmb2[:, :, 0],
        #     "DD_lmb0+lmb2",
        #     plotdir,
        #     show=False,
        # )
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_lmb4_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DD_lmb4 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q100_DD_lmb4[:, :, 0],
        #     "DD_lmb0+lmb2+lmb4",
        #     plotdir,
        #     show=False,
        # )

    ### ----------------------------------------------------------------------
    ### SS
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SS_lmb2_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_SS_lmb2 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q000_SS_lmb2[:, :, 0],
        #     "SS_lmb0+lmb2",
        #     plotdir,
        #     show=False,
        # )
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SS_lmb4_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_SS_lmb4 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # stats.ploteffmass(
        #     G2_q000_SS_lmb4[:, :, 0],
        #     "SS_lmb0+lmb2+lmb4",
        #     plotdir,
        #     show=False,
        # )

    ### ----------------------------------------------------------------------
    ### Construct correlation matrix
    matrix_1 = np.array(
        [
            [G2_unpert_qp100_nucl[:, :, 0], lmb_val * G2_q100_DS_lmb[:, :, 0]],
            [lmb_val * G2_q100_SD_lmb[:, :, 0], G2_unpert_q000_sigma[:, :, 0]],
        ]
    )
    matrix_2 = np.array(
        [
            [
                G2_unpert_qp100_nucl[:, :, 0] + lmb_val ** 2 * G2_q100_DD_lmb2[:, :, 0],
                lmb_val * G2_q100_DS_lmb[:, :, 0],
            ],
            [
                lmb_val * G2_q100_SD_lmb[:, :, 0],
                G2_unpert_q000_sigma[:, :, 0] + lmb_val ** 2 * G2_q000_SS_lmb2[:, :, 0],
            ],
        ]
    )
    matrix_3 = np.array(
        [
            [
                G2_unpert_qp100_nucl[:, :, 0] + lmb_val ** 2 * G2_q100_DD_lmb2[:, :, 0],
                lmb_val * G2_q100_DS_lmb[:, :, 0]
                + lmb_val ** 3 * G2_q100_DS_lmb3[:, :, 0],
            ],
            [
                lmb_val * G2_q100_SD_lmb[:, :, 0]
                + lmb_val ** 3 * G2_q100_SD_lmb3[:, :, 0],
                G2_unpert_q000_sigma[:, :, 0] + lmb_val ** 2 * G2_q000_SS_lmb2[:, :, 0],
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
    print(f"{np.shape(matrix_1)=}")
    ### ----------------------------------------------------------------------
    plotting_script_all(
        matrix_1 / 1e39,
        matrix_2 / 1e39,
        matrix_3 / 1e39,
        lmb_val,
        name="_l" + str(lmb_val),
        show=False,
    )
    plotting_script_nucl(matrix_1, matrix_2, matrix_3, name="", show=False)
    plotting_script_sigma(matrix_1, matrix_2, matrix_3, name="", show=False)
    plotting_script_nucl_sigma(matrix_1, matrix_3, name="", show=False)
    plotting_script_sigma_nucl(matrix_1, matrix_3, name="", show=False)
    # exit()

    Gt1_1, Gt2_1 = gevp(matrix_1, time_choice=13, delta_t=1, name="_test", show=False)
    effmassdata_1 = stats.bs_effmass(Gt1_1, time_axis=1, spacing=1)
    effmassdata_2 = stats.bs_effmass(Gt2_1, time_axis=1, spacing=1)
    diffG1 = (effmassdata_1 - effmassdata_2) / lmb_val / 2
    # plotting_script2(diffG1, name="_l" + str(lmb_val) + "_1", show=False)
    # plotting_script3(matrix_1, Gt1_1, Gt2_1, name="_l" + str(lmb_val) + "_1")

    Gt1_2, Gt2_2 = gevp(matrix_2, time_choice=13, delta_t=1, name="_test", show=False)
    effmassdata_1 = stats.bs_effmass(Gt1_2, time_axis=1, spacing=1)
    effmassdata_2 = stats.bs_effmass(Gt2_2, time_axis=1, spacing=1)
    diffG2 = (effmassdata_1 - effmassdata_2) / lmb_val / 2
    # plotting_script2(diffG1, name="_l" + str(lmb_val) + "_2", show=False)
    # plotting_script3(matrix_2, Gt1_2, Gt2_2, name="_l" + str(lmb_val) + "_2")

    Gt1_3, Gt2_3 = gevp(matrix_3, time_choice=13, delta_t=1, name="_test", show=False)
    # plotting_script3(matrix_3, Gt1_3, Gt2_3, name="_l" + str(lmb_val) + "_3")
    effmassdata_1_3 = stats.bs_effmass(Gt1_3, time_axis=1, spacing=1)
    effmassdata_2_3 = stats.bs_effmass(Gt2_3, time_axis=1, spacing=1)
    diffG3 = (effmassdata_1_3 - effmassdata_2_3) / lmb_val / 2

    Gt1_4, Gt2_4 = gevp(matrix_4, time_choice=13, delta_t=1, name="_test", show=False)
    # plotting_script3(matrix_4, Gt1_4, Gt2_4, name="_l" + str(lmb_val) + "_4")

    plotting_script_diff(
        diffG1,
        diffG2,
        diffG3,
        diffG3,
        lmb_val,
        name="_l" + str(lmb_val) + "_all",
        show=True,
    )

    exit()
    ### ----------------------------------------------------------------------
    ### Diagonalise the matrix
    # wl, vl = np.linalg.eig(matmul(Gtpdt, la.inv(Gt)).T)
    # wr, vr = np.linalg.eig(matmul(la.inv(Gt), Gtpdt))
    time_choice = 10
    delta_t = 1
    mat_0 = np.average(matrix_1[:, :, :, time_choice], axis=2)
    mat_1 = np.average(matrix_1[:, :, :, time_choice + delta_t], axis=2)

    wl, vl = np.linalg.eig(np.matmul(mat_1, np.linalg.inv(mat_0)).T)
    wr, vr = np.linalg.eig(np.matmul(np.linalg.inv(mat_0), mat_1))
    print(wl, vl)
    print(wr, vr)

    Gt1 = np.einsum("i,ijkl,j->kl", vl[:, 0], matrix_1, vr[:, 0])
    print(np.shape(Gt1))
    stats.ploteffmass(Gt1, "eig_1", plotdir, show=True)

    Gt2 = np.einsum("i,ijkl,j->kl", vl[:, 1], matrix_1, vr[:, 1])
    print(np.shape(Gt2))
    stats.ploteffmass(Gt2, "eig_2", plotdir, show=True)

    # print(f"{np.shape(mat)=}")
    # print(mat)
    # wl, vl = np.linalg.eig(mat.T)
    # wr, vr = np.linalg.eig(mat)
    # print(wl)
    # print(vl)
    # print(wr, vr)

    ### ----------------------------------------------------------------------
    ### ----------------------------------------------------------------------
    exit()
    ### ----------------------------------------------------------------------

    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_unp+lmb2+lmb4_0.04_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    for filename in filelist:
        G2_q100_DD_lmb4_lp04 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q100_DD_lmb4_lp04[:, :, 0],
            "DD_lmb0+lmb2+lmb4_lp04",
            plotdir,
            show=True,
        )

    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_unp+lmb2+lmb4_0.005_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
            )
        ).glob("barspec_nucleon_rel" + "_" + str(conf_num) + "cfgs.pickle")
    )
    for filename in filelist:
        G2_q100_DD_lmb4_lp005 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q100_DD_lmb4_lp005[:, :, 0],
            "DD_lmb0+lmb2+lmb4_lp005",
            plotdir,
            show=True,
        )

    ### ----------------------------------------------------------------------
    ratio3 = G2_q100_DD_lmb4_lp005[:, :, 0] / G2_unpert_q000_sigma[:, :, 0]
    stats.plot_correlator(ratio3, "ratio_DD_lmb4_lp005", plotdir, show=True)

    # ### ----------------------------------------------------------------------
    # filelist = list(
    #     (
    #         pickledir
    #         / Path(
    #             "baryon-3pt_DD_unp+lmb2+lmb4_0.005_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
    #         )
    #     ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    # )
    # for filename in filelist:
    #     G2_q100_DD_lmb4 = read_pickle(filename, nboot=pars.nboot, nbin=1)
    #     print(f"{np.shape(G2_q100_DD_lmb4)=}")
    #     stats.ploteffmass(G2_q100_DD_lmb4[:, :, 0], "DD_lmb4", plotdir, show=True)

    exit()

    ### ----------------------------------------------------------------------
    ### ----------------------------------------------------------------------
    ### ----------------------------------------------------------------------

    unpertfile_nucleon_neg = (
        evxptdir1
        / Path(momenta[0])
        / Path("rel/dump")
        / Path("TBC16/nucl_neg/dump.res")
    )
    unpertfile_sigma = (
        evxptdir1 / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/sigma/dump.res")
    )
    fh_file_pos = (
        evxptdir1 / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/FH_pos/dump.res")
    )
    fh_file_neg = (
        evxptdir1 / Path(momenta[0]) / Path("rel/dump") / Path("TBC16/FH_neg/dump.res")
    )

    ### ----------------------------------------------------------------------
    # Read the correlator data from evxpt dump.res files
    G2_unpert_qp100_nucl = evxptdata(
        unpertfile_nucleon_pos, numbers=[0, 1], nboot=500, nbin=1
    )
    G2_unpert_qm100_nucl = evxptdata(
        unpertfile_nucleon_neg, numbers=[0, 1], nboot=500, nbin=1
    )
    G2_unpert_q000_sigma = evxptdata(
        unpertfile_sigma, numbers=[0, 1], nboot=500, nbin=1
    )
    G2_q100_pos = evxptdata(fh_file_pos, numbers=[0, 1], nboot=500, nbin=1)
    G2_q100_neg = evxptdata(fh_file_neg, numbers=[0, 1], nboot=500, nbin=1)

    ### ----------------------------------------------------------------------
    # Average over the normal time and time-reversed correlators
    G2_unpert_q000_sigma_tavg = np.sum(G2_unpert_q000_sigma, axis=1) / 2
    G2_unpert_qp100_nucl_tavg = np.sum(G2_unpert_qp100_nucl, axis=1) / 2
    G2_unpert_qm100_nucl_tavg = np.sum(G2_unpert_qm100_nucl, axis=1) / 2
    # Average over the positive and negative momentum
    momaverage = 0.5 * (
        G2_unpert_qp100_nucl[:, 0, :, 0] + G2_unpert_qm100_nucl[:, 0, :, 0]
    )
    momaverage_tavg = 0.5 * (
        G2_unpert_qp100_nucl_tavg[:, :, 0] + G2_unpert_qm100_nucl_tavg[:, :, 0]
    )

    ### ----------------------------------------------------------------------
    ### Create the ratio of the two unperturbed correlators
    ratio_unpert = momaverage_tavg * G2_unpert_q000_sigma_tavg[:, :, 0] ** (-1)

    ### ----------------------------------------------------------------------
    time_limits = [[16, 17], [18, 29]]
    ### ----------------------------------------------------------------------

    # ### ----------------------------------------------------------------------
    # ### Fit to the 2pt. function ratio with a loop over fit windows
    # ratio_z_factors = fit_unpert_1(ratio_unpert, datadir, time_limits)
    # print(f"{np.average(ratio_z_factors)=}")

    # ### ----------------------------------------------------------------------
    # ### The ratio of the FH over nucl. neg
    # print("\nRatio of FH over nucl with negative momentum")
    # ratio_pert_nucl = G2_pert_q100_neg[:, 0, :, 0] * (
    #     G2_unpert_qm100_nucl[:, 0, :, 0]
    # ) ** (-1)
    # matrix_element_nucl_neg = fit_slope_nucl(
    #     ratio_pert_nucl, datadir, time_limits, ratio_z_factors
    # )

    # ### ----------------------------------------------------------------------
    # print("\n-->Ratio of FH neg mom over nucl momentum avgd")
    # ratio_pert_nucl_avg = G2_pert_q100_neg[:, 0, :, 0] * (momaverage) ** (-1)
    # matrix_element_nucl_avg = fit_slope_nucl_avg(
    #     ratio_pert_nucl_avg, datadir, time_limits, ratio_z_factors
    # )

    # ### ----------------------------------------------------------------------
    # ### The ratio of the FH neg over the sigma
    # print("\n-->Ratio of FH nucl with negative momentum over the Sigma")
    # ratio_pert_sigma = G2_pert_q100_neg[:, 0, :, 0] * (
    #     # G2_unpert_q000_sigma[:, 0, :, 0]
    #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # ) ** (-1)
    # matrix_element_sigma = fit_slope_sigma(
    #     ratio_pert_sigma, datadir, time_limits, ratio_z_factors
    # )

    # ### ----------------------------------------------------------------------
    # # time_limits2 = [[8, 17], [22, 27]]
    # time_limits2 = [[5, 19], [17, 29]]
    # filename1 = datadir / "nucl_twopt_q100_fit_bayes.pkl"
    # z_factor_nucl_best = fit3(
    #     G2_unpert_qm100_nucl_tavg[:, :, 0], datadir, time_limits2, filename1
    # )
    # filename2 = datadir / "sigma_twopt_q000_fit_bayes.pkl"
    # z_factor_sigma_best = fit4(
    #     G2_unpert_q000_sigma_tavg[:, :, 0], datadir, time_limits2, filename2
    # )

    # ratio_z2 = z_factor_nucl_best / z_factor_sigma_best
    # print(f"{np.average(ratio_z2)=}")

    ### ----------------------------------------------------------------------
    time_choice1 = np.array(
        [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    )
    time_choice2 = np.array(
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    )
    time_choice3 = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
    # time_choice1 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    filename = datadir / "nucl_twopt_q100_fit_bayes.pkl"
    with open(filename, "rb") as file_in:
        z_factor_nucl_best_fit = pickle.load(file_in)
    for ifit, fit in enumerate(z_factor_nucl_best_fit):
        # print(fit["x"])
        if np.array_equal(fit["x"], time_choice2):
            index = ifit
    # index = bayesian_weights(z_factor_nucl_best_fit)
    print(f"{index=}")
    print(z_factor_nucl_best_fit[index]["x"])
    z_factor_nucl_best = np.sqrt(
        np.abs(z_factor_nucl_best_fit[index]["param"][:, 0] * m_S / (m_S + m_N))
    )
    # ----------------------------------------------------------------------
    filename = datadir / "sigma_twopt_q000_fit_bayes.pkl"
    with open(filename, "rb") as file_in:
        z_factor_sigma_best_fit = pickle.load(file_in)
    for ifit, fit in enumerate(z_factor_sigma_best_fit):
        if np.array_equal(fit["x"], time_choice1):
            index = ifit
    # index = bayesian_weights(z_factor_sigma_best_fit)
    print(f"{index=}")
    print(z_factor_nucl_best_fit[index]["x"])
    z_factor_sigma_best = np.sqrt(
        np.abs(z_factor_sigma_best_fit[index]["param"][:, 0] / 2)
    )

    ### ----------------------------------------------------------------------
    ### The ratio of the 3pt. over the summed two-pt. functions of Nucl and Sigma
    print(
        "\n-->Ratio of FH nucl over the sum of the nucleon and sigma two-point functions"
    )
    ratio_summed = []
    timelength = len(G2_pert_q100_neg[0, 0, :, 0])
    # The first element has a div by zero otherwise
    ratio_summed.append(np.zeros(np.shape(G2_pert_q100_neg)[0]))
    for ti in np.arange(1, timelength):
        print(f"{ti=}")
        ratio_summed.append(
            G2_pert_q100_neg[:, 0, ti, 0]
            / np.sum(
                [
                    G2_unpert_qm100_nucl[:, 0, tau, 0]
                    * G2_unpert_q000_sigma[:, 0, ti - tau, 0]
                    for tau in np.arange(ti)
                ],
                axis=0,
            )
        )
    ratio_summed = np.moveaxis(ratio_summed, 0, 1)
    time_limits_summed = [[12, 21], [16, 30]]
    matrix_element_summed = fit_ratio_summed(
        ratio_summed,
        datadir,
        time_limits_summed,
        z_factor_nucl_best,
        z_factor_sigma_best,
    )

    ### ----------------------------------------------------------------------
    ### ----------------------------------------------------------------------
    exit()
    ### ----------------------------------------------------------------------
    ### ----------------------------------------------------------------------

    ### ----------------------------------------------------------------------
    ### The ratio of the 3pt. over two summed Sigma two-pt. functions
    print("\n-->Ratio of FH nucl over the sum of two sigma two-point functions")
    # time_limits3 = [[17, 17], [25, 25]]
    ratio_summed_sigma = []
    timelength = len(G2_pert_q100_neg[0, 0, :, 0])
    # The first element has a div by zero otherwise
    ratio_summed_sigma.append(np.zeros(np.shape(G2_pert_q100_neg)[0]))
    for ti in np.arange(1, timelength):
        ratio_summed_sigma.append(
            G2_pert_q100_neg[:, 0, ti, 0]
            / np.sum(
                np.abs(
                    [
                        G2_unpert_q000_sigma[:, 0, tau, 0]
                        * G2_unpert_q000_sigma[:, 0, ti - tau, 0]
                        for tau in np.arange(ti)
                    ]
                ),
                axis=0,
            )
        )
    ratio_summed_sigma = np.moveaxis(ratio_summed_sigma, 0, 1)
    matrix_element_summed_sigma = fit_ratio_summed_sigma(
        ratio_summed_sigma,
        datadir,
        time_limits,
        z_factor_nucl_best,
        z_factor_sigma_best,
    )

    ### ----------------------------------------------------------------------
    ### The ratio of the 3pt. over two summed nucl two-pt. functions
    print("\n-->Ratio of FH nucl over the sum of two nucleon two-point functions")
    ratio_summed_nucl = []
    timelength = len(G2_pert_q100_neg[0, 0, :, 0])
    # The first element has a div by zero otherwise
    ratio_summed_nucl.append(np.zeros(np.shape(G2_pert_q100_neg)[0]))
    for ti in np.arange(1, timelength):
        ratio_summed_nucl.append(
            G2_pert_q100_neg[:, 0, ti, 0]
            / np.sum(
                np.abs(
                    [
                        G2_unpert_qm100_nucl[:, 0, tau, 0]
                        * G2_unpert_qm100_nucl[:, 0, ti - tau, 0]
                        # G2_unpert_q000_nucl[:, 0, tau, 0]
                        # * G2_unpert_q000_nucl[:, 0, ti - tau, 0]
                        for tau in np.arange(ti)
                    ]
                ),
                axis=0,
            )
        )
    ratio_summed_nucl = np.moveaxis(ratio_summed_nucl, 0, 1)
    matrix_element_summed_nucl = fit_ratio_summed_nucl(
        ratio_summed_nucl,
        datadir,
        time_limits,
        z_factor_nucl_best,
        z_factor_nucl_best,
    )

    ### ----------------------------------------------------------------------
    ### Comparison plot of the results from the different ratios
    pypl.figure(figsize=(9, 6))
    pypl.errorbar(
        # ["nucl", "nucl negative", "sigma", "summed", "summed sigma", "summed nucl"],
        ["nucl", "sigma", "summed", "summed sigma", "summed nucl"],
        np.array(
            [
                np.average(matrix_element_nucl_avg),
                # np.average(matrix_element_nucl_neg),
                np.average(matrix_element_sigma),
                -np.average(matrix_element_summed),
                -np.average(matrix_element_summed_sigma),
                -np.average(matrix_element_summed_nucl),
            ]
        ),
        np.array(
            [
                np.std(matrix_element_nucl_avg),
                # np.std(matrix_element_nucl_neg),
                np.std(matrix_element_sigma),
                np.std(matrix_element_summed),
                np.std(matrix_element_summed_sigma),
                np.std(matrix_element_summed_nucl),
            ]
        ),
        capsize=4,
        elinewidth=1,
        color="b",
        fmt="s",
        markerfacecolor="none",
    )
    plotname = "matrix_elements"
    metadata["Title"] = plotname
    pypl.ylabel(r"$F_{1}- \frac{\mathbf{p}'^{2}}{(m_{N}+m_{\Sigma})^{2}} F_{2}$")
    pypl.xlabel("ratio type")
    pypl.ylim(0, 2.1)
    pypl.grid(True, alpha=0.4)
    pypl.savefig(plotdir / (plotname + ".pdf"), metadata=metadata)
    pypl.show()
    pypl.close()

    exit()
