import numpy as np
from pathlib import Path
import scipy.optimize as syopt
from scipy.optimize import curve_fit
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


metadata = {"Author": "Mischa Batelaan", "Creator": __file__}
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


if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    nboot = 200  # 700
    nbin = 1  # 10
    pickledir = Path.home() / Path(
        "Documents/PhD/analysis_results/six_point_fn/pickle/"
    )
    plotdir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn/plots/")
    datadir = Path.home() / Path("Documents/PhD/analysis_results/six_point_fn/data/")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    momenta = ["mass"]
    lambdas = [0.005, 0.02, 0.04]
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    # quarks = ["quark2"]

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
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    for filename in unpertfile_nucleon_pos:
        G2_unpert_qp100_nucl = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # print(f"{np.shape(G2_unpert_qp100_nucl)=}")
        stats.ploteffmass(
            G2_unpert_qp100_nucl[:, :, 0], "neutron_unpert", plotdir, show=False
        )
    ### ----------------------------------------------------------------------
    unpertfile_sigma = list(
        (
            pickledir
            / Path(
                "baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/p+0+0+0/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    for filename in unpertfile_sigma:
        G2_unpert_q000_sigma = read_pickle(filename, nboot=pars.nboot, nbin=1)
        # print(f"{np.shape(G2_unpert_q000_sigma)=}")
        stats.ploteffmass(
            G2_unpert_q000_sigma[:, :, 0], "sigma_unpert", plotdir, show=False
        )

    ratio = G2_unpert_qp100_nucl[:, :, 0] / G2_unpert_q000_sigma[:, :, 0]
    stats.plot_correlator(ratio, "ratio", plotdir, show=False, ylim=(-0.2, 0.3))

    ### ----------------------------------------------------------------------
    # Perturbed correlators
    ### ----------------------------------------------------------------------
    lmb_val = lambdas[2]

    ### ----------------------------------------------------------------------
    ### SD
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SD_lmb_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # + "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_SD_lmb = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(G2_q100_SD_lmb[:, :, 0], "SD_lmb", plotdir, show=False)
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SD_lmb+lmb3_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[2]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_SD_lmb_lmb3 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q100_SD_lmb_lmb3[:, :, 0], "SD_lmb+lmb3", plotdir, show=False
        )

    ### ----------------------------------------------------------------------
    ### DS
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DS_lmb_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[0]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DS_lmb = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(G2_q100_DS_lmb[:, :, 0], "DS_lmb", plotdir, show=False)
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DS_lmb+lmb3_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[0]  # "p+1+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q100_DS_lmb_lmb3 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q100_DS_lmb_lmb3[:, :, 0], "DS_lmb+lmb3", plotdir, show=False
        )

    ### ----------------------------------------------------------------------
    ### DD
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_unp+lmb2_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_DD_lmb2 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q000_DD_lmb2[:, :, 0],
            "DD_lmb0+lmb2",
            plotdir,
            show=False,
        )
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_DD_unp+lmb2+lmb4_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp121040/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_DD_lmb2_lmb4 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q000_DD_lmb2_lmb4[:, :, 0],
            "DD_lmb0+lmb2+lmb4",
            plotdir,
            show=False,
        )

    ### ----------------------------------------------------------------------
    ### SS
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SS_unp+lmb2_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_SS_lmb2 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q000_SS_lmb2[:, :, 0],
            "SS_lmb0+lmb2",
            plotdir,
            show=False,
        )
    ### ----------------------------------------------------------------------
    filelist = list(
        (
            pickledir
            / Path(
                "baryon-3pt_SS_unp+lmb2+lmb4_"
                + str(lmb_val)
                + "_TBC/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
                + mom_strings[1]  # "p+0+0+0/"
                + "/"
            )
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
    )
    # print(f"{filelist=}")
    for filename in filelist:
        G2_q000_SS_lmb2_lmb4 = read_pickle(filename, nboot=pars.nboot, nbin=1)
        stats.ploteffmass(
            G2_q000_SS_lmb2_lmb4[:, :, 0],
            "SS_lmb0+lmb2+lmb4",
            plotdir,
            show=False,
        )

    ### ----------------------------------------------------------------------
    ### Construct correlation matrix
    matrix_1 = np.array(
        [
            [G2_unpert_qp100_nucl[:, :, 0], G2_q100_DS_lmb[:, :, 0]],
            [G2_q100_SD_lmb[:, :, 0], G2_unpert_q000_sigma[:, :, 0]],
        ]
    )
    print(f"{np.shape(matrix_1)=}")
    ### ----------------------------------------------------------------------
    ### Diagonalise the matrix
    # wl, vl = np.linalg.eig(matmul(Gtpdt, la.inv(Gt)).T)
    # wr, vr = np.linalg.eig(matmul(la.inv(Gt), Gtpdt))
    time_choice = 10
    mat = np.average(matrix_1[:, :, :, time_choice], axis=2)
    print(f"{np.shape(mat)=}")
    print(mat)
    wl, vl = np.linalg.eig(mat.T)
    # wr, vr = np.linalg.eig(mat)
    print(wl)
    print(vl)
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
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
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
        ).glob("barspec_nucleon_rel" + "_[0-9]*cfgs.pickle")
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

    # ### ----------------------------------------------------------------------
    # ### ----------------------------------------------------------------------
    # ### ----------------------------------------------------------------------
    # ### ----------------------------------------------------------------------
    # ### The ratio of the FH pos over nucl. pos
    # print("\nRatio of FH over nucl with positive momentum")
    # ratio_pert_nucl = G2_pert_q100_pos[:, 0, :, 0] * (
    #     G2_unpert_qp100_nucl[:, 0, :, 0]
    # ) ** (-1)

    # fitrange = slice(9, 15)
    # popt, paramboots = correlator_fitting(
    #     ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    # )
    # print(
    #     "slope =",
    #     err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    # )
    # matrix_element = paramboots[:, 1] * ratio_z_factors
    # print("matrix element pos = ", np.average(matrix_element))
    # print(
    #     "matrix element pos = ",
    #     err_brackets(np.average(matrix_element), np.std(matrix_element)),
    # )

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
    #     "label": "slope ="
    #     + err_brackets(
    #         np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
    #     ),
    # }

    # plot_correlator(
    #     ratio_pert_nucl,
    #     "pert_ratio_nucl_pos_TBC16",
    #     plotdir,
    #     fitparam=fitparam,
    #     ylim=(-0.5, 90),
    #     ylabel=r"$G_{N}^{\lambda}(\mathbf{p}')/G_{N}(\mathbf{p}')$",
    # )

    ### ----------------------------------------------------------------------
    ### The ratio of the FH neg over nucl. neg
    print("\nRatio of FH over nucl with negative momentum")
    ratio_pert_nucl = G2_pert_q100_neg[:, 0, :, 0] * (
        G2_unpert_qm100_nucl[:, 0, :, 0]
    ) ** (-1)

    # fitrange = slice(9, 17)
    fitrange = slice(14, 19)
    popt, paramboots = correlator_fitting(
        ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    )
    print(
        "slope =",
        err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    )

    matrix_element_nucl_neg = paramboots[:, 1] * ratio_z_factors
    # print(f"{np.shape(paramboots[:,1])=}")
    # print(f"{np.shape(ratio_z_factors)=}")
    # print(f"{np.shape(matrix_element_nucl)=}")
    print(
        "matrix element = ",
        err_brackets(
            np.average(matrix_element_nucl_neg), np.std(matrix_element_nucl_neg)
        ),
    )

    fitparam = {
        "x": np.arange(64)[fitrange],
        "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
        "label": "slope ="
        + err_brackets(
            np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
        ),
    }

    plot_correlator(
        ratio_pert_nucl,
        "pert_ratio_nucl_neg_TBC16",
        plotdir,
        fitparam=fitparam,
        ylim=(-0.5, 200),
        ylabel=r"$G_{N}^{\lambda}(\mathbf{p}')/G_{N}(\mathbf{p}')$",
    )

    ### ----------------------------------------------------------------------
    ### We need the z-factors of the two-point functions
    ### NUCL
    unpert_nucl = G2_unpert_qm100_nucl_tavg

    ### ------------------------------------------------------------
    func_aexp = ff.initffncs("Aexp")
    # time_limits = [[7, 8], [14, 19]]
    time_limits = [[7, 8], [14, 19]]
    fitlist, weightlist = stats.fit_loop(
        G2_unpert_qm100_nucl_tavg[:, :, 0], func_aexp, time_limits
    )
    filename = plotdir / "unpert_twopt_q000_effmass.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)
    bestweight = np.argmax(weightlist)
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
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
        "unpert_twopt_q000_effmass_best",
        plotdir,
        ylim=(0, 1.8),
        # ylim=None,
        fitparam=fitparam_plot,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    ### ------------------------------------------------------------

    # fitrange = slice(9, 19)
    # popt, paramboots = correlator_fitting(
    #     G2_unpert_qm100_nucl_tavg[:, :, 0] / 1e37, fitrange, oneexp1, p0=[-1, 0.5]
    # )
    # paramboots[:, 0] = paramboots[:, 0] * 1e37
    # print(
    #     "amplitude =",
    #     err_brackets(np.average(paramboots, axis=0)[0], np.std(paramboots, axis=0)[0]),
    # )
    # z_factor_nucl = np.sqrt(np.abs(paramboots[:, 0] * m_S / (m_S + m_N)))
    # print(
    #     "z-factor nucleon =",
    #     err_brackets(np.average(z_factor_nucl), np.std(z_factor_nucl)),
    # )

    ### SIGMA
    ### ------------------------------------------------------------
    fitlist, weightlist = stats.fit_loop(
        G2_unpert_q000_sigma_tavg[:, :, 0], func_aexp, time_limits
    )
    filename = plotdir / "unpert_sigma_twopt_q000_effmass.pkl"
    with open(filename, "wb") as file_out:
        pickle.dump(fitlist, file_out)
    bestweight = np.argmax(weightlist)
    print(f"\n{bestweight=}")
    print(f"{fitlist[bestweight]['paramavg']=}")
    z_factor_sigma_best = np.sqrt(
        np.abs(fitlist[bestweight]["param"][:, 0] * m_S / (m_S + m_N))
    )
    print(
        "z-factor sigma =",
        err_brackets(np.average(z_factor_sigma_best), np.std(z_factor_sigma_best)),
    )
    z_factor_ratio_best = z_factor_nucl_best / z_factor_sigma_best
    print(
        "\nz-factor ratio =",
        err_brackets(np.average(z_factor_ratio_best), np.std(z_factor_ratio_best)),
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
        "unpert_sigma_twopt_q000_effmass_best",
        plotdir,
        ylim=(0, 1.8),
        # ylim=None,
        fitparam=fitparam_plot,
        # fitparam_q1=None,
        ylabel=None,
        show=False,
    )
    ### ------------------------------------------------------------
    # popt, paramboots = correlator_fitting(
    #     G2_unpert_q000_sigma_tavg[:, :, 0] / 1e37, fitrange, oneexp1, p0=[1, 0.1]
    # )
    # paramboots[:, 0] = paramboots[:, 0] * 1e37
    # z_factor_sigma = np.sqrt(np.abs(paramboots[:, 0] / 2))
    # print(
    #     "z-factor sigma =",
    #     err_brackets(np.average(z_factor_sigma), np.std(z_factor_sigma)),
    # )

    # print(f"{np.shape(z_factor_nucl)=}")
    # print(f"{np.shape(z_factor_sigma)=}")
    # ratio_z_factors2 = z_factor_nucl / z_factor_sigma
    # print(f"{np.average(ratio_z_factors2)=}")

    ### ----------------------------------------------------------------------
    ### The ratio of the two unperturbed correlators
    # print("\nThe ratio of the two unperturbed correlators pos mom. and trev avgd")
    # ratio_unpert = G2_unpert_qp100_nucl_tavg[:, :, 0] * (
    #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # ) ** (-1)
    # fitrange = slice(9, 19)
    # popt, paramboots = correlator_fitting(ratio_unpert, fitrange, ff.constant, p0=[0.1])
    # print("plateau =", err_brackets(np.average(paramboots), np.std(paramboots)))

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [ff.constant(np.arange(64)[fitrange], i) for i in paramboots],
    #     "label": "plateau ="
    #     + str(err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0))),
    # }

    # plotratio(ratio_unpert, 1, "unpert_eff_ratio_TBC16", plotdir, ylim=(-0.5, 0.5))
    # plot_correlator(
    #     ratio_unpert,
    #     "unpert_ratio_pos_TBC16",
    #     plotdir,
    #     ylim=(-0.2, 0.4),
    #     fitparam=fitparam,
    #     ylabel=r"$G_{N}(\mathbf{p}',\lambda=0)/G_{\Sigma}(\mathbf{0},\lambda=0)$",
    # )

    ### ----------------------------------------------------------------------
    ### The ratio of the two unperturbed correlators
    # print("\nThe ratio of the two unperturbed correlators neg mom. and trev avgd")
    # ratio_unpert = G2_unpert_qm100_nucl_tavg[:, :, 0] * (
    #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # ) ** (-1)
    # fitrange = slice(9, 19)
    # popt, paramboots = correlator_fitting(ratio_unpert, fitrange, ff.constant, p0=[0.1])
    # print("plateau =", err_brackets(np.average(paramboots), np.std(paramboots)))

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [ff.constant(np.arange(64)[fitrange], i) for i in paramboots],
    #     "label": "plateau ="
    #     + str(err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0))),
    # }

    # plotratio(ratio_unpert, 1, "unpert_eff_ratio_TBC16", plotdir, ylim=(-0.5, 0.5))
    # plot_correlator(
    #     ratio_unpert,
    #     "unpert_ratio_neg_TBC16",
    #     plotdir,
    #     ylim=(-0.2, 0.4),
    #     fitparam=fitparam,
    #     ylabel=r"$G_{N}(\mathbf{p}',\lambda=0)/G_{\Sigma}(\mathbf{0},\lambda=0)$",
    # )

    ### ----------------------------------------------------------------------
    ### The ratio of the FH neg over nucl. neg
    print("\nRatio of FH over nucl with negative momentum")
    ratio_pert_nucl = G2_pert_q100_neg[:, 0, :, 0] * (
        G2_unpert_qm100_nucl[:, 0, :, 0]
    ) ** (-1)

    # fitrange = slice(9, 17)
    fitrange = slice(14, 19)
    popt, paramboots = correlator_fitting(
        ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    )
    print(
        "slope =",
        err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    )

    matrix_element_nucl_neg = paramboots[:, 1] * ratio_z_factors
    # print(f"{np.shape(paramboots[:,1])=}")
    # print(f"{np.shape(ratio_z_factors)=}")
    # print(f"{np.shape(matrix_element_nucl)=}")
    print(
        "matrix element = ",
        err_brackets(
            np.average(matrix_element_nucl_neg), np.std(matrix_element_nucl_neg)
        ),
    )

    fitparam = {
        "x": np.arange(64)[fitrange],
        "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
        "label": "slope ="
        + err_brackets(
            np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
        ),
    }

    plot_correlator(
        ratio_pert_nucl,
        "pert_ratio_nucl_neg_TBC16",
        plotdir,
        fitparam=fitparam,
        ylim=(-0.5, 200),
        ylabel=r"$G_{N}^{\lambda}(\mathbf{p}')/G_{N}(\mathbf{p}')$",
    )

    ### ----------------------------------------------------------------------
    ### The ratio of the FH neg over nucl. mom avg
    print("\n-->Ratio of FH neg mom over nucl momentum avgd")
    ratio_pert_nucl = G2_pert_q100_neg[:, 0, :, 0] * (momaverage) ** (-1)

    # fitrange = slice(10, 17)
    # fitrange = slice(13, 21)
    fitrange = slice(14, 21)
    popt, paramboots = correlator_fitting(
        ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    )
    print(
        "slope =",
        err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    )
    matrix_element_nucl = paramboots[:, 1] * ratio_z_factors
    # print(f"{np.shape(paramboots[:,1])=}")
    # print(f"{np.shape(ratio_z_factors)=}")
    # print(f"{np.shape(matrix_element_nucl)=}")
    print(
        "matrix element = ",
        err_brackets(np.average(matrix_element_nucl), np.std(matrix_element_nucl)),
    )

    fitparam = {
        "x": np.arange(64)[fitrange],
        "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
        "label": "slope ="
        + err_brackets(
            np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
        ),
    }

    plot_correlator(
        ratio_pert_nucl,
        "pert_ratio_nucl_neg_momavg_TBC16",
        plotdir,
        fitparam=fitparam,
        ylim=(-0.5, 200),
        ylabel=r"$2G_{N}^{3}(\mathbf{p}')/(G_{N}(+\mathbf{p}')+G_{N}(-\mathbf{p}'))$",
    )

    ### ----------------------------------------------------------------------
    ### The ratio of the FH neg over the sigma
    print("\n-->Ratio of FH nucl with negative momentum over the Sigma")
    ratio_pert_nucl = G2_pert_q100_neg[:, 0, :, 0] * (
        # G2_unpert_q000_sigma[:, 0, :, 0]
        G2_unpert_q000_sigma_tavg[:, :, 0]
    ) ** (-1)

    # fitrange = slice(10, 19)
    fitrange = slice(16, 25)
    popt, paramboots = correlator_fitting(
        ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    )
    print(
        "slope =",
        err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    )
    matrix_element_sigma = paramboots[:, 1] / (ratio_z_factors * 0.5 * (1 + m_N / m_S))
    print(
        "matrix element neg = ",
        err_brackets(np.average(matrix_element_sigma), np.std(matrix_element_sigma)),
    )

    fitparam = {
        "x": np.arange(64)[fitrange],
        "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
        "label": "slope ="
        + err_brackets(
            np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
        ),
    }

    plot_correlator(
        ratio_pert_nucl,
        "pert_ratio_nucl_neg_sigma_TBC16",
        plotdir,
        fitparam=fitparam,
        ylim=(-0.5, 30),
        ylabel=r"$G_{N}^{3}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    )

    ### ----------------------------------------------------------------------
    # ### The ratio of the FH positive momentum over the sigma
    # print("\nRatio of FH nucl with positive momentum over the Sigma")
    # ratio_pert_nucl = G2_pert_q100_pos[:, 0, :, 0] * (
    #     G2_unpert_q000_sigma_tavg[:, :, 0]
    # ) ** (-1)

    # fitrange = slice(9, 19)
    # popt, paramboots = correlator_fitting(
    #     ratio_pert_nucl, fitrange, ff.linear, p0=[1, 0.1]
    # )
    # print(
    #     "slope =",
    #     err_brackets(np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]),
    # )
    # matrix_element = paramboots[:, 1] / (ratio_z_factors * 0.5 * (1 + m_N / m_S))
    # print(
    #     "matrix element neg = ",
    #     err_brackets(np.average(matrix_element), np.std(matrix_element)),
    # )

    # fitparam = {
    #     "x": np.arange(64)[fitrange],
    #     "y": [ff.linear(np.arange(64)[fitrange], *i) for i in paramboots],
    #     "label": "slope ="
    #     + err_brackets(
    #         np.average(paramboots, axis=0)[1], np.std(paramboots, axis=0)[1]
    #     ),
    # }

    # plot_correlator(
    #     ratio_pert_nucl,
    #     "pert_ratio_nucl_pos_sigma_TBC16",
    #     plotdir,
    #     fitparam=fitparam,
    #     ylim=(-0.5, 10),
    #     ylabel=r"$G_{N}^{\lambda}(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$",
    # )

    ### ----------------------------------------------------------------------
    ### The ratio of the FH negative momentum over the sum of the nucleon and sigma two-point functions
    print(
        "\n-->Ratio of FH nucl over the sum of the nucleon and sigma two-point functions"
    )

    ratio = []
    timelength = len(G2_pert_q100_pos[0, 0, :, 0])
    for ti in np.arange(timelength):
        ratio.append(
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

    # ratio = np.array(ratio)
    ratio = np.moveaxis(ratio, 0, 1)

    func_const = ff.initffncs("Constant")
    func_const.initpar = [2e-39]
    fitparamlist = []
    timeranges = []
    chisqlist = []
    for t in np.arange(15, 19):
        timerange = np.arange(t, 25)
        fitparam_unpert = stats.fitratio(
            func_const.eval,
            func_const.initpar,
            timerange,
            ratio[:, timerange],
            bounds=None,
            time=False,
        )
        fitparamlist.append(fitparam_unpert["param"])
        timeranges.append(timerange)
        chisqlist.append(fitparam_unpert["redchisq"])
        print(f"{fitparam_unpert['redchisq']=}")
        print(f"{fitparam_unpert['paramavg']=}")
    print("loop done")
    choice = 2
    fitparam = {
        "x": timeranges[choice],
        "y": [ff.constant(timeranges[choice], i) for i in fitparamlist[choice]],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitparamlist[choice], axis=0),
                np.std(fitparamlist[choice], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {chisqlist[choice]:0.2}$",
    }
    print("fitparam done")
    plot_correlator(
        ratio,
        "ratio_TBC16_summed",
        plotdir,
        ylim=(-1e-38, 0.5e-38),
        xlim=30,
        fitparam=fitparam,
        ylabel=r"$G_{N}^3(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{N}(\tau;\mathbf{p}') G_{\Sigma}(t-\tau;\mathbf{0})\right)$",
    )
    print("plot done")
    matr_summed = (
        fitparamlist[choice][:, 0] * z_factor_nucl_best * z_factor_sigma_best * 2
    )
    print(
        "\nmatr. summed =",
        err_brackets(np.average(matr_summed), np.std(matr_summed)),
    )

    ### ------------------------------------------------------------
    fitrange = slice(13, 23)
    popt, paramboots = correlator_fitting(ratio, fitrange, ff.constant, p0=[0.1])
    print(
        "constant =",
        err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0)),
    )

    matrix_element_summed = (
        paramboots[:, 0] * z_factor_nucl_best * z_factor_sigma_best * 2  # * m_S / m_N
    )
    print(
        "matrix element summed = ",
        err_brackets(np.average(matrix_element_summed), np.std(matrix_element_summed)),
    )

    fitparam = {
        "x": np.arange(64)[fitrange],
        "y": [ff.constant(np.arange(64)[fitrange], i) for i in paramboots],
        "label": "plateau ="
        + err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0)),
    }

    plot_correlator(
        ratio,
        "pert_ratio_nucl_sigma_summed_TBC16",
        plotdir,
        fitparam=fitparam,
        ylim=(-1e-38, 0.5e-38),
        # ylim=None,
        ylabel=r"$G_{N}^{3}(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{N}(\tau;\mathbf{p}') G_{\Sigma}(t-\tau;\mathbf{0})\right)$",
    )

    ### ----------------------------------------------------------------------
    ratio = []
    timelength = len(G2_pert_q100_pos[0, 0, :, 0])
    for ti in np.arange(timelength):
        ratio.append(
            G2_pert_q100_neg[:, 0, ti, 0]
            / np.sum(
                [
                    # G2_unpert_qm100_nucl[:, 0, tau, 0]
                    # * G2_unpert_qm100_nucl[:, 0, ti - tau, 0]
                    G2_unpert_q000_sigma[:, 0, tau, 0]
                    * G2_unpert_q000_sigma[:, 0, ti - tau, 0]
                    for tau in np.arange(ti)
                ],
                axis=0,
            )
        )

    # ratio = np.array(ratio)
    ratio = np.moveaxis(ratio, 0, 1)

    func_const = ff.initffncs("Constant")
    func_const.initpar = [2e-39]
    fitparamlist = []
    timeranges = []
    chisqlist = []
    for t in np.arange(15, 19):
        timerange = np.arange(t, 27)
        fitparam_unpert = stats.fitratio(
            func_const.eval,
            func_const.initpar,
            timerange,
            ratio[:, timerange],
            bounds=None,
            time=False,
        )
        fitparamlist.append(fitparam_unpert["param"])
        timeranges.append(timerange)
        chisqlist.append(fitparam_unpert["redchisq"])
        # print(f"{fitparam_unpert['redchisq']=}")
        # print(f"{fitparam_unpert['paramavg']=}")
    print("loop done")
    choice = 2
    fitparam = {
        "x": timeranges[choice],
        "y": [ff.constant(timeranges[choice], i) for i in fitparamlist[choice]],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitparamlist[choice], axis=0),
                np.std(fitparamlist[choice], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {chisqlist[choice]:0.2}$",
    }
    print("fitparam done")
    plot_correlator(
        ratio,
        "ratio_TBC16_summed_sigma",
        plotdir,
        ylim=(-1e-39, 1e-39),
        xlim=30,
        fitparam=fitparam,
        ylabel=r"$G_{N}^{3}(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{\Sigma}(\tau;\mathbf{p}') G_{\Sigma}(t-\tau;\mathbf{0})\right)$",
    )
    fitrange = slice(13, 23)
    popt, paramboots = correlator_fitting(ratio, fitrange, ff.constant, p0=[0.1])
    print(
        "constant =",
        err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0)),
    )

    matrix_element_summed_sigma = (
        paramboots[:, 0]
        * z_factor_sigma_best ** 3
        * 4
        / z_factor_nucl_best  # * m_S / m_N
        / (1 + m_N / m_S)
    )
    print(
        "matrix element summed = ",
        err_brackets(
            np.average(matrix_element_summed_sigma), np.std(matrix_element_summed_sigma)
        ),
    )

    ### ----------------------------------------------------------------------
    ratio = []
    timelength = len(G2_pert_q100_pos[0, 0, :, 0])
    for ti in np.arange(timelength):
        ratio.append(
            G2_pert_q100_neg[:, 0, ti, 0]
            / np.sum(
                [
                    G2_unpert_qm100_nucl[:, 0, tau, 0]
                    * G2_unpert_qm100_nucl[:, 0, ti - tau, 0]
                    # G2_unpert_q000_sigma[:, 0, tau, 0]
                    # * G2_unpert_q000_sigma[:, 0, ti - tau, 0]
                    for tau in np.arange(ti)
                ],
                axis=0,
            )
        )

    # ratio = np.array(ratio)
    ratio = np.moveaxis(ratio, 0, 1)

    func_const = ff.initffncs("Constant")
    func_const.initpar = [2e-39]
    fitparamlist = []
    timeranges = []
    chisqlist = []
    for t in np.arange(15, 19):
        timerange = np.arange(t, 22)
        fitparam_unpert = stats.fitratio(
            func_const.eval,
            func_const.initpar,
            timerange,
            ratio[:, timerange],
            bounds=None,
            time=False,
        )
        fitparamlist.append(fitparam_unpert["param"])
        timeranges.append(timerange)
        chisqlist.append(fitparam_unpert["redchisq"])
        # print(f"{fitparam_unpert['redchisq']=}")
        # print(f"{fitparam_unpert['paramavg']=}")
    print("loop done")
    choice = 2
    fitparam = {
        "x": timeranges[choice],
        "y": [ff.constant(timeranges[choice], i) for i in fitparamlist[choice]],
        "label": "plateau = $ "
        + str(
            err_brackets(
                np.average(fitparamlist[choice], axis=0),
                np.std(fitparamlist[choice], axis=0),
            )
        )
        + f"$\n $\chi^2_{{\\textrm{{dof}}}} = {chisqlist[choice]:0.2}$",
    }
    print("fitparam done")
    plot_correlator(
        ratio,
        "ratio_TBC16_summed_nucl",
        plotdir,
        ylim=(-1e-37, 1e-39),
        xlim=30,
        fitparam=fitparam,
        ylabel=r"$G_{N}^{3}(t;\mathbf{p}')/ \left(\sum_{\tau=0}^{t} G_{n}(\tau;\mathbf{p}') G_{n}(t-\tau;\mathbf{0})\right)$",
    )
    fitrange = slice(13, 23)
    popt, paramboots = correlator_fitting(ratio, fitrange, ff.constant, p0=[0.1])
    print(
        "constant =",
        err_brackets(np.average(paramboots, axis=0), np.std(paramboots, axis=0)),
    )

    matrix_element_summed_nucl = (
        paramboots[:, 0]
        * z_factor_nucl_best ** 3
        * (1 + m_N / m_S)
        / z_factor_sigma_best  # * m_S / m_N
    )
    print(
        "matrix element summed = ",
        err_brackets(
            np.average(matrix_element_summed_nucl), np.std(matrix_element_summed_nucl)
        ),
    )

    ### ----------------------------------------------------------------------
    ### Comparison plot of the results from the different ratios
    pypl.figure(figsize=(9, 6))
    pypl.errorbar(
        ["nucl", "nucl negative", "sigma", "summed", "summed sigma", "summed nucl"],
        np.array(
            [
                np.average(matrix_element_nucl),
                np.average(matrix_element_nucl_neg),
                np.average(matrix_element_sigma),
                -np.average(matrix_element_summed),
                -np.average(matrix_element_summed_sigma),
                -np.average(matrix_element_summed_nucl),
            ]
        ),
        np.array(
            [
                np.std(matrix_element_nucl),
                np.std(matrix_element_nucl_neg),
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

    mat_elements = np.array(
        [
            matrix_element_nucl,
            matrix_element_sigma,
            matrix_element_summed,
        ]
    )

    with open("mat_elements.pkl", "wb") as file_out:
        pickle.dump(mat_elements, file_out)

        # with open("mat_elements.csv", "w") as csvfile:
        #     datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
        #     datawrite.writerow(headernames)
