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

# pars1 = 0


def fitfunction2(lmb, E_nucl_p, E_sigma_p, matrix_element):
    deltaE = 0.5 * (E_nucl_p + E_sigma_p) - 0.5 * np.sqrt(
        (E_nucl_p - E_sigma_p) ** 2 + 4 * lmb ** 2 * matrix_element ** 2
    )
    return deltaE


def fitfunction3(lmb, pars0, pars2):
    deltaE = 0.5 * (pars0 + pars1) + 0.5 * np.sqrt(
        (pars0 - pars1) ** 2 + 4 * lmb ** 2 * pars2 ** 2
    )
    return deltaE

def fitfunction4(lmb, E_nucl_p, E_sigma_p, matrix_element):
    deltaE = 0.5 * np.sqrt(
        (E_nucl_p - E_sigma_p) ** 2 + 4 * lmb ** 2 * matrix_element ** 2
    )
    return deltaE

def fitfunction5(lmb, Delta_E, matrix_element):
    deltaE = 0.5 * np.sqrt(
        Delta_E ** 2 + 4 * lmb ** 2 * matrix_element ** 2
    )
    return deltaE


def fit_lmb(ydata, function, lambdas, plotdir, p0=None):
    """Fit the lambda dependence

    data is a correlator with tht bootstraps on the first index and the time on the second
    lambdas is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """

    # bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    bounds = ([0, 0], [np.inf, np.inf])
    ydata = ydata.T
    # print(np.shape(ydata))
    data_set = ydata
    ydata_avg = np.average(data_set, axis=0)
    # print("ydata_avg", ydata_avg)
    # print("lambdas", lambdas)
    covmat = np.cov(data_set.T)
    diag = np.diagonal(covmat)
    norms = np.einsum('i,j->ij',diag,diag)**0.5
    covmat_norm = covmat/ norms
    pypl.figure(figsize=(11,11))
    mat = pypl.matshow(np.linalg.inv(covmat))
    pypl.colorbar(mat, shrink=0.5)
    pypl.savefig(plotdir / ("cov_matrix.pdf"))
    # print(covmat)

    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    popt_avg, pcov_avg = curve_fit(
        function,
        lambdas,
        ydata_avg,
        sigma=diag_sigma,
        p0=p0,
        maxfev=4000,
        bounds=bounds,
    )
    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, np.linalg.pinv(covmat))
    print("popt_avg", popt_avg)
    p0 = popt_avg
    redchisq = chisq / len(lambdas)
    bootfit = []
    for iboot, values in enumerate(ydata):
        # print(iboot)
        popt, pcov = curve_fit(
            function, lambdas, values, sigma=diag_sigma, 
            # maxfev=4000,
            p0 = p0,
            bounds=bounds
        )  # , p0=popt_avg)
        # print(popt)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    print("bootfit", np.average(bootfit, axis=0))
    return bootfit, redchisq


def plot_lmb_dep(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift"""
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1),
        np.std(all_data["order0_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas1"] + 0.0001,
        np.average(all_data["order1_fit"], axis=1),
        np.std(all_data["order1_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas2"] + 0.0002,
        np.average(all_data["order2_fit"], axis=1),
        np.std(all_data["order2_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas3"] + 0.0003,
        np.average(all_data["order3_fit"], axis=1),
        np.std(all_data["order3_fit"], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    # pypl.ylim(-0.003, 0.035)
    # pypl.xlim(-0.01, 0.22)
    pypl.xlim(-0.01, all_data["lambdas3"][-1] * 1.1)
    pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)

    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        fitBS0 = np.array([fitfunction5(all_data["lambdas0"][lmb_range], *bf) for bf in fit_data["bootfit0"]])
        # print(np.std(fitBS0, axis=0))
        # print(
        #     np.average(fit_data["bootfit0"], axis=0)[1],
        #     np.std(fit_data["bootfit0"], axis=0)[1],
        # )
        m_e_0 = err_brackets(
            np.average(fit_data["bootfit0"], axis=0)[1],
            np.std(fit_data["bootfit0"], axis=0)[1],
        )
        m_e_1 = err_brackets(
            np.average(fit_data["bootfit1"], axis=0)[1],
            np.std(fit_data["bootfit1"], axis=0)[1],
        )
        m_e_2 = err_brackets(
            np.average(fit_data["bootfit2"], axis=0)[1],
            np.std(fit_data["bootfit2"], axis=0)[1],
        )
        print(np.std(fit_data["bootfit3"], axis=0))
        print(np.std(fit_data["bootfit3"], axis=0)[1])
        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[1],
            np.std(fit_data["bootfit3"], axis=0)[1],
        )
        # print(m_e_0)

        pypl.fill_between(
            all_data["lambdas0"][lmb_range],
            np.average(fitBS0, axis=0) - np.std(fitBS0, axis=0),
            np.average(fitBS0, axis=0) + np.std(fitBS0, axis=0),
            alpha=0.3,
            color=_colors[0],
            label=rf"$\textrm{{M.E.}}={m_e_0}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_0}$",
        )
        fitBS1 = np.array(
            [fitfunction5(all_data["lambdas1"][lmb_range], *bf) for bf in fit_data["bootfit1"]]
            # [fitfunction5(lambdas1[:fitlim], *bf) for bf in fit_data["bootfit1"]]
        )
        print(np.std(fitBS1, axis=0))

        pypl.fill_between(
            all_data["lambdas1"][lmb_range], #[:fitlim],
            np.average(fitBS1, axis=0) - np.std(fitBS1, axis=0),
            np.average(fitBS1, axis=0) + np.std(fitBS1, axis=0),
            alpha=0.3,
            color=_colors[1],
            label=rf"$\textrm{{M.E.}}={m_e_1}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_1}$",
        )
        fitBS2 = np.array(
            [fitfunction5(all_data["lambdas2"][lmb_range], *bf) for bf in fit_data["bootfit2"]]
        )
        print(np.std(fitBS2, axis=0))
        pypl.fill_between(
            all_data["lambdas2"][lmb_range],
            np.average(fitBS2, axis=0) - np.std(fitBS2, axis=0),
            np.average(fitBS2, axis=0) + np.std(fitBS2, axis=0),
            alpha=0.3,
            color=_colors[2],
            label=rf"$\textrm{{M.E.}}={m_e_2}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_2}$",
        )
        fitBS3 = np.array([fitfunction5(all_data["lambdas3"][lmb_range], *bf) for bf in fit_data["bootfit3"]])
        print(np.std(fitBS3, axis=0))
        pypl.fill_between(
            all_data["lambdas3"][lmb_range],
            np.average(fitBS3, axis=0) - np.std(fitBS3, axis=0),
            np.average(fitBS3, axis=0) + np.std(fitBS3, axis=0),
            alpha=0.3,
            color=_colors[3],
            label= rf"$\textrm{{M.E.}}={m_e_3}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_3}$",
        )

        pypl.legend(fontsize="xx-small")
        # pypl.xlim(-0.01, 0.16)
        # pypl.ylim(0, 0.15)
        # pypl.xlim(-0.001, 0.045)
        # pypl.ylim(-0.003, 0.035)
        pypl.xlim(-0.01, all_data["lambdas3"][-1] * 1.1)
        pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)
        pypl.savefig(plotdir / ("lambda_dep_fit.pdf"))

        # pypl.xlim(-0.005, 0.08)
        # pypl.ylim(0.015, 0.065)
        pypl.xlim(-0.0001, 0.025)
        pypl.ylim(-0.0002, 0.015)
        pypl.savefig(plotdir / ("lambda_dep_zoom.pdf"))

    pypl.close()
    return

def plot_lmb_depR(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift"""
    pypl.figure(figsize=(6, 6))

    pypl.fill_between(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1) - np.std(all_data["order0_fit"], axis=1),
        np.average(all_data["order0_fit"], axis=1) + np.std(all_data["order0_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        all_data["lambdas1"],
        np.average(all_data["order1_fit"], axis=1) - np.std(all_data["order1_fit"], axis=1),
        np.average(all_data["order1_fit"], axis=1) + np.std(all_data["order1_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        all_data["lambdas2"],
        np.average(all_data["order2_fit"], axis=1) - np.std(all_data["order2_fit"], axis=1),
        np.average(all_data["order2_fit"], axis=1) + np.std(all_data["order2_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=0,
        alpha=0.3
    )
    pypl.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1) - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1) + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3
    )

    # pypl.errorbar(
    #     all_data["lambdas0"],
    #     np.average(all_data["order0_fit"], axis=1),
    #     np.std(all_data["order0_fit"], axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^1)$",
    #     color=_colors[0],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     all_data["lambdas1"] + 0.0001,
    #     np.average(all_data["order1_fit"], axis=1),
    #     np.std(all_data["order1_fit"], axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^2)$",
    #     color=_colors[1],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     all_data["lambdas2"] + 0.0002,
    #     np.average(all_data["order2_fit"], axis=1),
    #     np.std(all_data["order2_fit"], axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^3)$",
    #     color=_colors[2],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    # pypl.errorbar(
    #     all_data["lambdas3"] + 0.0003,
    #     np.average(all_data["order3_fit"], axis=1),
    #     np.std(all_data["order3_fit"], axis=1),
    #     fmt="s",
    #     label=r"$\mathcal{O}(\lambda^4)$",
    #     color=_colors[3],
    #     capsize=4,
    #     elinewidth=1,
    #     markerfacecolor="none",
    # )
    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    # pypl.ylim(-0.003, 0.035)
    # pypl.xlim(-0.01, 0.22)
    pypl.xlim(-0.01, all_data["lambdas3"][-1] * 1.1)
    pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)

    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        fitBS0 = np.array([fitfunction5(lambdas0[lmb_range], *bf) for bf in fit_data["bootfit0"]])
        # print(np.std(fitBS0, axis=0))
        # print(
        #     np.average(fit_data["bootfit0"], axis=0)[1],
        #     np.std(fit_data["bootfit0"], axis=0)[1],
        # )
        m_e_0 = err_brackets(
            np.average(fit_data["bootfit0"], axis=0)[1],
            np.std(fit_data["bootfit0"], axis=0)[1],
        )
        m_e_1 = err_brackets(
            np.average(fit_data["bootfit1"], axis=0)[1],
            np.std(fit_data["bootfit1"], axis=0)[1],
        )
        m_e_2 = err_brackets(
            np.average(fit_data["bootfit2"], axis=0)[1],
            np.std(fit_data["bootfit2"], axis=0)[1],
        )
        print(np.std(fit_data["bootfit3"], axis=0))
        print(np.std(fit_data["bootfit3"], axis=0)[1])
        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[1],
            np.std(fit_data["bootfit3"], axis=0)[1],
        )
        # print(m_e_0)

        pypl.fill_between(
            lambdas0[lmb_range],
            np.average(fitBS0, axis=0) - np.std(fitBS0, axis=0),
            np.average(fitBS0, axis=0) + np.std(fitBS0, axis=0),
            alpha=0.3,
            color=_colors[0],
            label=rf"$\textrm{{M.E.}}={m_e_0}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_0}$",
        )
        fitBS1 = np.array(
            [fitfunction5(lambdas1[lmb_range], *bf) for bf in fit_data["bootfit1"]]
            # [fitfunction5(lambdas1[:fitlim], *bf) for bf in fit_data["bootfit1"]]
        )
        print(np.std(fitBS1, axis=0))

        pypl.fill_between(
            lambdas1[lmb_range], #[:fitlim],
            np.average(fitBS1, axis=0) - np.std(fitBS1, axis=0),
            np.average(fitBS1, axis=0) + np.std(fitBS1, axis=0),
            alpha=0.3,
            color=_colors[1],
            label=rf"$\textrm{{M.E.}}={m_e_1}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_1}$",
        )
        fitBS2 = np.array(
            [fitfunction5(lambdas2[lmb_range], *bf) for bf in fit_data["bootfit2"]]
        )
        print(np.std(fitBS2, axis=0))
        pypl.fill_between(
            lambdas2[lmb_range],
            np.average(fitBS2, axis=0) - np.std(fitBS2, axis=0),
            np.average(fitBS2, axis=0) + np.std(fitBS2, axis=0),
            alpha=0.3,
            color=_colors[2],
            label=rf"$\textrm{{M.E.}}={m_e_2}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_2}$",
        )
        fitBS3 = np.array([fitfunction5(lambdas3[lmb_range], *bf) for bf in fit_data["bootfit3"]])
        print(np.std(fitBS3, axis=0))
        pypl.fill_between(
            lambdas3[lmb_range],
            np.average(fitBS3, axis=0) - np.std(fitBS3, axis=0),
            np.average(fitBS3, axis=0) + np.std(fitBS3, axis=0),
            alpha=0.3,
            color=_colors[3],
            label= rf"$\textrm{{M.E.}}={m_e_3}$",
            # rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:0.2}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_3}$",
        )

        pypl.legend(fontsize="xx-small")
        # pypl.xlim(-0.01, 0.16)
        # pypl.ylim(0, 0.15)
        # pypl.xlim(-0.001, 0.045)
        # pypl.ylim(-0.003, 0.035)
        pypl.xlim(-0.01, lambdas3[-1] * 1.1)
        pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)
        pypl.savefig(plotdir / ("lambda_dep_fit.pdf"))

        # pypl.xlim(-0.005, 0.08)
        # pypl.ylim(0.015, 0.065)
        pypl.xlim(-0.0001, 0.025)
        pypl.ylim(-0.0002, 0.015)
        pypl.savefig(plotdir / ("lambda_dep_zoom.pdf"))

    pypl.close()
    return

def fit_const(xdata, data_set, lmb_range):
    bounds = ([0], [np.inf])
    ydata_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
    # print('diag sigma',np.shape(diag_sigma), diag_sigma)
    function = ff.constant
    p0 = 0.7
    popt_avg, pcov_avg = curve_fit(
        function,
        xdata,
        ydata_avg,
        sigma=diag_sigma,
        # sigma=np.linalg.inv(covmat),
        p0=p0,
        # maxfev=4000,
        # bounds=bounds,
    )
    # chisq = ff.chisqfn2(popt_avg, function, xdata, ydata_avg, np.linalg.inv(covmat))
    chisq = ff.chisqfn2(popt_avg, function, xdata, ydata_avg, np.linalg.inv(covmat))
    print("\n chisq",chisq, popt_avg)

    fit_param = np.zeros(len(data_set))
    for iboot, values in enumerate(data_set):
        popt, pcov_avg = curve_fit(
            function,
            xdata,
            values,
            sigma=diag_sigma,
            # sigma=np.linalg.inv(covmat),
            p0=p0,
            # maxfev=4000,
            # bounds=bounds,
        )
        fit_param[iboot] = popt
    return fit_param, chisq

def plot_lmb_dep2(all_data, plotdir, lmb_range=None):
    """Make a plot of the lambda dependence of the energy shift"""

    xdata0 = all_data["lambdas0"][lmb_range]
    data_set0 = np.einsum('ij,j->ij',all_data["order0_fit"][lmb_range].T, xdata0**(-1))
    fit_param0, chisq0 = fit_const(xdata0, data_set0, lmb_range)
    xdata1 = all_data["lambdas1"][lmb_range]
    data_set1 = np.einsum('ij,j->ij',all_data["order1_fit"][lmb_range].T, xdata1**(-1))
    fit_param1, chisq1 = fit_const(xdata1, data_set1, lmb_range)
    xdata2 = all_data["lambdas2"][lmb_range]
    data_set2 = np.einsum('ij,j->ij',all_data["order2_fit"][lmb_range].T, xdata2**(-1))
    fit_param2, chisq2 = fit_const(xdata2, data_set2, lmb_range)
    xdata3 = all_data["lambdas3"][lmb_range]
    data_set3 = np.einsum('ij,j->ij',all_data["order3_fit"][lmb_range].T, xdata3**(-1))
    fit_param3, chisq3 = fit_const(xdata3, data_set3, lmb_range)

    pypl.figure(figsize=(6, 6))

    pypl.errorbar(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1)/all_data["lambdas0"],
        np.std(all_data["order0_fit"], axis=1)/all_data["lambdas0"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas1"] + 0.0001,
        np.average(all_data["order1_fit"], axis=1)/all_data["lambdas1"],
        np.std(all_data["order1_fit"], axis=1)/all_data["lambdas1"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas2"] + 0.0002,
        np.average(all_data["order2_fit"], axis=1)/all_data["lambdas2"],
        np.std(all_data["order2_fit"], axis=1)/all_data["lambdas2"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        all_data["lambdas3"] + 0.0003,
        np.average(all_data["order3_fit"], axis=1)/all_data["lambdas3"],
        np.std(all_data["order3_fit"], axis=1)/all_data["lambdas3"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    # label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:0.2}$"
    pypl.plot(xdata0, [np.average(fit_param0)]*len(xdata0), color=_colors[0], label = rf"$\chi^2_{{\textrm{{dof}}}} = {chisq0:0.2}$")
    pypl.plot(xdata1, [np.average(fit_param1)]*len(xdata1), color=_colors[1], label = rf"$\chi^2_{{\textrm{{dof}}}} = {chisq1:0.2}$")
    pypl.plot(xdata2, [np.average(fit_param2)]*len(xdata2), color=_colors[2], label = rf"$\chi^2_{{\textrm{{dof}}}} = {chisq2:0.2}$")
    pypl.plot(xdata3, [np.average(fit_param3)]*len(xdata3), color=_colors[3], label = rf"$\chi^2_{{\textrm{{dof}}}} = {chisq3:0.2}$")
    pypl.fill_between(
        xdata0,
        np.average(fit_param0) - np.std(fit_param0),
        np.average(fit_param0) + np.std(fit_param0),
        alpha=0.3,
        color=_colors[0],
    )
    pypl.fill_between(
        xdata1,
        np.average(fit_param1) - np.std(fit_param1),
        np.average(fit_param1) + np.std(fit_param1),
        alpha=0.3,
        color=_colors[1],
    )
    pypl.fill_between(
        xdata2,
        np.average(fit_param2) - np.std(fit_param2),
        np.average(fit_param2) + np.std(fit_param2),
        alpha=0.3,
        color=_colors[2],
    )
    pypl.fill_between(
        xdata3,
        np.average(fit_param3) - np.std(fit_param3),
        np.average(fit_param3) + np.std(fit_param3),
        alpha=0.3,
        color=_colors[3],
    )

    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    # pypl.ylim(-0.003, 0.035)
    # pypl.xlim(-0.01, 0.22)
    # pypl.xlim(-0.01, lambdas3[-1] * 1.1)
    # pypl.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.3)

    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E / \lambda$")
    pypl.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("lambda_dep_divlmb.pdf"))

    pypl.close()
    return



def main():
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
        config_file = "data_dir_theta2.yaml" #default file
    with open(config_file) as f:
        config = yaml.safe_load(f)
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    print("datadir: ", datadir / ("lambda_dep.pkl"))

    t_range = np.arange(config["t_range0"], config["t_range1"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    lmb_val = config["lmb_val"]

    # Read data from the pickle file
    with open(
        datadir
        / (f"lambda_dep_t{time_choice}_dt{delta_t}_fit{t_range[0]}-{t_range[-1]}.pkl"),
        "rb",
    ) as file_in:
        data = pickle.load(file_in)
    lambdas = data["lambdas"]
    order0_fit = data["order0_fit"]
    order1_fit = data["order1_fit"]
    order2_fit = data["order2_fit"]
    order3_fit = data["order3_fit"]
    redchisq = data["redchisq"]
    time_choice = data["time_choice"]
    delta_t = data["delta_t"]

    # Filter out data points with a high reduced chi-squared value
    chisq_tol = 1.5 #1.7
    order0_fit = order0_fit[np.where(redchisq[0] <= chisq_tol)]
    lambdas0 = lambdas[np.where(redchisq[0] <= chisq_tol)]
    order1_fit = order1_fit[np.where(redchisq[1] <= chisq_tol)]
    lambdas1 = lambdas[np.where(redchisq[1] <= chisq_tol)]
    order2_fit = order2_fit[np.where(redchisq[2] <= chisq_tol)]
    lambdas2 = lambdas[np.where(redchisq[2] <= chisq_tol)]
    order3_fit = order3_fit[np.where(redchisq[3] <= chisq_tol)]
    lambdas3 = lambdas[np.where(redchisq[3] <= chisq_tol)]

    all_data = {
        "lambdas0": lambdas0,
        "lambdas1": lambdas1,
        "lambdas2": lambdas2,
        "lambdas3": lambdas3,
        "order0_fit": order0_fit,
        "order1_fit": order1_fit,
        "order2_fit": order2_fit,
        "order3_fit": order3_fit,
        "redchisq": redchisq,
        "time_choice": time_choice,
        "delta_t": delta_t,
    }

    # scaled_z0 = (redchisq[0] - redchisq[0].min()) / redchisq[0].ptp()
    # colors_0 = [[0., 0., 0., i] for i in scaled_z0]

    plot_lmb_depR(all_data, plotdir)
    # plot_lmb_dep2(all_data, plotdir)

    # Fit the quadratic behaviour in lambda
    # p0 = (0.01, 0.01, 0.7)
    # p0 = (1, 1, 0.7)
    p0 = (1e-3, 0.7)
    fitlim = 30
    lmb_range=np.arange(4,14)
    # lmb_range=np.arange(5,10)
    # lmb_range=np.arange(6,9)
    plot_lmb_dep2(all_data, plotdir, lmb_range)

    # Fit to the lambda dependence at each order in lambda
    print("\n")
    try:
        bootfit0, redchisq0 = fit_lmb(order0_fit[lmb_range], fitfunction5, lambdas0[lmb_range], plotdir, p0=p0)
        print("redchisq order 1", redchisq0, "\n")
        print("fit order 1", np.average(bootfit0, axis=0), "\n")
        p0 = np.average(bootfit0, axis=0)
        # print(p0)

        bootfit1, redchisq1 = fit_lmb(
            order1_fit[lmb_range], fitfunction5, lambdas1[lmb_range], plotdir, p0=p0
        )
        print("redchisq order 2", redchisq1, "\n")
        print("fit order 2", np.average(bootfit1, axis=0), "\n")

        bootfit2, redchisq2 = fit_lmb(
            order2_fit[lmb_range], fitfunction5, lambdas2[lmb_range], plotdir, p0=p0
        )
        print("redchisq order 3", redchisq2, "\n")
        print("fit order 3", np.average(bootfit2, axis=0), "\n")

        bootfit3, redchisq3 = fit_lmb(
            order3_fit[lmb_range], fitfunction5, lambdas3[lmb_range], plotdir, p0=p0
        )
        print("redchisq order 4", redchisq3, "\n")
        print("fit order 4", np.average(bootfit3, axis=0), "\n")
        print("fit std order 4", np.std(bootfit3, axis=0), "\n")

        fit_data = {
            "lmb_range": lmb_range,
            "fitlim": fitlim,
            "bootfit0": bootfit0,
            "bootfit1": bootfit1,
            "bootfit2": bootfit2,
            "bootfit3": bootfit3,
            "redchisq0": redchisq0,
            "redchisq1": redchisq1,
            "redchisq2": redchisq2,
            "redchisq3": redchisq3,
        }
        with open(datadir / (f"matrix_element.pkl"), "wb") as file_out:
            pickle.dump(fit_data, file_out)
        
    except RuntimeError as e:
        print("====================\nFitting Failed\n", e, "\n====================")
        fit_data = None

    plot_lmb_dep(all_data, plotdir, fit_data)

    ### ----------------------------------------------------------------------
    # lmb_val = 0.06 #0.16
    time_choice_range = np.arange(5, 10)
    delta_t_range = np.arange(1, 4)
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
    delta_t_choice = np.where(delta_t_range == config["delta_t"])[0][0]

    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        time_choice_range,
        np.average(order0_fit[:, delta_t_choice, :], axis=1),
        np.std(order0_fit[:, delta_t_choice, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range + 0.03,
        np.average(order1_fit[:, delta_t_choice, :], axis=1),
        np.std(order1_fit[:, delta_t_choice, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range + 0.06,
        np.average(order2_fit[:, delta_t_choice, :], axis=1),
        np.std(order2_fit[:, delta_t_choice, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range + 0.09,
        np.average(order3_fit[:, delta_t_choice, :], axis=1),
        np.std(order3_fit[:, delta_t_choice, :], axis=1),
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
    # pypl.ylim(0.03, 0.055)
    pypl.xlabel("$t_{0}$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$\Delta t = {delta_t_range[delta_t_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    # pypl.show()

    # --------------------------------------------------------------------------------
    # t0_choice = 0

    t0_choice = np.where(time_choice_range == config["time_choice"])[0][0]
    # t0_choice = config["time_choice"]

    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        delta_t_range,
        np.average(order0_fit[t0_choice, :, :], axis=1),
        np.std(order0_fit[t0_choice, :, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range + 0.03,
        np.average(order1_fit[t0_choice, :, :], axis=1),
        np.std(order1_fit[t0_choice, :, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range + 0.06,
        np.average(order2_fit[t0_choice, :, :], axis=1),
        np.std(order2_fit[t0_choice, :, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        delta_t_range + 0.09,
        np.average(order3_fit[t0_choice, :, :], axis=1),
        np.std(order3_fit[t0_choice, :, :], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    pypl.legend(fontsize="x-small")
    # pypl.ylim(0, 0.2)
    # pypl.ylim(0.03, 0.055)
    pypl.xlabel("$\Delta t$")
    pypl.ylabel("$\Delta E$")
    pypl.title(rf"$t_{{0}} = {time_choice_range[t0_choice]}, \lambda = {lmb_val}$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"delta_t_dep_l{lmb_val}.pdf"))
    # pypl.show()


if __name__ == "__main__":
    main()
