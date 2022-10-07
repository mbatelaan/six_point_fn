import numpy as np
from scipy import linalg
from pathlib import Path
import pickle
import yaml
import sys
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY
from gevpanalysis.util import find_file
from gevpanalysis.util import read_config

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

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


def fitfunction5(lmb, Delta_E, matrix_element):
    deltaE = np.sqrt(Delta_E ** 2 + 4 * lmb ** 2 * matrix_element ** 2)
    return deltaE


def fitfunction6(lmb, matrix_element, delta_E_fix):
    deltaE = np.sqrt(delta_E_fix ** 2 + 4 * lmb ** 2 * matrix_element ** 2)
    return deltaE


def fitfunction_order4(lmb, Delta_E, A, B):
    """The fit function Ross proposed to capture the compton amplitude"""
    deltaE = np.sqrt(Delta_E ** 2 + 4 * lmb ** 2 * A ** 2 + lmb ** 4 * B ** 2)
    return deltaE


def fit_lmb(ydata, function, lambdas, p0=None):
    """Fit the lambda dependence

    ydata is an array with the lambda values on the first index and bootstraps on the second index
    lambdas is an array of values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    bounds = ([0, 0], [np.inf, np.inf])
    # get the bootstrp on the first index
    ydata = ydata.T
    ydata_avg = np.average(ydata, axis=0)

    covmat = np.cov(ydata.T)
    covmat_inverse = linalg.pinv(covmat)
    diag = np.diagonal(covmat)
    diag_sigma = np.diag(np.std(ydata, axis=0) ** 2)
    dof = len(lambdas) - len(p0)

    popt_avg, pcov_avg = curve_fit(
        function,
        lambdas,
        ydata_avg,
        sigma=diag_sigma,
        p0=p0,
        # maxfev=4000,
        bounds=bounds,
    )

    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, covmat_inverse)
    redchisq = chisq / dof

    # Fit each bootstrap resample
    p0 = popt_avg
    bootfit = []
    for iboot, values in enumerate(ydata):
        popt, pcov = curve_fit(
            function,
            lambdas,
            values,
            sigma=diag_sigma,
            p0=p0,
            bounds=bounds,
        )
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    return bootfit, redchisq, chisq


def fit_lambda_dep(fitlist, order, lmb_range):
    """Fit the lambda dependence of the energy shift"""
    p0 = (1e-3, 0.7)
    fit_data = np.array([fit[f"order{order}_fit"][:, 1] for fit in fitlist])
    lambdas = np.array([fit[f"lambdas"] for fit in fitlist])

    # Check if we haven't excluded some of the chosen fit range
    if lmb_range[-1] >= len(lambdas):
        lmb_range = np.arange(min(len(lambdas) - 5, lmb_range[0]), len(lambdas))
    else:
        lmb_range = lmb_range
    bootfit, redchisq_fit, chisq_fit = fit_lmb(
        fit_data[lmb_range],
        fitfunction5,
        lambdas[lmb_range],
        p0=p0,
    )
    print(f"redchisq order {order}:", redchisq_fit)
    print(f"chisq order {order}:", chisq_fit)
    print(f"fit order {order}:", np.average(bootfit, axis=0), "\n")
    return lmb_range, bootfit, redchisq_fit, chisq_fit


def fit_lambda_dep_2(fitlist, delta_E_fix, order, lmb_range, fitfunction):
    """Fit the lambda dependence of the energy shift
    Now fitting with only one parameter, we set the y-intercept by using the energy ratio gotten from fits.
    """
    p0_2 = (
        1e-4,
        0.7,
    )
    p0_1 = (0.7,)
    fit_data = np.array([fit[f"order{order}_fit"][:, 1] for fit in fitlist])
    lambdas = np.array([fit[f"lambdas"] for fit in fitlist])

    # Check if we haven't excluded some of the chosen fit range
    if lmb_range[-1] >= len(lambdas):
        lmb_range = np.arange(min(len(lambdas) - 5, lmb_range[0]), len(lambdas))
    else:
        lmb_range = lmb_range

    ydata = fit_data[lmb_range].T
    ydata_avg = np.average(ydata, axis=0)
    covmat = np.cov(ydata.T)
    invcovmat = linalg.inv(covmat)
    diag = np.diagonal(covmat)
    diag_sigma = np.diag(np.std(ydata, axis=0) ** 2)

    xdata = lambdas[lmb_range]
    # print('\n\n',xdata)
    # print('\n\n',ydata_avg)
    # print('\n\n',delta_E_fix)

    # # ============================================================
    # # two-parameter function
    # resavg = syopt.minimize(
    #     ff.chisqfn2,
    #     p0_2,
    #     args=(fitfunction5, xdata, ydata_avg, invcovmat),
    #     method="Nelder-Mead",
    #     # bounds=bounds,
    #     options={"disp": False},
    # )
    # print('\nresavg.x = ', resavg.x)
    # print('resavg.fun = ', resavg.fun)
    # bootfit = []
    # chisq_vals = []
    # for iboot, values in enumerate(ydata):
    #     resavg = syopt.minimize(
    #         ff.chisqfn2,
    #         p0_2,
    #         # args=(fitfunction5, xdata, values, diag_sigma),
    #         args=(fitfunction5, xdata, values, diag_sigma),
    #         method="Nelder-Mead",
    #         # bounds=bounds,
    #         options={"disp": False},
    #     )
    #     bootfit.append(resavg.x)
    #     chisq_vals.append(resavg.fun)
    # chisq_vals = np.array(chisq_vals)
    # bootfit = np.array(bootfit)
    # bootfit_avg = np.average(bootfit, axis=0)
    # chisq = ff.chisqfn2(bootfit_avg, fitfunction5, xdata, ydata_avg, invcovmat)
    # dof = len(xdata) - len(p0_2)
    # redchisq = chisq / dof

    # print('\nbootfit avg fn5 = ',bootfit_avg)
    # print('bootfit chisq fn5 = ',np.average(chisq_vals, axis=0))
    # print('bootfit chisq fn5 = ',redchisq, '\n\n')

    # ============================================================
    # one-parameter function
    resavg = syopt.minimize(
        ff.chisqfn4,
        p0_1,
        args=(fitfunction6, xdata, ydata_avg, (delta_E_fix,), invcovmat),
        method="Nelder-Mead",
        options={"disp": False},
    )
    # print('\nresavg.x = ', resavg.x)
    # print('resavg.fun = ', resavg.fun)

    bootfit = []
    chisq_vals = []
    for iboot, values in enumerate(ydata):
        resavg = syopt.minimize(
            ff.chisqfn4,
            p0_1,
            args=(fitfunction6, xdata, values, (delta_E_fix,), diag_sigma),
            method="Nelder-Mead",
            options={"disp": False},
        )
        bootfit.append(resavg.x)
        chisq_vals.append(resavg.fun)
    bootfit = np.array(bootfit)
    chisq_vals = np.array(chisq_vals)
    bootfit_avg = np.average(bootfit, axis=0)
    chisq = ff.chisqfn4(
        bootfit_avg, fitfunction6, xdata, ydata_avg, (delta_E_fix,), invcovmat
    )
    dof = len(xdata) - len(p0_1)
    redchisq = chisq / dof

    # print('\nbootfit avg fn6 = ',np.average(bootfit, axis=0))
    # print('bootfit chisq fn6 = ',np.average(chisq_vals, axis=0))
    # print('bootfit chisq fn6 = ',redchisq, '\n\n')
    return bootfit, redchisq


def plot_lmb_depR(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    plt.figure(figsize=(9, 6))
    plt.fill_between(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1)
        - np.std(all_data["order0_fit"], axis=1),
        np.average(all_data["order0_fit"], axis=1)
        + np.std(all_data["order0_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas1"],
        np.average(all_data["order1_fit"], axis=1)
        - np.std(all_data["order1_fit"], axis=1),
        np.average(all_data["order1_fit"], axis=1)
        + np.std(all_data["order1_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas2"],
        np.average(all_data["order2_fit"], axis=1)
        - np.std(all_data["order2_fit"], axis=1),
        np.average(all_data["order2_fit"], axis=1)
        + np.std(all_data["order2_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1)
        - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1)
        + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="small", loc="upper left")
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    # plt.xlim(-0.01, all_data["lambdas0"][-1] * 1.1)
    # plt.xlim(-0.01, 0.065)
    # plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)
    # plt.ylim(-0.005, 0.06)
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E_{\lambda}$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands.pdf"), metadata=_metadata)
    plt.ylim(-0.015, 0.15)
    plt.savefig(plotdir / ("lambda_dep_bands_ylim.pdf"), metadata=_metadata)


    if fit_data:
        lmb_range = fit_data["lmb_range"]
        lmb_range0 = fit_data["lmb_range0"]
        lmb_range1 = fit_data["lmb_range1"]
        lmb_range2 = fit_data["lmb_range2"]
        lmb_range3 = fit_data["lmb_range3"]

        plt.fill_between(
            np.array(
                [
                    all_data["lambdas3"][lmb_range3[0]],
                    all_data["lambdas3"][lmb_range3[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[3],
            alpha=0.1,
            linewidth=0,
        )

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
        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[1],
            np.std(fit_data["bootfit3"], axis=0)[1],
        )

        fitBS0 = np.array(
            [fitfunction5(all_data["lambdas0"], *bf) for bf in fit_data["bootfit0"]]
        )
        fitBS1 = np.array(
            [fitfunction5(all_data["lambdas1"], *bf) for bf in fit_data["bootfit1"]]
        )
        fitBS2 = np.array(
            [fitfunction5(all_data["lambdas2"], *bf) for bf in fit_data["bootfit2"]]
        )
        fitBS3 = np.array(
            [fitfunction5(all_data["lambdas3"], *bf) for bf in fit_data["bootfit3"]]
        )

        plt.plot(
            all_data["lambdas0"],
            np.average(fitBS0, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_0}$",
            color=_colors[0],
            linestyle="--",
            linewidth=1,
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas1"],
            np.average(fitBS1, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_1}$",
            color=_colors[1],
            linestyle="--",
            linewidth=1,
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas2"],
            np.average(fitBS2, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_2}$",
            color=_colors[2],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas3"],
            np.average(fitBS3, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )

        plt.legend(fontsize="small", loc="upper left")
        # plt.legend(fontsize="x-small")
        # plt.xlim(-0.01, 0.16)
        # plt.ylim(0, 0.15)
        # plt.xlim(-0.001, 0.045)
        # plt.ylim(-0.003, 0.035)
        # plt.xlim(-0.01, all_data["lambdas0"][-1] * 1.1)
        plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
        # plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)
        # plt.ylim(np.average(all_data["order3_fit"], axis=1)[0] * 0.9, np.average(all_data["order3_fit"], axis=1)[-1] * 1.1)
        plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_bands_fit.pdf"), metadata=_metadata)
        plt.ylim(0, 0.15)
        plt.savefig(plotdir / ("lambda_dep_bands_fit_ylim.pdf"), metadata=_metadata)

    plt.close()
    return

def plot_lmb_dep_bw(all_data, plotdir):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    plt.figure(figsize=(9, 6))
    # plt.figure(figsize=(6, 5))
    plt.fill_between(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1)
        - np.std(all_data["order0_fit"], axis=1),
        np.average(all_data["order0_fit"], axis=1)
        + np.std(all_data["order0_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=1.5,
        linestyle='solid',
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas1"],
        np.average(all_data["order1_fit"], axis=1)
        - np.std(all_data["order1_fit"], axis=1),
        np.average(all_data["order1_fit"], axis=1)
        + np.std(all_data["order1_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=1.5,
        linestyle='dashed',
        # linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas2"],
        np.average(all_data["order2_fit"], axis=1)
        - np.std(all_data["order2_fit"], axis=1),
        np.average(all_data["order2_fit"], axis=1)
        + np.std(all_data["order2_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=1.5,
        linestyle='dotted',
        # linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1)
        - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1)
        + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=1.5,
        linestyle='dashdot',
        # linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E_{\lambda}$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands_bw.pdf"), metadata=_metadata)
    plt.ylim(-0.015, 0.15)
    plt.savefig(plotdir / ("lambda_dep_bands_bw_ylim.pdf"), metadata=_metadata)
    plt.close()
    return
def plot_lmb_dep_bw_pres(all_data, plotdir):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    plt.figure(figsize=(6, 5))
    plt.fill_between(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1)
        - np.std(all_data["order0_fit"], axis=1),
        np.average(all_data["order0_fit"], axis=1)
        + np.std(all_data["order0_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=1.5,
        linestyle='solid',
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas1"],
        np.average(all_data["order1_fit"], axis=1)
        - np.std(all_data["order1_fit"], axis=1),
        np.average(all_data["order1_fit"], axis=1)
        + np.std(all_data["order1_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=1.5,
        linestyle='dashed',
        # linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas2"],
        np.average(all_data["order2_fit"], axis=1)
        - np.std(all_data["order2_fit"], axis=1),
        np.average(all_data["order2_fit"], axis=1)
        + np.std(all_data["order2_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=1.5,
        linestyle='dotted',
        # linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1)
        - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1)
        + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=1.5,
        linestyle='dashdot',
        # linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    # plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E_{\lambda}$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.ylim(-0.015, 0.15)
    plt.savefig(plotdir / ("lambda_dep_bands_bw_ylim_pres.pdf"), metadata=_metadata)
    plt.savefig(plotdir / ("lambda_dep_bands_bw_ylim_pres.png"), dpi=500, metadata=_metadata)
    plt.close()
    return

def plot_lmb_dep_abs(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    Take the absolute value of the energy shift
    """

    order0_delta_e = abs(all_data["order0_fit"])
    order1_delta_e = abs(all_data["order1_fit"])
    order2_delta_e = abs(all_data["order2_fit"])
    order3_delta_e = abs(all_data["order3_fit"])


    plt.figure(figsize=(9, 6))
    plt.fill_between(
        all_data["lambdas0"],
        np.average(order0_delta_e, axis=1)
        - np.std(order0_delta_e, axis=1),
        np.average(order0_delta_e, axis=1)
        + np.std(order0_delta_e, axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas1"],
        np.average(order1_delta_e, axis=1)
        - np.std(order1_delta_e, axis=1),
        np.average(order1_delta_e, axis=1)
        + np.std(order1_delta_e, axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas2"],
        np.average(order2_delta_e, axis=1)
        - np.std(order2_delta_e, axis=1),
        np.average(order2_delta_e, axis=1)
        + np.std(order2_delta_e, axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas3"],
        np.average(order3_delta_e, axis=1)
        - np.std(order3_delta_e, axis=1),
        np.average(order3_delta_e, axis=1)
        + np.std(order3_delta_e, axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="x-small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$|\Delta E|$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(plotdir / ("lambda_dep_bands.pdf"), metadata=_metadata)
    plt.ylim(-0.015, 0.15)
    plt.savefig(plotdir / ("lambda_dep_bands_ylim_abs.pdf"), metadata=_metadata)
    return


def plot_lmb_dep4(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    # print(all_data["lambdas0"])
    # print(
    #     np.average(all_data["order0_fit"], axis=1)
    #     - np.std(all_data["order0_fit"], axis=1)
    # )
    plt.figure(figsize=(6, 5))
    plt.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1)
        - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1)
        + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="x-small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands_4.pdf"), metadata=_metadata)

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        lmb_range3 = fit_data["lmb_range3"]

        # plt.fill_between(
        #     np.array(
        #         [
        #             all_data["lambdas3"][lmb_range3[0]],
        #             all_data["lambdas3"][lmb_range3[-1]],
        #         ]
        #     ),
        #     np.array([-10, -10]),
        #     np.array([10, 10]),
        #     color=_colors[3],
        #     alpha=0.1,
        #     linewidth=0,
        # )
        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[1],
            np.std(fit_data["bootfit3"], axis=0)[1],
        )

        fitBS3 = np.array(
            [fitfunction5(all_data["lambdas3"], *bf) for bf in fit_data["bootfit3"]]
        )

        plt.plot(
            all_data["lambdas3"],
            np.average(fitBS3, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            # label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            # + "\n"
            # + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )

        plt.legend(fontsize="x-small", loc="upper left")
        plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
        plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_bands_fit_4.pdf"), metadata=_metadata)
        plt.ylim(0, 0.15)
        plt.savefig(plotdir / ("lambda_dep_bands_fit_ylim_4.pdf"), metadata=_metadata)

    plt.close()
    return


def plot_lmb_dep4_1par(all_data, plotdir, fit_data, delta_E_fix):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    # print(all_data["lambdas0"])
    # print(
    #     np.average(all_data["order0_fit"], axis=1)
    #     - np.std(all_data["order0_fit"], axis=1)
    # )
    plt.figure(figsize=(9, 6))
    plt.fill_between(
        all_data["lambdas3"],
        np.average(all_data["order3_fit"], axis=1)
        - np.std(all_data["order3_fit"], axis=1),
        np.average(all_data["order3_fit"], axis=1)
        + np.std(all_data["order3_fit"], axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="x-small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(plotdir / ("lambda_dep_bands_4.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        # lmb_range3 = fit_data["lmb_range3"]

        plt.fill_between(
            np.array(
                [
                    all_data["lambdas3"][lmb_range[0]],
                    all_data["lambdas3"][lmb_range[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[3],
            alpha=0.1,
            linewidth=0,
        )
        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0),
            np.std(fit_data["bootfit3"], axis=0),
        )

        fitBS3 = np.array(
            [
                fitfunction6(all_data["lambdas3"], *bf, delta_E_fix)
                for bf in fit_data["bootfit3"]
            ]
        )

        plt.plot(
            all_data["lambdas3"],
            np.average(fitBS3, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )

        plt.legend(fontsize="x-small", loc="upper left")
        plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
        plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_bands_fit_4_1par.pdf"), metadata=_metadata)
        plt.ylim(0, 0.15)
        plt.savefig(
            plotdir / ("lambda_dep_bands_fit_ylim_4_1par.pdf"), metadata=_metadata
        )

    plt.close()
    return


def main():
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    pars = params(0)
    nboot = 200
    nbin = 1

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
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
    print("datadir: ", datadir / ("lambda_dep.pkl"))

    t_range = np.arange(config["t_range0"], config["t_range1"])
    time_choice = config["time_choice"]
    delta_t = config["delta_t"]
    lmb_val = config["lmb_val"]

    # Read data from the pickle file
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "rb",
    ) as file_in:
        data = pickle.load(file_in)

    # Filter out data points with a high reduced chi-squared value
    chisq_tol = 1.5  # 1.7
    redchisq0 = np.array([d["red_chisq0"] for d in data])
    redchisq1 = np.array([d["red_chisq1"] for d in data])
    redchisq2 = np.array([d["red_chisq2"] for d in data])
    redchisq3 = np.array([d["red_chisq3"] for d in data])
    indices0 = np.where(redchisq0 <= chisq_tol)[0]
    indices1 = np.where(redchisq1 <= chisq_tol)[0]
    indices2 = np.where(redchisq2 <= chisq_tol)[0]
    indices3 = np.where(redchisq3 <= chisq_tol)[0]
    fitlist0 = [data[ind] for ind in indices0]
    fitlist1 = [data[ind] for ind in indices1]
    fitlist2 = [data[ind] for ind in indices2]
    fitlist3 = [data[ind] for ind in indices3]
    fitlists = [fitlist0, fitlist1, fitlist2, fitlist3]

    lmb_range = np.arange(config["lmb_init"], config["lmb_final"])

    # Fit to the lambda dependence for each order
    fit_data = {"lmb_range": lmb_range}
    for order in np.arange(4):
        lmb_range = np.arange(config["lmb_init"], config["lmb_final"])
        print("\n==========\nOrder: ", order)
        print("lmb_range = ", lmb_range)
        lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
            fitlists[order], order, lmb_range
        )
        fit_data[f"lmb_range{order}"] = lmb_range
        fit_data[f"bootfit{order}"] = bootfit
        fit_data[f"redchisq{order}"] = redchisq_fit

    # print([key for key in fit_data])
    # with open(datadir / (f"matrix_element.pkl"), "wb") as file_out:
    #     pickle.dump(fit_data, file_out)

    all_data = {
        "lambdas0": np.array([fit[f"lambdas"] for fit in fitlist0]),
        "lambdas1": np.array([fit[f"lambdas"] for fit in fitlist1]),
        "lambdas2": np.array([fit[f"lambdas"] for fit in fitlist2]),
        "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3]),
        "time_choice": data[0]["time_choice"],
        "delta_t": data[0]["delta_t"],
    }
    for order in np.arange(4):
        all_data[f"order{order}_fit"] = np.array(
            [fit[f"order{order}_fit"][:, 1] for fit in fitlists[order]]
        )
        all_data[f"redchisq{order}"] = np.array(
            [fit[f"red_chisq{order}"] for fit in fitlists[order]]
        )

    plot_lmb_depR(all_data, plotdir, fit_data)
    plot_lmb_dep_bw(all_data, plotdir)
    plot_lmb_dep_bw_pres(all_data, plotdir)
    plot_lmb_dep_abs(all_data, plotdir, fit_data)
    plot_lmb_dep4(all_data, plotdir, fit_data)

    delta_E_fix = np.average(data[0]["weighted_energy_nucldivsigma"])
    # print('\n\n',delta_E_fix)
    fit_data = {"lmb_range": lmb_range}
    bootfit, redchisq = fit_lambda_dep_2(
        fitlist3, delta_E_fix, order, lmb_range, fitfunction6
    )
    fit_data[f"bootfit3"] = bootfit
    fit_data[f"redchisq3"] = redchisq
    plot_lmb_dep4_1par(all_data, plotdir, fit_data, delta_E_fix)


if __name__ == "__main__":
    main()
