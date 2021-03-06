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
from matplotlib import colors

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
    deltaE = 0.5 * np.sqrt(Delta_E ** 2 + 4 * lmb ** 2 * matrix_element ** 2)
    return deltaE


def fit_lmb(ydata, function, lambdas, plotdir, p0=None, order=1):
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
    norms = np.einsum("i,j->ij", diag, diag) ** 0.5
    covmat_norm = covmat / norms

    # Calculate the eigenvalues of the covariance matrix
    eval_left, evec_left = np.linalg.eig(covmat)
    # print('\nevals: ', eval_left)
    plt.figure(figsize=(5, 4))
    plt.scatter(np.arange(len(eval_left)), eval_left)
    plt.ylim(np.min(eval_left), 1.1*np.max(eval_left))
    plt.semilogy()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(plotdir / (f"evals_{order}.pdf"))
    plt.close()
    sorted_evals = np.sort(eval_left)[::-1]
    # print(sorted_evals)
    svd = 3 #How many singular values do we want to keep for the inversion
    rcond = (sorted_evals[svd-1] - sorted_evals[svd+1]) / 2 / sorted_evals[0]
    # print(rcond)
    covmat_inverse = np.linalg.pinv(covmat, rcond=rcond)
    # covmat_inverse = linalg.pinv(covmat)

    # u_, s_, v_ = np.linalg.svd(covmat)
    # print('singular values: ', s_)

    plt.figure(figsize=(11, 11))
    # mat = plt.matshow(np.linalg.inv(covmat))
    # mat = plt.matshow(covmat)
    mat = plt.matshow(covmat_inverse)
    plt.colorbar(mat, shrink=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"cov_matrix_inverse_{order}.pdf"))
    plt.close()
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
    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, covmat_inverse)
    # print("fit_avg", popt_avg)
    p0 = popt_avg
    # dof = len(lambdas)
    dof = svd-2
    redchisq = chisq / dof
    bootfit = []
    for iboot, values in enumerate(ydata):
        # print(iboot)
        popt, pcov = curve_fit(
            function,
            lambdas,
            values,
            sigma=diag_sigma,
            # maxfev=4000,
            p0=p0,
            bounds=bounds,
        )  # , p0=popt_avg)
        # print(popt)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    print("bootfit", np.average(bootfit, axis=0))
    return bootfit, redchisq, chisq


def plot_lmb_dep(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift"""
    plt.figure(figsize=(9, 6))
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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
    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    plt.xlim(-0.01, all_data["lambdas1"][-1] * 1.1)
    plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        fitBS0 = np.array(
            [
                fitfunction5(all_data["lambdas0"][lmb_range], *bf)
                for bf in fit_data["bootfit0"]
            ]
        )
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

        plt.fill_between(
            all_data["lambdas0"][lmb_range],
            np.average(fitBS0, axis=0) - np.std(fitBS0, axis=0),
            np.average(fitBS0, axis=0) + np.std(fitBS0, axis=0),
            alpha=0.3,
            color=_colors[0],
            # label=rf"$\textrm{{M.E.}}={m_e_0}$",
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_0}$",
        )
        fitBS1 = np.array(
            [
                fitfunction5(all_data["lambdas1"][lmb_range], *bf)
                for bf in fit_data["bootfit1"]
            ]
            # [fitfunction5(lambdas1[:fitlim], *bf) for bf in fit_data["bootfit1"]]
        )
        print(np.std(fitBS1, axis=0))

        plt.fill_between(
            all_data["lambdas1"][lmb_range],  # [:fitlim],
            np.average(fitBS1, axis=0) - np.std(fitBS1, axis=0),
            np.average(fitBS1, axis=0) + np.std(fitBS1, axis=0),
            alpha=0.3,
            color=_colors[1],
            # label=rf"$\textrm{{M.E.}}={m_e_1}$",
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_1}$",
        )
        fitBS2 = np.array(
            [
                fitfunction5(all_data["lambdas2"][lmb_range], *bf)
                for bf in fit_data["bootfit2"]
            ]
        )
        print(np.std(fitBS2, axis=0))
        plt.fill_between(
            all_data["lambdas2"][lmb_range],
            np.average(fitBS2, axis=0) - np.std(fitBS2, axis=0),
            np.average(fitBS2, axis=0) + np.std(fitBS2, axis=0),
            alpha=0.3,
            color=_colors[2],
            # label=rf"$\textrm{{M.E.}}={m_e_2}$",
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_2}$",
        )
        fitBS3 = np.array(
            [
                fitfunction5(all_data["lambdas3"][lmb_range], *bf)
                for bf in fit_data["bootfit3"]
            ]
        )
        print(np.std(fitBS3, axis=0))
        plt.fill_between(
            all_data["lambdas3"][lmb_range],
            np.average(fitBS3, axis=0) - np.std(fitBS3, axis=0),
            np.average(fitBS3, axis=0) + np.std(fitBS3, axis=0),
            alpha=0.3,
            color=_colors[3],
            # label=rf"$\textrm{{M.E.}}={m_e_3}$",
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
        )

        plt.legend(fontsize="x-small")
        # plt.xlim(-0.01, 0.16)
        # plt.ylim(0, 0.15)
        # plt.xlim(-0.001, 0.045)
        # plt.ylim(-0.003, 0.035)
        plt.xlim(-0.01, all_data["lambdas0"][-1] * 1.1)
        plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_fit.pdf"))

        # plt.xlim(-0.005, 0.08)
        # plt.ylim(0.015, 0.065)
        plt.xlim(-0.0001, 0.025)
        plt.ylim(-0.0002, 0.015)
        plt.savefig(plotdir / ("lambda_dep_zoom.pdf"))
        plt.close()

    plt.close()
    return


def plot_lmb_depR(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift

    Where the plot uses colored bands to show the dependence"""
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
    plt.legend(fontsize="x-small",loc='upper left')
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    plt.xlim(-0.01, all_data["lambdas0"][-1] * 1.1)
    plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        print(lmb_range)
        plt.fill_between(np.array([-1,all_data["lambdas0"][lmb_range[0]]]), np.array([-1,-1]), np.array([1,1]), color='k', alpha=0.2, linewidth=0)
        plt.fill_between(np.array([all_data["lambdas0"][lmb_range[-1]], 1]), np.array([-1,-1]), np.array([1,1]), color='k', alpha=0.2, linewidth=0)

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
            [
                fitfunction5(all_data["lambdas0"], *bf)
                for bf in fit_data["bootfit0"]
            ]
        )
        fitBS1 = np.array(
            [
                fitfunction5(all_data["lambdas1"], *bf)
                for bf in fit_data["bootfit1"]
            ]
        )
        fitBS2 = np.array(
            [
                fitfunction5(all_data["lambdas2"], *bf)
                for bf in fit_data["bootfit2"]
            ]
        )
        fitBS3 = np.array(
            [
                fitfunction5(all_data["lambdas3"], *bf)
                for bf in fit_data["bootfit3"]
            ]
        )


        plt.plot(
            all_data["lambdas0"],
            np.average(fitBS0, axis=0),
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_0}$",
            color=_colors[0],
            linestyle='--',
            linewidth=1,
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas1"],
            np.average(fitBS1, axis=0),
            # np.average(all_data["order1_fit"], axis=1),
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_1}$",
            color=_colors[1],
            linestyle='--',
            linewidth=1,
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas2"],
            np.average(fitBS2, axis=0),
            # np.average(all_data["order2_fit"], axis=1),
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_2}$",
            color=_colors[2],
            linewidth=1,
            linestyle='--',
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas3"],
            np.average(fitBS3, axis=0),
            # np.average(all_data["order3_fit"], axis=1),
            label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle='--',
            alpha=0.9,
        )
        



        # print(m_e_0)

        # plt.fill_between(
        #     all_data["lambdas0"][lmb_range],
        #     np.average(fitBS0, axis=0) - np.std(fitBS0, axis=0),
        #     np.average(fitBS0, axis=0) + np.std(fitBS0, axis=0),
        #     alpha=0.3,
        #     color=_colors[0],
        #     # label=rf"$\textrm{{M.E.}}={m_e_0}$",
        #     label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:2.3}$"
        #     + "\n"
        #     + rf"$\textrm{{M.E.}}={m_e_0}$",
        # )
        # fitBS1 = np.array(
        #     [
        #         fitfunction5(all_data["lambdas1"][lmb_range], *bf)
        #         for bf in fit_data["bootfit1"]
        #     ]
        #     # [fitfunction5(lambdas1[:fitlim], *bf) for bf in fit_data["bootfit1"]]
        # )
        # print(np.std(fitBS1, axis=0))

        # plt.fill_between(
        #     all_data["lambdas1"][lmb_range],  # [:fitlim],
        #     np.average(fitBS1, axis=0) - np.std(fitBS1, axis=0),
        #     np.average(fitBS1, axis=0) + np.std(fitBS1, axis=0),
        #     alpha=0.3,
        #     color=_colors[1],
        #     # label=rf"$\textrm{{M.E.}}={m_e_1}$",
        #     label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
        #     + "\n"
        #     + rf"$\textrm{{M.E.}}={m_e_1}$",
        # )
        # fitBS2 = np.array(
        #     [
        #         fitfunction5(all_data["lambdas2"][lmb_range], *bf)
        #         for bf in fit_data["bootfit2"]
        #     ]
        # )
        # print(np.std(fitBS2, axis=0))
        # plt.fill_between(
        #     all_data["lambdas2"][lmb_range],
        #     np.average(fitBS2, axis=0) - np.std(fitBS2, axis=0),
        #     np.average(fitBS2, axis=0) + np.std(fitBS2, axis=0),
        #     alpha=0.3,
        #     color=_colors[2],
        #     # label=rf"$\textrm{{M.E.}}={m_e_2}$",
        #     label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
        #     + "\n"
        #     + rf"$\textrm{{M.E.}}={m_e_2}$",
        # )
        # fitBS3 = np.array(
        #     [
        #         fitfunction5(all_data["lambdas3"][lmb_range], *bf)
        #         for bf in fit_data["bootfit3"]
        #     ]
        # )
        # print(np.std(fitBS3, axis=0))
        # plt.fill_between(
        #     all_data["lambdas3"][lmb_range],
        #     np.average(fitBS3, axis=0) - np.std(fitBS3, axis=0),
        #     np.average(fitBS3, axis=0) + np.std(fitBS3, axis=0),
        #     alpha=0.3,
        #     color=_colors[3],
        #     # label=rf"$\textrm{{M.E.}}={m_e_3}$",
        #     label = rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
        #     + "\n"
        #     + rf"$\textrm{{M.E.}}={m_e_3}$",
        # )

        plt.legend(fontsize="x-small",loc='upper left')
        # plt.legend(fontsize="x-small")
        # plt.xlim(-0.01, 0.16)
        # plt.ylim(0, 0.15)
        # plt.xlim(-0.001, 0.045)
        # plt.ylim(-0.003, 0.035)
        plt.xlim(-0.01, all_data["lambdas0"][-1] * 1.1)
        plt.ylim(-0.005, np.average(all_data["order0_fit"], axis=1)[-1] * 1.1)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_bands_fit.pdf"))

    plt.close()
    return


def fit_const(xdata, data_set, lmb_range):
    bounds = ([0], [np.inf])
    ydata_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)

    eval_left, evec_left = np.linalg.eig(covmat)
    print('\nevals: ', eval_left)
    sorted_evals = np.sort(eval_left)[::-1]
    # rcond = (sorted_evals[1] - sorted_evals[2])/2
    rcond = (sorted_evals[1] - sorted_evals[2]) / 2 / sorted_evals[0]
    covmat_inverse = np.linalg.pinv(covmat, rcond=rcond)

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
    chisq = ff.chisqfn2(popt_avg, function, xdata, ydata_avg, covmat_inverse)
    print("\n chisq", chisq, popt_avg)

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
    """Make a plot of the lambda dependence of the energy shift and divide it by lambda to show the linear behaviour"""

    xdata0 = all_data["lambdas0"][lmb_range]
    data_set0 = np.einsum(
        "ij,j->ij", all_data["order0_fit"][lmb_range].T, xdata0 ** (-1)
    )
    fit_param0, chisq0 = fit_const(xdata0, data_set0, lmb_range)
    xdata1 = all_data["lambdas1"][lmb_range]
    data_set1 = np.einsum(
        "ij,j->ij", all_data["order1_fit"][lmb_range].T, xdata1 ** (-1)
    )
    fit_param1, chisq1 = fit_const(xdata1, data_set1, lmb_range)
    xdata2 = all_data["lambdas2"][lmb_range]
    data_set2 = np.einsum(
        "ij,j->ij", all_data["order2_fit"][lmb_range].T, xdata2 ** (-1)
    )
    fit_param2, chisq2 = fit_const(xdata2, data_set2, lmb_range)
    xdata3 = all_data["lambdas3"][lmb_range]
    data_set3 = np.einsum(
        "ij,j->ij", all_data["order3_fit"][lmb_range].T, xdata3 ** (-1)
    )
    fit_param3, chisq3 = fit_const(xdata3, data_set3, lmb_range)

    plt.figure(figsize=(9, 6))

    ydata0 = np.divide(np.average(all_data["order0_fit"], axis=1), all_data["lambdas0"], out=np.zeros_like(np.average(all_data["order0_fit"], axis=1)), where=all_data["lambdas0"]!=0)
    plt.errorbar(
        all_data["lambdas0"],
        ydata0,
        np.std(all_data["order0_fit"], axis=1) / all_data["lambdas0"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        all_data["lambdas1"] + 0.0001,
        np.average(all_data["order1_fit"], axis=1) / all_data["lambdas1"],
        np.std(all_data["order1_fit"], axis=1) / all_data["lambdas1"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        all_data["lambdas2"] + 0.0002,
        np.average(all_data["order2_fit"], axis=1) / all_data["lambdas2"],
        np.std(all_data["order2_fit"], axis=1) / all_data["lambdas2"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        all_data["lambdas3"] + 0.0003,
        np.average(all_data["order3_fit"], axis=1) / all_data["lambdas3"],
        np.std(all_data["order3_fit"], axis=1) / all_data["lambdas3"],
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    # label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:0.2}$"
    plt.plot(
        xdata0,
        [np.average(fit_param0)] * len(xdata0),
        color=_colors[0],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {chisq0:0.2}$",
    )
    plt.plot(
        xdata1,
        [np.average(fit_param1)] * len(xdata1),
        color=_colors[1],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {chisq1:0.2}$",
    )
    plt.plot(
        xdata2,
        [np.average(fit_param2)] * len(xdata2),
        color=_colors[2],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {chisq2:0.2}$",
    )
    plt.plot(
        xdata3,
        [np.average(fit_param3)] * len(xdata3),
        color=_colors[3],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {chisq3:0.2}$",
    )
    plt.fill_between(
        xdata0,
        np.average(fit_param0) - np.std(fit_param0),
        np.average(fit_param0) + np.std(fit_param0),
        alpha=0.3,
        color=_colors[0],
    )
    plt.fill_between(
        xdata1,
        np.average(fit_param1) - np.std(fit_param1),
        np.average(fit_param1) + np.std(fit_param1),
        alpha=0.3,
        color=_colors[1],
    )
    plt.fill_between(
        xdata2,
        np.average(fit_param2) - np.std(fit_param2),
        np.average(fit_param2) + np.std(fit_param2),
        alpha=0.3,
        color=_colors[2],
    )
    plt.fill_between(
        xdata3,
        np.average(fit_param3) - np.std(fit_param3),
        np.average(fit_param3) + np.std(fit_param3),
        alpha=0.3,
        color=_colors[3],
    )

    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    # plt.xlim(-0.01, lambdas3[-1] * 1.1)
    # plt.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.1)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E / \lambda$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_divlmb.pdf"))

    plt.close()
    return



def main_loop():
    """ Plot the results of the loop over the fit windows of the fit to the lambda dependence of the energy shift """
    plt.style.use("./mystyle.txt")

    pars = params(0)
    nboot = 200
    nbin = 1

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir_theta2.yaml"  # default file
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Set parameters to defaults defined in another YAML file
    with open("defaults.yaml") as f:
        defaults = yaml.safe_load(f)
    for key, value in defaults.items():
        config.setdefault(key, value)

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
        / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
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

    lambdas3 =  np.array([fit[f"lambdas"] for fit in fitlist3])
    # redchisq3 =  np.array([fit[f"redchisq3"] for fit in fitlist3])
    order3_fit =  np.array([fit[f"order3_fit"][:, 1] for fit in fitlist3])

    p0 = (1e-3, 0.7)
    fitlim = 30

    # lambdas = data["lambdas"]
    # order0_fit = data["order0_fit"]
    # order1_fit = data["order1_fit"]
    # order2_fit = data["order2_fit"]
    # order3_fit = data["order3_fit"]
    # redchisq = data["redchisq"]
    # time_choice = data["time_choice"]
    # delta_t = data["delta_t"]

    # # Filter out data points with a high reduced chi-squared value
    # chisq_tol = 1.5  # 1.7
    # order0_fit = order0_fit[np.where(redchisq[0] <= chisq_tol)]
    # lambdas0 = lambdas[np.where(redchisq[0] <= chisq_tol)]
    # order1_fit = order1_fit[np.where(redchisq[1] <= chisq_tol)]
    # lambdas1 = lambdas[np.where(redchisq[1] <= chisq_tol)]
    # order2_fit = order2_fit[np.where(redchisq[2] <= chisq_tol)]
    # lambdas2 = lambdas[np.where(redchisq[2] <= chisq_tol)]
    # order3_fit = order3_fit[np.where(redchisq[3] <= chisq_tol)]
    # lambdas3 = lambdas[np.where(redchisq[3] <= chisq_tol)]

    all_data = {
        "lambdas3": lambdas3,
        "order3_fit": order3_fit,
        # "redchisq": redchisq3,
        "time_choice": time_choice,
        "delta_t": delta_t,
    }

    # Fit to the lambda dependence at each order in lambda
    with open(datadir / (f"matrix_elements_loop.pkl"), "rb") as file_in:
        fit_data_list = pickle.load(file_in)

    x_coord = np.array([elem["lmb_range"][0] for elem in fit_data_list])
    y_coord = np.array([elem["lmb_range"][-1] for elem in fit_data_list])
    redchisq_vals = np.array([elem["redchisq3"] for elem in fit_data_list])
    print('redchisq = ', redchisq_vals)

    min_chisq = np.min(redchisq_vals)
    min_chisq_index = np.argmin(redchisq_vals)
    best_chisq_index = np.argmin(abs(redchisq_vals-1))
    best_chisq = redchisq_vals[best_chisq_index]

    print(f"min red chisq. = {min_chisq}, x = {x_coord[min_chisq_index]}, y = {y_coord[min_chisq_index]}")
    print(f"min red chisq. = {min_chisq}, x = {all_data['lambdas3'][x_coord[min_chisq_index]]}, y = {all_data['lambdas3'][y_coord[min_chisq_index]]}")
    print(f"best red chisq. = {best_chisq}, x = {x_coord[best_chisq_index]}, y = {y_coord[best_chisq_index]}")
    print(f"best red chisq. = {best_chisq}, x = {all_data['lambdas3'][x_coord[best_chisq_index]]}, y = {all_data['lambdas3'][y_coord[best_chisq_index]]}")

    unique_x = np.unique(x_coord)
    unique_y = np.unique(y_coord)
    min_x = np.min(x_coord)
    min_y = np.min(y_coord)

    # matrix = np.zeros((len(unique_x), len(unique_y)))
    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    # # uniques = np.unique(x_coord)
    # for x in np.arange(x_coord):
    #     for y in np.arange(y_coord):
    #         matrix[x-min_x, y-min_y] = redchisq_vals[x+y]

    for i,x in enumerate(x_coord):
        # for j,y in enumerate(y_coord):
        matrix[x-min_x, y_coord[i]-min_y] = redchisq_vals[i]
        # if matrix[x-min_x, y_coord[i]-min_y]>50:
        #     matrix[x-min_x, y_coord[i]-min_y] =50

    # for i, x in enumerate(x_coord):
    #     matrix[x-min_x, x_coord[i]-min_y] = redchisq_vals[i]
    # print(matrix)

    # matrix = np.array([x_coord, y_coord, redchisq_vals/np.max(redchisq_vals)])
    # print(matrix)
    # print(np.shape(matrix))

    plt.figure(figsize=(5, 4))
    # mat = plt.matshow(matrix)
    # mat = plt.imshow(matrix)
    # mat = plt.pcolormesh(unique_x, unique_y, matrix/np.max(redchisq_vals))
    # mat = plt.pcolormesh(unique_x, unique_y, matrix.T, cmap = 'Blues', norm=colors.LogNorm(vmin=matrix.min(), vmax=matrix.max()))
    # mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu', norm=colors.LogNorm(vmin=0.5, vmax=np.max(matrix)))
    # mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu', norm=colors.LogNorm(vmin=0.5, vmax=200))
    mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu', vmin=0, vmax=8)
    mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu', vmax=8)
    # mat = plt.pcolormesh(unique_x, unique_y, matrix.T, cmap = 'Blues', norm=colors.LogNorm())
    # plt.colorbar(mat, shrink=0.5)
    plt.colorbar(mat, label = r"$\chi^2_{\textrm{dof}}$")
    plt.xlabel(r"$\lambda_{\textrm{min}}$")
    plt.ylabel(r"$\lambda_{\textrm{max}}$")
    plt.savefig(plotdir / (f"chisq_matrix.pdf"))
    plt.close()


    #--------------------------------------------------------------------------------
    # Plot the matrix element in a matrix

    # redchisq_vals = np.array([elem["redchisq3"] for elem in fit_data_list])
    me_vals = np.array([np.average(elem["bootfit3"],axis=0)[1] for elem in fit_data_list])
    # matrix = np.zeros((len(unique_x), len(unique_y)))
    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i,x in enumerate(x_coord):
        matrix[x-min_x, y_coord[i]-min_y] = me_vals[i]


    plt.figure(figsize=(5, 4))

    # mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu') #, norm=colors.LogNorm(vmin=1e-1, vmax=np.max(matrix)))
    # mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu',vmax=np.max(matrix), vmin = np.min(np.where(matrix==0,1,matrix)))
    mat = plt.pcolormesh(all_data["lambdas3"][unique_x], all_data["lambdas3"][unique_y], matrix.T, cmap = 'GnBu')
    plt.colorbar(mat, label = r"Matrix element")
    plt.xlabel(r"$\lambda_{\textrm{min}}$")
    plt.ylabel(r"$\lambda_{\textrm{max}}$")
    plt.savefig(plotdir / (f"me_matrix.pdf"))
    plt.close()


if __name__ == "__main__":
    main_loop()
