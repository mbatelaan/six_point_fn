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
        (E_nucl_p - E_sigma_p) ** 2 + 4 * lmb**2 * matrix_element**2
    )
    return deltaE


def fitfunction3(lmb, pars0, pars2):
    deltaE = 0.5 * (pars0 + pars1) + 0.5 * np.sqrt(
        (pars0 - pars1) ** 2 + 4 * lmb**2 * pars2**2
    )
    return deltaE


def fitfunction4(lmb, E_nucl_p, E_sigma_p, matrix_element):
    deltaE = 0.5 * np.sqrt(
        (E_nucl_p - E_sigma_p) ** 2 + 4 * lmb**2 * matrix_element**2
    )
    return deltaE


def fitfunction5(lmb, Delta_E, matrix_element):
    deltaE = np.sqrt(Delta_E**2 + 4 * lmb**2 * matrix_element**2)
    return deltaE


def fitfunction_4(lmb, Delta_E, A, B):
    deltaE = np.sqrt(Delta_E**2 + 4 * lmb**2 * A**2, lmb**4 * B**2)
    return deltaE


def fit_lmb(ydata, function, lambdas, plotdir, p0=None, order=1, svd_inv=False):
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

    covmat = np.cov(data_set.T)
    diag = np.diagonal(covmat)
    norms = np.einsum("i,j->ij", diag, diag) ** 0.5
    covmat_norm = covmat / norms

    if svd_inv:
        # Calculate the eigenvalues of the covariance matrix
        eval_left, evec_left = np.linalg.eig(covmat_norm)
        # plt.figure(figsize=(5, 4))
        # plt.scatter(np.arange(len(eval_left)), eval_left)
        # plt.ylim(np.min(eval_left), 1.1*np.max(eval_left))
        # plt.semilogy()
        # plt.grid(True, alpha=0.4)
        # plt.tight_layout()
        # plt.savefig(plotdir / (f"evals_{order}.pdf"))
        # plt.close()
        sorted_evals = np.sort(eval_left)[::-1]
        svd = 10  # How many singular values do we want to keep for the inversion

        lmb_min = np.sum(sorted_evals[svd:]) / len(sorted_evals[svd:])
        print(f"lmb_min = {lmb_min}")
        print(f"sorted_evals = {sorted_evals}")
        K = len(sorted_evals) * np.sum(
            [max(lmb_i, lmb_min) for lmb_i in sorted_evals]
        ) ** (-1)
        print(f"K = {K}")
        for ilmb, lmb in enumerate(sorted_evals[svd:]):
            sorted_evals[svd + ilmb] = K * max(lmb, lmb_min)
        print(f"sorted_evals = {sorted_evals}")

        inv_evals = linalg.inv(np.diag(sorted_evals))
        covmat_inverse = evec_left * inv_evals * linalg.inv(evec_left)
        print(f"covmat_inverse = {covmat_inverse}")
        dof = len(sorted_evals) - 2

        plt.figure(figsize=(5, 4))
        plt.scatter(
            np.arange(len(eval_left)), eval_left, color="b", label="eigenvalues"
        )
        plt.scatter(
            np.arange(len(sorted_evals)),
            sorted_evals,
            color="k",
            label="modified eigenvalues",
        )
        plt.ylim(np.min(eval_left), 1.1 * np.max(eval_left))
        plt.semilogy()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(plotdir / (f"evals_{order}.pdf"))
        plt.close()

        # # Calculate the eigenvalues of the covariance matrix
        # eval_left, evec_left = np.linalg.eig(covmat)
        # # print('\nevals: ', eval_left)
        # plt.figure(figsize=(5, 4))
        # plt.scatter(np.arange(len(eval_left)), eval_left)
        # plt.ylim(np.min(eval_left), 1.1*np.max(eval_left))
        # plt.semilogy()
        # plt.grid(True, alpha=0.4)
        # plt.tight_layout()
        # plt.savefig(plotdir / (f"evals_{order}.pdf"))
        # plt.close()
        # sorted_evals = np.sort(eval_left)[::-1]
        # # print(sorted_evals)
        # svd = 3 #How many singular values do we want to keep for the inversion
        # rcond = (sorted_evals[svd-1] - sorted_evals[svd+1]) / 2 / sorted_evals[0]
        # covmat_inverse = np.linalg.pinv(covmat, rcond=rcond)
        # dof = svd-2

    else:
        covmat_inverse = linalg.pinv(covmat)
        dof = len(lambdas)

    # u_, s_, v_ = np.linalg.svd(covmat)
    # print('singular values: ', s_)

    plt.figure(figsize=(11, 11))
    mat = plt.matshow(covmat_inverse)
    plt.colorbar(mat, shrink=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"cov_matrix_inverse_{order}.pdf"))
    plt.close()

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


def fit_lmb_4(ydata, function, lambdas, plotdir, p0=None):
    """Fit the lambda dependence

    data is a correlator with tht bootstraps on the first index and the time on the second
    lambdas is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """

    # bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    ydata = ydata.T
    # print(np.shape(ydata))
    data_set = ydata
    ydata_avg = np.average(data_set, axis=0)

    covmat = np.cov(data_set.T)
    diag = np.diagonal(covmat)
    norms = np.einsum("i,j->ij", diag, diag) ** 0.5
    covmat_norm = covmat / norms

    covmat_inverse = linalg.pinv(covmat)
    dof = len(lambdas)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)

    popt_avg, pcov_avg = curve_fit(
        function,
        lambdas,
        ydata_avg,
        sigma=diag_sigma,
        # p0=p0,
        # maxfev=4000,
        # bounds=bounds,
    )
    chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, covmat_inverse)
    # print("fit_avg", popt_avg)
    p0 = popt_avg
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
            # p0=p0,
            # bounds=bounds,
        )  # , p0=popt_avg)
        # print(popt)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    print("bootfit", np.average(bootfit, axis=0))
    return bootfit, redchisq, chisq


def plot_lmb_dep(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift"""
    print(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"], axis=1),
        np.std(all_data["order0_fit"],axis=1),
    )
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        all_data["lambdas0"],
        np.average(all_data["order0_fit"],axis=1),
        np.std(all_data["order0_fit"],axis=1),
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
    # plt.ylim(-0.003, 0.16)
    plt.ylim(-0.16, 0.16)
    # plt.xlim(-0.01, 0.22)
    plt.xlim(-0.01, all_data["lambdas3"][-1] * 1.1)
    # plt.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.1)

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
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq0']:2.3}$"
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
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
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
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
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
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
        )

        plt.legend(fontsize="x-small")
        # plt.xlim(-0.01, 0.16)
        # plt.ylim(0, 0.15)
        # plt.xlim(-0.001, 0.045)
        # plt.ylim(-0.003, 0.035)
        plt.xlim(-0.01, all_data["lambdas3"][-1] * 1.1)
        plt.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.1)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_fit.pdf"))

        plt.ylim(-0.005, -0.11)
        plt.savefig(plotdir / ("lambda_dep_fit_ylim.pdf"))

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
    plt.legend(fontsize="x-small", loc="upper left")
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
    plt.ylabel("$\Delta E$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands.pdf"))

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        lmb_range0 = fit_data["lmb_range0"]
        lmb_range1 = fit_data["lmb_range1"]
        lmb_range2 = fit_data["lmb_range2"]
        lmb_range3 = fit_data["lmb_range3"]

        print(lmb_range)
        # plt.fill_between(np.array([-1,all_data["lambdas0"][lmb_range[0]]]), np.array([-1,-1]), np.array([1,1]), color='k', alpha=0.2, linewidth=0)
        # plt.fill_between(np.array([all_data["lambdas0"][lmb_range[-1]], 1]), np.array([-1,-1]), np.array([1,1]), color='k', alpha=0.2, linewidth=0)
        plt.fill_between(
            np.array(
                [
                    all_data["lambdas0"][lmb_range0[0]],
                    all_data["lambdas0"][lmb_range0[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[0],
            alpha=0.1,
            linewidth=0,
        )
        plt.fill_between(
            np.array(
                [
                    all_data["lambdas1"][lmb_range1[0]],
                    all_data["lambdas1"][lmb_range1[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[1],
            alpha=0.1,
            linewidth=0,
        )
        plt.fill_between(
            np.array(
                [
                    all_data["lambdas2"][lmb_range2[0]],
                    all_data["lambdas2"][lmb_range2[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[2],
            alpha=0.1,
            linewidth=0,
        )
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
            # np.average(all_data["order1_fit"], axis=1),
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
            # np.average(all_data["order2_fit"], axis=1),
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
            # np.average(all_data["order3_fit"], axis=1),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle="--",
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

        plt.legend(fontsize="x-small", loc="upper left")
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
        plt.savefig(plotdir / ("lambda_dep_bands_fit.pdf"))
        plt.ylim(0, 0.15)
        plt.savefig(plotdir / ("lambda_dep_bands_fit_ylim.pdf"))

    plt.close()
    return


def fit_const(xdata, data_set, lmb_range):
    bounds = ([0], [np.inf])
    ydata_avg = np.average(data_set, axis=0)
    covmat = np.cov(data_set.T)
    diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)

    # eval_left, evec_left = np.linalg.eig(covmat)
    # print('\nevals: ', eval_left)
    # sorted_evals = np.sort(eval_left)[::-1]
    # # rcond = (sorted_evals[1] - sorted_evals[2])/2
    # rcond = (sorted_evals[1] - sorted_evals[2]) / 2 / sorted_evals[0]
    # covmat_inverse = np.linalg.pinv(covmat, rcond=rcond)

    covmat_inverse = np.linalg.inv(covmat)

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

    ydata0 = np.divide(
        np.average(all_data["order0_fit"], axis=1),
        all_data["lambdas0"],
        out=np.zeros_like(np.average(all_data["order0_fit"], axis=1)),
        where=all_data["lambdas0"] != 0,
    )
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
    plt.ylim(-0.3, 5)
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


def plot_lmb_dep3(all_data, plotdir, fit_data=None):
    """Make a plot of the lambda dependence of the energy shift and divide it by lambda to show the linear behaviour"""

    xdata0 = all_data["lambdas0"]
    data_set0 = np.einsum("ij,i->ij", all_data["order0_fit"][1:], xdata0[1:] ** (-1))
    # fit_param0, chisq0 = fit_const(xdata0, data_set0, lmb_range)
    xdata1 = all_data["lambdas1"]
    data_set1 = np.einsum("ij,i->ij", all_data["order1_fit"][1:], xdata1[1:] ** (-1))
    # fit_param1, chisq1 = fit_const(xdata1, data_set1, lmb_range)
    xdata2 = all_data["lambdas2"]
    data_set2 = np.einsum("ij,i->ij", all_data["order2_fit"][1:], xdata2[1:] ** (-1))
    # fit_param2, chisq2 = fit_const(xdata2, data_set2, lmb_range)
    xdata3 = all_data["lambdas3"]
    data_set3 = np.einsum("ij,i->ij", all_data["order3_fit"][1:], xdata3[1:] ** (-1))
    # fit_param3, chisq3 = fit_const(xdata3, data_set3, lmb_range)

    plt.figure(figsize=(9, 6))

    ydata0 = np.divide(
        np.average(all_data["order0_fit"], axis=1),
        all_data["lambdas0"],
        out=np.zeros_like(np.average(all_data["order0_fit"], axis=1)),
        where=all_data["lambdas0"] != 0,
    )
    print(f"data_set0, {np.shape(data_set0)}")

    plt.fill_between(
        all_data["lambdas0"][1:],
        np.average(data_set0, axis=1) - np.std(data_set0, axis=1),
        np.average(data_set0, axis=1) + np.std(data_set0, axis=1),
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas1"][1:],
        np.average(data_set1, axis=1) - np.std(data_set1, axis=1),
        np.average(data_set1, axis=1) + np.std(data_set1, axis=1),
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas2"][1:],
        np.average(data_set2, axis=1) - np.std(data_set2, axis=1),
        np.average(data_set2, axis=1) + np.std(data_set2, axis=1),
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        linewidth=0,
        alpha=0.3,
    )
    plt.fill_between(
        all_data["lambdas3"][1:],
        np.average(data_set3, axis=1) - np.std(data_set3, axis=1),
        np.average(data_set3, axis=1) + np.std(data_set3, axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=1,
        alpha=0.3,
    )

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        lmb_range0 = fit_data["lmb_range0"]
        lmb_range1 = fit_data["lmb_range1"]
        lmb_range2 = fit_data["lmb_range2"]
        lmb_range3 = fit_data["lmb_range3"]

        # plt.fill_between(np.array([-1,all_data["lambdas0"][lmb_range[0]]]), np.array([-10,-10]), np.array([10,10]), color='k', alpha=0.2, linewidth=0)
        # plt.fill_between(np.array([all_data["lambdas0"][lmb_range[-1]], 1]), np.array([-10,-10]), np.array([10,10]), color='k', alpha=0.2, linewidth=0)

        plt.fill_between(
            np.array(
                [
                    all_data["lambdas0"][lmb_range0[0]],
                    all_data["lambdas0"][lmb_range0[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[0],
            alpha=0.1,
            linewidth=0,
        )
        plt.fill_between(
            np.array(
                [
                    all_data["lambdas1"][lmb_range1[0]],
                    all_data["lambdas1"][lmb_range1[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[1],
            alpha=0.1,
            linewidth=0,
        )
        plt.fill_between(
            np.array(
                [
                    all_data["lambdas2"][lmb_range2[0]],
                    all_data["lambdas2"][lmb_range2[-1]],
                ]
            ),
            np.array([-10, -10]),
            np.array([10, 10]),
            color=_colors[2],
            alpha=0.1,
            linewidth=0,
        )
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
        fitBS0 = np.einsum("ij,j->ij", fitBS0[:, 1:], all_data["lambdas0"][1:] ** (-1))
        fitBS1 = np.array(
            [fitfunction5(all_data["lambdas1"], *bf) for bf in fit_data["bootfit1"]]
        )
        fitBS1 = np.einsum("ij,j->ij", fitBS1[:, 1:], all_data["lambdas1"][1:] ** (-1))
        fitBS2 = np.array(
            [fitfunction5(all_data["lambdas2"], *bf) for bf in fit_data["bootfit2"]]
        )
        fitBS2 = np.einsum("ij,j->ij", fitBS2[:, 1:], all_data["lambdas2"][1:] ** (-1))
        fitBS3 = np.array(
            [fitfunction5(all_data["lambdas3"], *bf) for bf in fit_data["bootfit3"]]
        )
        fitBS3 = np.einsum("ij,j->ij", fitBS3[:, 1:], all_data["lambdas3"][1:] ** (-1))

        plt.plot(
            all_data["lambdas0"][1:],
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
            all_data["lambdas1"][1:],
            np.average(fitBS1, axis=0),
            # np.average(all_data["order1_fit"], axis=1),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq1']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_1}$",
            color=_colors[1],
            linestyle="--",
            linewidth=1,
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas2"][1:],
            np.average(fitBS2, axis=0),
            # np.average(all_data["order2_fit"], axis=1),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq2']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_2}$",
            color=_colors[2],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )
        plt.plot(
            all_data["lambdas3"][1:],
            np.average(fitBS3, axis=0),
            # np.average(all_data["order3_fit"], axis=1),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[3],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )
    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(-0.003, 0.035)
    # plt.xlim(-0.01, 0.22)
    # plt.xlim(-0.01, lambdas3[-1] * 1.1)
    # plt.ylim(-0.005, np.average(all_data["order3_fit"], axis=1)[-1] * 1.1)

    # plt.xlim(-0.01, 0.065)
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, 5)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E / \lambda$")
    # plt.title(rf"$t_{{0}}={all_data['time_choice']}, \Delta t={all_data['delta_t']}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_divlmb.pdf"))

    plt.close()
    return


def plotting_script_diff_2(
    diffG1,
    diffG2,
    diffG3,
    diffG4,
    fitvals,
    t_range,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
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
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim), ylim=(-0.05, 0.25))
    plt.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    plt.xlabel("$t/a$")
    plt.legend(fontsize="x-small")
    plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        plt.show()
    plt.close()
    return


def plot_energy_diffs(data, config, plotdir):
    """Make plots of the effective energy of the ratio of the two correlators from the GEVP. These can be made for all or some lambda values."""
    lambdas = data["lambdas"]
    t_range = np.arange(config["t_range0"], config["t_range1"])
    for i, lmb_val in enumerate(lambdas):
        print(f"Lambda = {lmb_val}\n")
        Gt1_0 = data["order0_corrs"][i, 0]
        Gt2_0 = data["order0_corrs"][i, 1]
        ratio0 = Gt1_0 / Gt2_0
        effmass_ratio0 = stats.bs_effmass(ratio0, time_axis=1, spacing=1)
        bootfit0 = data["order0_fit"][i]

        Gt1_1 = data["order1_corrs"][i, 0]
        Gt2_1 = data["order1_corrs"][i, 1]
        ratio1 = Gt1_1 / Gt2_1
        effmass_ratio1 = stats.bs_effmass(ratio1, time_axis=1, spacing=1)
        bootfit1 = data["order1_fit"][i]

        Gt1_2 = data["order2_corrs"][i, 0]
        Gt2_2 = data["order2_corrs"][i, 1]
        ratio2 = Gt1_2 / Gt2_2
        effmass_ratio2 = stats.bs_effmass(ratio2, time_axis=1, spacing=1)
        bootfit2 = data["order2_fit"][i]

        Gt1_3 = data["order3_corrs"][i, 0]
        Gt2_3 = data["order3_corrs"][i, 1]
        ratio3 = Gt1_3 / Gt2_3
        effmass_ratio3 = stats.bs_effmass(ratio3, time_axis=1, spacing=1)
        bootfit3 = data["order3_fit"][i]

        plotting_script_diff_2(
            effmass_ratio0,
            effmass_ratio1,
            effmass_ratio2,
            effmass_ratio3,
            [bootfit0, bootfit1, bootfit2, bootfit3],
            t_range,
            lmb_val,
            plotdir,
            name="_l" + str(lmb_val) + "_all",
            show=False,
        )
    return


def main():
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

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
    # Need to read in the correct file!!!
    with open(
        datadir / (f"lambda_dep_t{time_choice}_dt{delta_t}.pkl"),
        "rb",
    ) as file_in:
        data = pickle.load(file_in)

    # plot_energy_diffs(data, config, plotdir)

    lambdas = np.array([d["lambdas"] for d in data])
    print("\n\n", np.shape(lambdas))
    print("\n\n", lambdas)
    order0_fit = np.array([d["order0_fit"][:,1] for d in data])
    order1_fit = np.array([d["order1_fit"][:,1] for d in data])
    order2_fit = np.array([d["order2_fit"][:,1] for d in data])
    order3_fit = np.array([d["order3_fit"][:,1] for d in data])
    # order1_fit = data["order1_fit"]
    # order2_fit = data["order2_fit"]
    # order3_fit = data["order3_fit"]
    # redchisq = data["redchisq"]
    redchisq0 = np.array([d["red_chisq0"] for d in data])
    redchisq1 = np.array([d["red_chisq1"] for d in data])
    redchisq2 = np.array([d["red_chisq2"] for d in data])
    redchisq3 = np.array([d["red_chisq3"] for d in data])
    time_choice = data[0]["time_choice"]
    delta_t = data[0]["delta_t"]

    print(data[7]["red_chisq0"])
    print(data[7]["red_chisq1"])
    print(data[7]["red_chisq2"])
    print(data[7]["red_chisq3"])

    print("\n\n", np.shape(order0_fit))

    all_data0 = {
        "lambdas0": lambdas,
        "lambdas1": lambdas,
        "lambdas2": lambdas,
        "lambdas3": lambdas,
        "order0_fit": order0_fit,
        "order1_fit": order1_fit,
        "order2_fit": order2_fit,
        "order3_fit": order3_fit,
        "redchisq0": redchisq0,
        "redchisq1": redchisq1,
        "redchisq2": redchisq2,
        "redchisq3": redchisq3,
        "time_choice": time_choice,
        "delta_t": delta_t,
    }
    plot_lmb_dep(all_data0, plotdir)

    # Filter out data points with a high reduced chi-squared value
    chisq_tol = 1.5  # 1.7
    print(np.where(redchisq0 <= chisq_tol))
    print(np.where(redchisq1 <= chisq_tol))
    print(np.where(redchisq2 <= chisq_tol))
    print(np.where(redchisq3 <= chisq_tol))

    print(redchisq0)
    print(redchisq1)
    print(redchisq2)
    print(redchisq3)

    order0_fit = order0_fit[np.where(redchisq0 <= chisq_tol)]
    lambdas0 = lambdas[np.where(redchisq0 <= chisq_tol)]
    order1_fit = order1_fit[np.where(redchisq1 <= chisq_tol)]
    lambdas1 = lambdas[np.where(redchisq1 <= chisq_tol)]
    order2_fit = order2_fit[np.where(redchisq2 <= chisq_tol)]
    lambdas2 = lambdas[np.where(redchisq2 <= chisq_tol)]
    order3_fit = order3_fit[np.where(redchisq3 <= chisq_tol)]
    lambdas3 = lambdas[np.where(redchisq3 <= chisq_tol)]


    all_data = {
        "lambdas0": lambdas0,
        "lambdas1": lambdas1,
        "lambdas2": lambdas2,
        "lambdas3": lambdas3,
        "order0_fit": order0_fit,
        "order1_fit": order1_fit,
        "order2_fit": order2_fit,
        "order3_fit": order3_fit,
        "redchisq0": redchisq0,
        "redchisq1": redchisq1,
        "redchisq2": redchisq2,
        "redchisq3": redchisq3,
        "time_choice": time_choice,
        "delta_t": delta_t,
    }

    # scaled_z0 = (redchisq[0] - redchisq[0].min()) / redchisq[0].ptp()
    # colors_0 = [[0., 0., 0., i] for i in scaled_z0]

    # plot_lmb_depR(all_data, plotdir)
    # plot_lmb_dep(all_data, plotdir)

    p0 = (1e-3, 0.7)
    fitlim = 30
    lmb_range = np.arange(config["lmb_init"], config["lmb_final"])

    # Fit to the lambda dependence at each order in lambda
    print("\n")
    try:
        if lmb_range[-1] >= len(lambdas0):
            lmb_range0 = np.arange(min(len(lambdas0) - 5, lmb_range[0]), len(lambdas0))
        else:
            lmb_range0 = lmb_range
        bootfit0, redchisq0, chisq0 = fit_lmb(
            order0_fit[lmb_range0],
            fitfunction5,
            lambdas0[lmb_range0],
            plotdir,
            p0=p0,
            order=1,
        )
        print("redchisq order 1:", redchisq0)
        print("chisq order 1:", chisq0)
        print("fit order 1:", np.average(bootfit0, axis=0), "\n")

        p0 = np.average(bootfit0, axis=0)

        print(lmb_range)
        print(len(lambdas1))
        if lmb_range[-1] >= len(lambdas1):
            lmb_range1 = np.arange(min(len(lambdas1) - 5, lmb_range[0]), len(lambdas1))
        else:
            lmb_range1 = lmb_range
        print(lmb_range1)
        bootfit1, redchisq1, chisq1 = fit_lmb(
            order1_fit[lmb_range1],
            fitfunction5,
            lambdas1[lmb_range1],
            plotdir,
            p0=p0,
            order=2,
        )
        print("redchisq order 2:", redchisq1)
        print("chisq order 2:", chisq1)
        print("fit order 2:", np.average(bootfit1, axis=0), "\n")

        print(lmb_range)
        print(len(lambdas2))
        if lmb_range[-1] >= len(lambdas2):
            lmb_range2 = np.arange(min(len(lambdas2) - 5, lmb_range[0]), len(lambdas2))
        else:
            lmb_range2 = lmb_range
        print(lmb_range2)

        bootfit2, redchisq2, chisq2 = fit_lmb(
            order2_fit[lmb_range2],
            fitfunction5,
            lambdas2[lmb_range2],
            plotdir,
            p0=p0,
            order=3,
        )
        print("redchisq order 3:", redchisq2)
        print("chisq order 3:", chisq2)
        print("fit order 3:", np.average(bootfit2, axis=0), "\n")

        if lmb_range[-1] >= len(lambdas3):
            lmb_range3 = np.arange(min(len(lambdas3) - 5, lmb_range[0]), len(lambdas3))
        else:
            lmb_range3 = lmb_range
        bootfit3, redchisq3, chisq3 = fit_lmb(
            order3_fit[lmb_range3],
            fitfunction5,
            lambdas3[lmb_range3],
            plotdir,
            p0=p0,
            order=4,
        )
        print("redchisq order 4:", redchisq3)
        print("chisq order 4:", chisq3)
        print("fit order 4:", np.average(bootfit3, axis=0))
        print("fit std order 4:", np.std(bootfit3, axis=0), "\n")

        # Fit with the expanded fit function
        p0 = (1e-3, 0.7, 0, 7)
        bootfit3_4, redchisq3_4, chisq3_4 = fit_lmb_4(
            order3_fit[lmb_range3],
            fitfunction_4,
            lambdas3[lmb_range3],
            plotdir,
            p0=p0,
        )
        print("\nNew fit function")
        print("redchisq order 4:", redchisq3_4)
        print("chisq order 4:", chisq3_4)
        print("fit order 4:", np.average(bootfit3_4, axis=0))
        print("fit std order 4:", np.std(bootfit3_4, axis=0), "\n")

        fit_data = {
            "lmb_range": lmb_range,
            "lmb_range0": lmb_range0,
            "lmb_range1": lmb_range1,
            "lmb_range2": lmb_range2,
            "lmb_range3": lmb_range3,
            "fitlim": fitlim,
            "bootfit0": bootfit0,
            "bootfit1": bootfit1,
            "bootfit2": bootfit2,
            "bootfit3": bootfit3,
            "bootfit3_4": bootfit3_4,
            "redchisq0": redchisq0,
            "redchisq1": redchisq1,
            "redchisq2": redchisq2,
            "redchisq3": redchisq3,
            "redchisq3_4": redchisq3_4,
        }
        with open(datadir / (f"matrix_element.pkl"), "wb") as file_out:
            pickle.dump(fit_data, file_out)

    except RuntimeError as e:
        print("====================\nFitting Failed\n", e, "\n====================")
        fit_data = None

    plot_lmb_dep3(all_data, plotdir, fit_data)
    plot_lmb_depR(all_data, plotdir, fit_data)
    # plot_lmb_dep(all_data, plotdir, fit_data)

    ### ----------------------------------------------------------------------
    # lmb_val = 0.06 #0.16
    # time_choice_range = np.arange(5, 10)
    # delta_t_range = np.arange(1, 4)
    # t_range = np.arange(4, 9)

    # with open(datadir / ("fit_data_time_choice"+str(time_choice_range[0])+"-"+str(time_choice_range[-1])+".pkl"), "rb") as file_in:
    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    lambdas = np.array([d["lambdas"] for d in data])
    # lambdas = data["lambdas0"]
    order0_fit = data["order0_fit"]
    order1_fit = data["order1_fit"]
    order2_fit = data["order2_fit"]
    order3_fit = data["order3_fit"]
    time_choice_range = data["time_choice"]
    delta_t_range = data["delta_t"]
    delta_t_choice = np.where(delta_t_range == config["delta_t"])[0][0]
    order3_evals = data["order3_evals"]
    order3_evecs = data["order3_evecs"]

    plt.figure(figsize=(9, 6))
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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

    plt.legend(fontsize="x-small")
    # plt.xlim(-0.01, 0.22)
    # plt.ylim(0, 0.06)
    # plt.ylim(0.03, 0.055)
    plt.xlabel("$t_{0}$")
    plt.ylabel("$\Delta E$")
    # plt.title(rf"$\Delta t = {delta_t_range[delta_t_choice]}, \lambda = {lmb_val}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    plt.close()

    # --------------------------------------------------------------------------------
    # plot the eigenvector values against t0
    evec1 = order3_evecs[:, delta_t_choice, :, 0, 0] ** 2
    evec2 = order3_evecs[:, delta_t_choice, :, 0, 1] ** 2
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        time_choice_range,
        np.average(evec1, axis=1),
        np.std(evec1, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        time_choice_range,
        np.average(evec2, axis=1),
        np.std(evec2, axis=1),
        fmt="x",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.legend(fontsize="x-small")
    # plt.xlim(-0.01, 0.22)
    # plt.ylim(0, 0.06)
    # plt.ylim(0.03, 0.055)
    plt.xlabel("$t_{0}$")
    plt.ylabel("eigenvector squared")
    # plt.title(rf"$\Delta t = {delta_t_range[delta_t_choice]}, \lambda = {lmb_val}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"time_choice_dep_l{lmb_val}_eigenvectors.pdf"))
    plt.close()

    # --------------------------------------------------------------------------------
    t0_choice = np.where(time_choice_range == config["time_choice"])[0][0]
    # t0_choice = config["time_choice"]

    plt.figure(figsize=(9, 6))
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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
    plt.errorbar(
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

    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(0.03, 0.055)
    plt.xlabel("$\Delta t$")
    plt.ylabel("$\Delta E$")
    # plt.title(rf"$t_{{0}} = {time_choice_range[t0_choice]}, \lambda = {lmb_val}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"delta_t_dep_l{lmb_val}.pdf"))
    plt.close()

    # --------------------------------------------------------------------------------
    # plot the eigenvector values against delta t
    evec1 = order3_evecs[t0_choice, :, :, 0, 0] ** 2
    evec2 = order3_evecs[t0_choice, :, :, 0, 1] ** 2
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        delta_t_range,
        np.average(evec1, axis=1),
        np.std(evec1, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    plt.errorbar(
        delta_t_range,
        np.average(evec2, axis=1),
        np.std(evec2, axis=1),
        fmt="x",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    plt.legend(fontsize="x-small")
    # plt.ylim(0, 0.2)
    # plt.ylim(0.03, 0.055)
    plt.xlabel("$\Delta t$")
    plt.ylabel("eigenvector squared")
    # plt.title(rf"$t_{{0}} = {time_choice_range[t0_choice]}, \lambda = {lmb_val}$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / (f"delta_t_dep_l{lmb_val}_eigenvector.pdf"))
    plt.close()


def main_loop():
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

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
    chisq_tol = 1.5  # 1.7
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

    p0 = (1e-3, 0.7)
    fitlim = 30
    # plot_lmb_dep2(all_data, plotdir, lmb_range)

    # Fit to the lambda dependence at each order in lambda
    print("\n")

    fit_data_list = []
    min_len = np.min([len(lambdas0), len(lambdas1), len(lambdas2), len(lambdas3)])
    for lmb_initial in np.arange(0, min_len):
        for lmb_final in np.arange(lmb_initial + 5, min_len):
            lmb_range = np.arange(lmb_initial, lmb_final)
            print(f"lmb_range = {lmb_range}")
            try:
                # bootfit0, redchisq0, chisq0 = fit_lmb(
                #     order0_fit[lmb_range], fitfunction5, lambdas0[lmb_range], plotdir, p0=p0, order=1
                # )
                # p0 = np.average(bootfit0, axis=0)

                # bootfit1, redchisq1, chisq1 = fit_lmb(
                #     order1_fit[lmb_range], fitfunction5, lambdas1[lmb_range], plotdir, p0=p0, order=2
                # )

                # bootfit2, redchisq2, chisq2 = fit_lmb(
                #     order2_fit[lmb_range], fitfunction5, lambdas2[lmb_range], plotdir, p0=p0, order=3
                # )

                bootfit3, redchisq3, chisq3 = fit_lmb(
                    order3_fit[lmb_range],
                    fitfunction5,
                    lambdas3[lmb_range],
                    plotdir,
                    p0=p0,
                    order=4,
                )
                print("redchisq order 4:", redchisq3)

                fit_data = {
                    "lmb_range": lmb_range,
                    "fitlim": fitlim,
                    # "bootfit0": bootfit0,
                    # "bootfit1": bootfit1,
                    # "bootfit2": bootfit2,
                    "bootfit3": bootfit3,
                    "lambdas3": lambdas3[lmb_range],
                    # "redchisq0": redchisq0,
                    # "redchisq1": redchisq1,
                    # "redchisq2": redchisq2,
                    "redchisq3": redchisq3,
                }
            except RuntimeError as e:
                print(
                    "====================\nFitting Failed\n",
                    e,
                    "\n====================",
                )
                fit_data = None
            fit_data_list.append(fit_data)

    with open(datadir / (f"matrix_elements_loop.pkl"), "wb") as file_out:
        pickle.dump(fit_data_list, file_out)

    # plot_lmb_dep(all_data, plotdir, fit_data)
    # plot_lmb_depR(all_data, plotdir, fit_data)


if __name__ == "__main__":
    main()
