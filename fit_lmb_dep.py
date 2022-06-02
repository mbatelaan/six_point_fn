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
    """The fit function Ross proposed to capture the compton amplitude"""
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
        plotdir,
        p0=p0,
        order=order,
    )
    print("redchisq order 1:", redchisq_fit)
    print("chisq order 1:", chisq)
    print("fit order 1:", np.average(bootfit, axis=0), "\n")
    return lmb_range, bootfit, redchisq_fit, chisq_fit


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
    indices0 = np.where(redchisq0 <= chisq_tol)
    indices1 = np.where(redchisq1 <= chisq_tol)
    indices2 = np.where(redchisq2 <= chisq_tol)
    indices3 = np.where(redchisq3 <= chisq_tol)
    fitlist0 = data[indices0]
    fitlist1 = data[indices1]
    fitlist2 = data[indices2]
    fitlist3 = data[indices3]
    fitlists = [fitlist0, fitlist1, fitlist2, fitlist3]

    lmb_range = np.arange(config["lmb_init"], config["lmb_final"])

    # Fit to the lambda dependence for each order
    fit_data = {"lmb_range": lmb_range}
    for order in np.arange(3):
        lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
            fitlists[order], order, lmb_range
        )
        fit_data[f"lmb_range{order}"] = lmb_range
        fit_data[f"boofit{order}"] = bootfit
        fit_data[f"redchisq{order}"] = redchisq_fit

    with open(datadir / (f"matrix_element.pkl"), "wb") as file_out:
        pickle.dump(fit_data, file_out)

    exit()

    # ==================================================

    # Fit to the lambda dependence at each order in lambda
    print("\n")
    try:
        # Check if we haven't excluded some of the chosen fit range
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

        # Fit with the expanded fit function with a lambda^4 term to capture the Compton amplitude
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

    exit()

    # with open(datadir / ("fit_data_time_choice"+str(time_choice_range[0])+"-"+str(time_choice_range[-1])+".pkl"), "rb") as file_in:
    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    # lambdas = np.array([d["lambdas"] for d in data])
    # lambdas = data["lambdas0"]
    order0_fit = np.array([d["order0_fit_bs"][:, 1] for d in data])
    order1_fit = np.array([d["order1_fit_bs"][:, 1] for d in data])
    order2_fit = np.array([d["order2_fit_bs"][:, 1] for d in data])
    order3_fit = np.array([d["order3_fit_bs"][:, 1] for d in data])
    # order0_fit = data["order0_fit"]
    # order1_fit = data["order1_fit"]
    # order2_fit = data["order2_fit"]
    # order3_fit = data["order3_fit"]
    time_choice_range = np.array([d["t_0"] for d in data])
    # time_choice_range = data["time_choice"]
    # delta_t_range = data["delta_t"]
    delta_t_range = np.array([d["delta_t"] for d in data])
    delta_t_choice = np.where(delta_t_range == config["delta_t"])[0][0]
    # order3_evals = data["order3_evals"]
    # order3_evecs = data["order3_evecs"]
    # order3_evals = data["order3_evals"]
    # order3_evecs = data["order3_evecs"]

    energy_shifts = np.array([fit["order3_fit"][:, 1] for fit in fitlist])[indices]

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


if __name__ == "__main__":
    main()
