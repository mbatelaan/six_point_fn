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


class Fitfunction5:
    """The function to fit to Delta E
    The function has two parameters, the energy gap at lambda=0 and the matrix element
    """

    def __init__(self):
        self.npar = 2
        self.label = r"fn1"
        self.initpar = np.array([1.0, 1.0])
        self.bounds = ([0, 0], [np.inf, np.inf])

    def eval(self, lmb, Delta_E, matrix_element):
        deltaE = np.sqrt(Delta_E**2 + 4 * lmb**2 * matrix_element**2)
        return deltaE


class Fitfunction1:
    """The function to fit to the square of Delta E
    The function has two parameters, the energy gap at lambda=0 and the matrix element
    """

    def __init__(self):
        self.npar = 2
        self.label = r"fn1"
        self.initpar = np.array([1.0, 1.0])
        self.bounds = ([0, 0], [np.inf, np.inf])

    def eval(self, lmb, Delta_E, matrix_element):
        deltaE = Delta_E**2 + 4 * lmb**2 * matrix_element**2
        return deltaE


class Fitfunction6:
    """A function to fit to Delta E
    The function has one parameter, the matrix element.
    The energy difference of the unperturbed correlators is a fixed parameter
    """

    def __init__(self):
        self.npar = 1
        self.label = r"fn3"
        self.initpar = np.array([1.0])
        self.bounds = ([0], [np.inf])

    def eval(self, lmb, matrix_element, delta_E_fix):
        deltaE = np.sqrt(delta_E_fix**2 + 4 * lmb**2 * matrix_element**2)
        return deltaE


class Fitfunction_order4:
    """The function to fit to the square of Delta E
    The function has three parameters, the energy gap at lambda=0, the matrix element and the coefficient of lambda^4 in the higher order term.
    """

    def __init__(self):
        self.npar = 3
        self.label = r"fn4"
        self.initpar = np.array([1.0, 1.0, 1.0])
        self.bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    def eval(self, lmb, Delta_E, A, B):
        """The fit function Ross proposed to capture the compton amplitude"""
        deltaE = np.sqrt(Delta_E**2 + 4 * lmb**2 * A**2 + lmb**4 * B**2)
        return deltaE


def fit_lmb(ydata, function, lambdas, p0=None, bounds=None):
    """Fit the lambda dependence

    ydata is an array with the lambda values on the first index and bootstraps on the second index
    lambdas is an array of values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    # bounds = ([0, 0], [np.inf, np.inf])
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
    print(f"dof = {dof}")
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


def fit_lambda_dep(fitlist, order, lmb_range, fitfunction, p0, bounds):
    """Fit the lambda dependence of the energy shift"""
    # p0 = (1e-3, 0.7)
    fit_data = np.array([fit[f"order{order}_fit"][:, 1] for fit in fitlist])
    lambdas = np.array([fit[f"lambdas"] for fit in fitlist])

    # Check if we haven't excluded some of the chosen fit range
    if lmb_range[-1] >= len(lambdas):
        lmb_range = np.arange(min(len(lambdas) - 5, lmb_range[0]), len(lambdas))
    else:
        lmb_range = lmb_range
    bootfit, redchisq_fit, chisq_fit = fit_lmb(
        fit_data[lmb_range],
        fitfunction,
        lambdas[lmb_range],
        p0=p0,
        bounds=bounds,
    )
    print(f"redchisq order {order}:", redchisq_fit)
    print(f"chisq order {order}:", chisq_fit)
    print(f"fit order {order}:", np.average(bootfit, axis=0), "\n")
    return lmb_range, bootfit, redchisq_fit, chisq_fit


def lambdafit_3pt(lambdas3, fitlists, datadir, fitfunction):
    p0 = fitfunction.initpar
    bounds = fitfunction.bounds
    fit_data_list = []
    min_len = len(lambdas3)
    for lmb_initial in np.arange(0, 4):
        for lmb_step in np.arange(1, min_len / 2 - 1):
            lmb_range = np.array(
                [
                    lmb_initial,
                    int(lmb_initial + lmb_step),
                    int(lmb_initial + lmb_step * 2),
                ]
            )
            if lmb_range[-1] >= min_len:
                continue
            print(f"lmb_range = {lmb_range}")
            try:
                if lmb_range[-1] < len(lambdas3):
                    order = 3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range, fitfunction.eval, p0, bounds
                    )
                fit_data = {
                    "lmb_range": lmb_range,
                    "bootfit3": bootfit,
                    # "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "lambdas3": lambdas3[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print(
                    "====================\nFitting Failed\n",
                    e,
                    "\n====================",
                )
                fit_data = None

    with open(
        datadir / (f"matrix_elements_loop_3pts_{fitfunction.label}.pkl"), "wb"
    ) as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def lambdafit_4pt(lambdas3, fitlists, datadir, fitfunction):
    p0 = fitfunction.initpar
    bounds = fitfunction.bounds
    fit_data_list = []
    min_len = len(lambdas3)
    for lmb_initial in np.arange(0, 4):
        for lmb_step in np.arange(1, min_len / 3):
            lmb_range = np.array(
                [
                    lmb_initial,
                    int(lmb_initial + lmb_step),
                    int(lmb_initial + lmb_step * 2),
                    int(lmb_initial + lmb_step * 3),
                ]
            )
            if lmb_range[-1] >= min_len:
                continue
            print(f"lmb_range = {lmb_range}")
            try:
                if lmb_range[-1] < len(lambdas3):
                    order = 3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range, fitfunction.eval, p0, bounds
                    )
                fit_data = {
                    "lmb_range": lmb_range,
                    "bootfit3": bootfit,
                    "lambdas3": lambdas3[lmb_range],
                    # "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print(
                    "====================\nFitting Failed\n",
                    e,
                    "\n====================",
                )
                fit_data = None

    with open(
        datadir / (f"matrix_elements_loop_4pts_{fitfunction.label}.pkl"), "wb"
    ) as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def lambdafit_3pt_squared(lambdas3, fitlists, datadir, fitfunction):
    p0 = fitfunction.initpar
    bounds = fitfunction.bounds
    fit_data_list = []
    min_len = len(lambdas3)
    for lmb_initial in np.arange(0, 4):
        for lmb_step in np.arange(1, min_len / 2 - 1):
            lmb_range = np.array(
                [
                    lmb_initial,
                    int(lmb_initial + lmb_step),
                    int(lmb_initial + lmb_step * 2),
                ]
            )
            if lmb_range[-1] >= min_len:
                continue
            print(f"lmb_range = {lmb_range}")
            try:
                if lmb_range[-1] < len(lambdas3):
                    order = 3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range, fitfunction.eval, p0, bounds
                    )
                fit_data = {
                    "lmb_range": lmb_range,
                    "bootfit3": bootfit,
                    # "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "lambdas3": lambdas3[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print(
                    "====================\nFitting Failed\n",
                    e,
                    "\n====================",
                )
                fit_data = None

    with open(
        datadir / (f"matrix_elements_loop_3pts_{fitfunction.label}.pkl"), "wb"
    ) as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def lambdafit_allpt(lambdas3, fitlists, datadir, fitfunction):
    p0 = fitfunction.initpar
    bounds = fitfunction.bounds
    fit_data_list = []
    min_len = len(lambdas3)
    # print('len(lambdas3) = ', min_len)
    for lmb_initial in np.arange(0, min_len):
        for lmb_final in np.arange(lmb_initial + len(p0) + 1, min_len):
            lmb_range = np.arange(lmb_initial, lmb_final)
            print(f"lmb_range = {lmb_range}")
            try:
                if lmb_range[-1] < len(lambdas3):
                    order = 3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range, fitfunction.eval, p0, bounds
                    )

                fit_data = {
                    "lmb_range": lmb_range,
                    "bootfit3": bootfit,
                    "lambdas3": lambdas3[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print(
                    "====================\nFitting Failed\n",
                    e,
                    "\n====================",
                )
                fit_data = None

    with open(
        datadir / (f"matrix_elements_loop_{fitfunction.label}.pkl"), "wb"
    ) as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list
