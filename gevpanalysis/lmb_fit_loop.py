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
    deltaE = np.sqrt(Delta_E ** 2 + 4 * lmb ** 2 * matrix_element ** 2)
    return deltaE


# def fit_lmb(ydata, function, lambdas, plotdir, p0=None, order=1, svd_inv = True):
#     """Fit the lambda dependence

#     data is a correlator with tht bootstraps on the first index and the time on the second
#     lambdas is an array of time values to fit over
#     the function will return an array of fit parameters for each bootstrap
#     """

#     # bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
#     bounds = ([0, 0], [np.inf, np.inf])
#     ydata = ydata.T
#     data_set = ydata
#     ydata_avg = np.average(data_set, axis=0)

#     covmat = np.cov(data_set.T)
#     diag = np.diagonal(covmat)

#     if svd_inv:
#         # Calculate the eigenvalues of the covariance matrix
#         eval_left, evec_left = np.linalg.eig(covmat)
#         sorted_evals = np.sort(eval_left)[::-1]
#         svd = 5 #How many singular values do we want to keep for the inversion
#         rcond = (sorted_evals[svd-1] - sorted_evals[svd+1]) / 2 / sorted_evals[0]
#         covmat_inverse = np.linalg.pinv(covmat, rcond=rcond)
#         dof = svd-2
#     else:
#         covmat_inverse = linalg.pinv(covmat)
#         dof = len(lambdas)
        
#     diag_sigma = np.diag(np.std(data_set, axis=0) ** 2)
#     popt_avg, pcov_avg = curve_fit(
#         function,
#         lambdas,
#         ydata_avg,
#         sigma=diag_sigma,
#         p0=p0,
#         maxfev=4000,
#         bounds=bounds,
#     )
#     chisq = ff.chisqfn2(popt_avg, function, lambdas, ydata_avg, covmat_inverse)
#     p0 = popt_avg
#     redchisq = chisq / dof
#     bootfit = []
#     for iboot, values in enumerate(ydata):
#         # print(iboot)
#         popt, pcov = curve_fit(
#             function,
#             lambdas,
#             values,
#             sigma=diag_sigma,
#             # maxfev=4000,
#             p0=p0,
#             bounds=bounds,
#         )  # , p0=popt_avg)
#         # print(popt)
#         bootfit.append(popt)
#     bootfit = np.array(bootfit)
#     print("bootfit", np.average(bootfit, axis=0))
#     return bootfit, redchisq, chisq

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

def lambdafit_3pt(lambdas3, fitlists, datadir):
    p0 = (1e-3, 0.7)
    fitlim = 30
    fit_data_list = []
    min_len = len(lambdas3)
    for lmb_initial in np.arange(0,4):
        for lmb_step in np.arange(1,min_len/2-1):
            lmb_range = np.array([lmb_initial, int(lmb_initial + lmb_step),int(lmb_initial + lmb_step*2)])
            if lmb_range[-1]>=min_len:
                continue
            print(f'lmb_range = {lmb_range}')
            try:
                if lmb_range[-1] < len(lambdas3):
                    order=3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range
                    )
                fit_data = {
                    "lmb_range": lmb_range,
                    "fitlim": fitlim,
                    "bootfit3": bootfit,
                    # "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "lambdas3": lambdas3[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print("====================\nFitting Failed\n", e, "\n====================")
                fit_data = None
                
    with open(datadir / (f"matrix_elements_loop_3pts.pkl"), "wb") as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def lambdafit_4pt(lambdas3, fitlists, datadir):
    p0 = (1e-3, 0.7)
    fitlim = 30
    fit_data_list = []
    min_len = len(lambdas3)
    for lmb_initial in np.arange(0,4):
        for lmb_step in np.arange(1,min_len/3):
            lmb_range = np.array([lmb_initial, int(lmb_initial + lmb_step), int(lmb_initial + lmb_step*2), int(lmb_initial + lmb_step*3)])
            if lmb_range[-1]>=min_len:
                continue
            print(f'lmb_range = {lmb_range}')
            try:
                if lmb_range[-1] < len(lambdas3):
                    order=3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range
                    )
                fit_data = {
                    "lmb_range": lmb_range,
                    "fitlim": fitlim,
                    "bootfit3": bootfit,
                    "lambdas3": lambdas3[lmb_range],
                    # "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print("====================\nFitting Failed\n", e, "\n====================")
                fit_data = None
                
    with open(datadir / (f"matrix_elements_loop_4pts.pkl"), "wb") as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def lambdafit_allpt(lambdas3, fitlists, datadir):
    p0 = (1e-3, 0.7)
    fitlim = 30
    fit_data_list = []
    min_len = len(lambdas3)
    # print('len(lambdas3) = ', min_len)
    for lmb_initial in np.arange(0,min_len):
        for lmb_final in np.arange(lmb_initial+3,min_len):
            lmb_range = np.arange(lmb_initial, lmb_final)
            print(f'lmb_range = {lmb_range}')
            try:
                if lmb_range[-1] < len(lambdas3):
                    order=3
                    lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
                        fitlists[order], order, lmb_range
                    )

                fit_data = {
                    "lmb_range": lmb_range,
                    "fitlim": fitlim,
                    "bootfit3": bootfit,
                    "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
                    "chisq3": chisq_fit,
                    "redchisq3": redchisq_fit,
                }
                fit_data_list.append(fit_data)
            except RuntimeError as e:
                print("====================\nFitting Failed\n", e, "\n====================")
                fit_data = None
                
    with open(datadir / (f"matrix_elements_loop.pkl"), "wb") as file_out:
        pickle.dump(fit_data_list, file_out)
    return fit_data_list


def main_loop():
    """ Fit to the lambda dependence of the energy shift and loop over the fit windows """
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

    lambdafit_3pt(lambdas3, fitlists, datadir)
    lambdafit_4pt(lambdas3, fitlists, datadir)
    lambdafit_allpt(lambdas3, fitlists, datadir)

    # p0 = (1e-3, 0.7)
    # fitlim = 30
    # fit_data_list = []
    # min_len = len(lambdas3)
    # # print('len(lambdas3) = ', min_len)
    # for lmb_initial in np.arange(0,min_len):
    #     for lmb_final in np.arange(lmb_initial+3,min_len):
    #         lmb_range = np.arange(lmb_initial, lmb_final)
    #         print(f'lmb_range = {lmb_range}')
    #         try:
    #             if lmb_range[-1] < len(lambdas3):
    #                 order=3
    #                 lmb_range, bootfit, redchisq_fit, chisq_fit = fit_lambda_dep(
    #                     fitlists[order], order, lmb_range
    #                 )

    #             fit_data = {
    #                 "lmb_range": lmb_range,
    #                 "fitlim": fitlim,
    #                 "bootfit3": bootfit,
    #                 "lambdas3": np.array([fit[f"lambdas"] for fit in fitlist3])[lmb_range],
    #                 "chisq3": chisq_fit,
    #                 "redchisq3": redchisq_fit,
    #             }
    #             fit_data_list.append(fit_data)
    #         except RuntimeError as e:
    #             print("====================\nFitting Failed\n", e, "\n====================")
    #             fit_data = None
                
    # with open(datadir / (f"matrix_elements_loop.pkl"), "wb") as file_out:
    #     pickle.dump(fit_data_list, file_out)

if __name__ == "__main__":
    main_loop()
