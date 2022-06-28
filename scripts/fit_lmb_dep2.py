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

from gevpanalysis.lambda_fitting import Fitfunction1
from gevpanalysis.lambda_fitting import Fitfunction2
from gevpanalysis.lambda_fitting import Fitfunction6
from gevpanalysis.lambda_fitting import Fitfunction_order4

# from gevpanalysis.lambda_fitting import fit_lmb
from gevpanalysis.lambda_fitting import fit_lambda_dep


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


def plot_lmb_dep4_sq(all_data, plotdir, fit_data=None, fitfunction=None):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    deltaEsquared = np.array(all_data["order3_fit"]) ** 2
    xdata = np.average(deltaEsquared, axis=1)
    xerr = np.std(deltaEsquared, axis=1)

    plt.figure(figsize=(9, 6))
    plt.fill_between(
        all_data["lambdas3"],
        xdata - xerr,
        xdata + xerr,
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )

    plt.legend(fontsize="x-small", loc="upper left")
    plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.ylim(0, xdata[-1] * 1.2)

    plt.xlabel("$\lambda$")
    plt.ylabel("$\Delta E$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands_4_sq.pdf"), metadata=_metadata)

    if fit_data:
        lmb_range = fit_data["lmb_range"]
        plt.errorbar(
            all_data["lambdas3"][lmb_range],
            xdata[lmb_range],
            xerr[lmb_range],
            capsize=4,
            elinewidth=1,
            color=_colors[2],
            fmt="s",
            markerfacecolor="none",
        )

        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[1],
            np.std(fit_data["bootfit3"], axis=0)[1],
        )

        fitBS3 = np.array(
            [fitfunction(all_data["lambdas3"], *bf) for bf in fit_data["bootfit3"]]
        )

        print(all_data["lambdas3"][lmb_range])
        fitBS3_ = np.array(
            [
                fitfunction(all_data["lambdas3"][lmb_range], *bf)
                for bf in fit_data["bootfit3"]
            ]
        )
        # plt.errorbar(all_data["lambdas3"][lmb_range], np.average(fitBS3_, axis=0),np.std(fitBS3_, axis=0),
        #              capsize=4,
        #              elinewidth=1,
        #              color=_colors[2],
        #              fmt="s",
        #              markerfacecolor="none",
        # )

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
        plt.fill_between(
            all_data["lambdas3"],
            np.average(fitBS3, axis=0) - np.std(fitBS3, axis=0),
            np.average(fitBS3, axis=0) + np.std(fitBS3, axis=0),
            label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
            + "\n"
            + rf"$\textrm{{M.E.}}={m_e_3}$",
            color=_colors[4],
            linewidth=1,
            linestyle="--",
            alpha=0.1,
        )

        plt.legend(fontsize="x-small", loc="upper left")
        plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
        plt.ylim(0, xdata[-1] * 1.2)
        plt.tight_layout()
        plt.savefig(plotdir / ("lambda_dep_bands_fit_4_sq.pdf"), metadata=_metadata)
        plt.ylim(0, 0.007)
        plt.savefig(
            plotdir / ("lambda_dep_bands_fit_ylim_4_sq.pdf"), metadata=_metadata
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

    fitfunc2 = Fitfunction2()

    with open(datadir / (f"matrix_elements_loop_3pts_sq_fn2.pkl"), "rb") as file_in:
        data_3pts_sq = pickle.load(file_in)
    chisq_values = np.array([elem["redchisq3"] for elem in data_3pts_sq])
    print("\n3 points")
    for i, elem in enumerate(data_3pts_sq):
        print(elem["lmb_range"], "\t\t", elem["redchisq3"])
    chosen_fit = [
        i for i in data_3pts_sq if i["lmb_range"][0] == 0 and i["lmb_range"][-1] == 12
    ][0]
    print(chosen_fit["redchisq3"])
    print(chosen_fit["lmb_range"])

    # Plot the fit to the lambda-dependence
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
    plot_lmb_dep4_sq(all_data, plotdir, fit_data=chosen_fit, fitfunction=fitfunc2.eval)


if __name__ == "__main__":
    main()