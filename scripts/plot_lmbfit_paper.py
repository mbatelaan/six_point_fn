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
from gevpanalysis.util import save_plot

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

from gevpanalysis.params import params

from gevpanalysis.lambda_fitting import Fitfunction1
from gevpanalysis.lambda_fitting import Fitfunction2
from gevpanalysis.lambda_fitting import Fitfunction3
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

def plot_lmb_dep4(all_data, plotdir, fit_data=None, fitfunction=None, delta_E_fix=None):
    """Make a plot of the lambda dependence of the energy shift
    Where the plot uses colored bands to show the dependence
    """

    deltaEsquared = np.array(all_data["order3_fit"]) ** 2
    xdata = np.average(deltaEsquared, axis=1)
    xerr = np.std(deltaEsquared, axis=1)

    # plot_data = np.abs(all_data["order3_fit"])
    plot_data = all_data["order3_fit"]

    plt.figure(figsize=(6, 5))
    plt.fill_between(
        all_data["lambdas3"],
        np.average(plot_data, axis=1)
        - np.std(plot_data, axis=1),
        np.average(plot_data, axis=1)
        + np.std(plot_data, axis=1),
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        linewidth=0,
        alpha=0.3,
    )
    plt.legend(fontsize="x-small", loc="upper left")
    # plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    plt.xlim(0,0.05)
    plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)

    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$|\Delta E_{\lambda}|$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(plotdir / ("lambda_dep_bands_4_paper.pdf"), metadata=_metadata)

    if fit_data:
        print([i for i in fit_data])
        lmb_range = fit_data["lmb_range"]
        print(lmb_range)
        lmbfit_range = np.linspace(all_data["lambdas3"][lmb_range[0]], all_data["lambdas3"][lmb_range[-1]], 100)
        print(lmbfit_range)
        # plt.errorbar(
        #     all_data["lambdas3"][lmb_range]**2,
        #     xdata[lmb_range],
        #     xerr[lmb_range],
        #     capsize=4,
        #     elinewidth=1,
        #     color=_colors[3],
        #     fmt="s",
        #     markerfacecolor="none",
        #     label=r"$\mathcal{O}(\lambda^4)$",
        # )

        m_e_3 = err_brackets(
            np.average(fit_data["bootfit3"], axis=0)[0],
            np.std(fit_data["bootfit3"], axis=0)[0],
        )
        fitBS3 = np.sqrt(
            [fitfunction(all_data["lambdas3"], *bf, delta_E_fix[ibf]) for ibf, bf in enumerate(fit_data["bootfit3"])]
        )
        fitBS_lmbs = np.sqrt(
            [fitfunction( lmbfit_range , *bf, delta_E_fix[ibf]) for ibf, bf in enumerate(fit_data["bootfit3"])]
        )

        plt.plot(
            # all_data["lambdas3"],
            lmbfit_range,
            np.average(fitBS_lmbs, axis=0),
            color=_colors[4],
            linewidth=1,
            linestyle="--",
            alpha=0.9,
        )
        newline='\n'
        plt.fill_between(
            # all_data["lambdas3"],
            lmbfit_range,
            np.average(fitBS_lmbs, axis=0) - np.std(fitBS_lmbs, axis=0),
            np.average(fitBS_lmbs, axis=0) + np.std(fitBS_lmbs, axis=0),
            # label=rf"$\chi^2_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}${newline}$\textrm{{M.E.}}={m_e_3}$",
            # label=rf"$\textrm{{M.E.}}={m_e_3}$",
            label=rf"$\textrm{{fit}}$",
            # label=rf"$\chi^2_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$",
            color=_colors[4],
            linewidth=0,
            alpha=0.3,
        )
        plt.legend(fontsize="x-small", loc="upper left")
        plt.savefig(plotdir / ("lambda_dep_bands_fit_4_paper.pdf"), metadata=_metadata)
        plt.ylim(0, 0.15)
        plt.savefig(plotdir / ("lambda_dep_bands_fit_4_paper_ylim.pdf"), metadata=_metadata)
        plt.savefig(plotdir / ("lambda_dep_fit.pdf"), metadata=_metadata)

    # if fit_data:
    #     lmb_range = fit_data["lmb_range"]
    #     # lmb_range3 = fit_data["lmb_range3"]

    #     # plt.fill_between(
    #     #     np.array(
    #     #         [
    #     #             all_data["lambdas3"][lmb_range3[0]],
    #     #             all_data["lambdas3"][lmb_range3[-1]],
    #     #         ]
    #     #     ),
    #     #     np.array([-10, -10]),
    #     #     np.array([10, 10]),
    #     #     color=_colors[3],
    #     #     alpha=0.1,
    #     #     linewidth=0,
    #     # )
    #     m_e_3 = err_brackets(
    #         np.average(fit_data["bootfit3"], axis=0)[0],
    #         np.std(fit_data["bootfit3"], axis=0)[0],
    #     )

    #     fitBS3 = np.array(
    #         [fitfunction5(all_data["lambdas3"], *bf) for bf in fit_data["bootfit3"]]
    #     )

    #     plt.plot(
    #         all_data["lambdas3"],
    #         np.average(fitBS3, axis=0),
    #         label=rf"$\chi_{{\textrm{{dof}} }} = {fit_data['redchisq3']:2.3}$"
    #         + "\n"
    #         + rf"$\textrm{{M.E.}}={m_e_3}$",
    #         color=_colors[3],
    #         linewidth=1,
    #         linestyle="--",
    #         alpha=0.9,
    #     )

    #     plt.legend(fontsize="x-small", loc="upper left")
    #     plt.xlim(all_data["lambdas3"][0] * 0.9, all_data["lambdas3"][-1] * 1.1)
    #     plt.ylim(0, np.average(all_data["order3_fit"], axis=1)[-1] * 1.2)
    #     plt.tight_layout()
    #     plt.savefig(plotdir / ("lambda_dep_bands_fit_4.pdf"), metadata=_metadata)
    #     plt.ylim(0, 0.15)
    #     plt.savefig(plotdir / ("lambda_dep_bands_fit_ylim_4.pdf"), metadata=_metadata)

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

    with open(datadir / (f"matrix_element.pkl"), "rb") as file_in:
        chosen_fit = pickle.load(file_in)
    with open(datadir / (f"lambda_dep_plot_data.pkl"), "rb") as file_in:
        all_data = pickle.load(file_in)
        
    print(len(chosen_fit))
    print([i for i in chosen_fit])
    print(np.shape(chosen_fit["bootfit3"]))

    fitfunc3 = Fitfunction3()
    # delta_E_0 = np.array(
    #     [fit[f"order3_fit"][:, 1] for fit in fitlists[3]][0]
    # )
    delta_E_0 =  all_data["order3_fit"][0]
    print(np.shape(delta_E_0))


    plot_lmb_dep4(all_data, plotdir, chosen_fit, fitfunction=fitfunc3.eval, delta_E_fix = delta_E_0)


if __name__ == "__main__":
    main()
