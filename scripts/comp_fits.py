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


def plot_fits(data_, name, plotdir):
    tmin0_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 0 
    ]
    tmin0_tmax = [i["lmb_range"][-1] for i in tmin0_fits_]
    tmin0_redchisq = [i["redchisq3"] for i in tmin0_fits_]
    tmin1_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 1 
    ]
    tmin1_tmax = [i["lmb_range"][-1] for i in tmin1_fits_]
    tmin1_redchisq = [i["redchisq3"] for i in tmin1_fits_]
    tmin2_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 2 
    ]
    tmin2_tmax = [i["lmb_range"][-1] for i in tmin2_fits_]
    tmin2_redchisq = [i["redchisq3"] for i in tmin2_fits_]

    plt.figure(figsize=(5, 4))
    plt.plot(tmin0_tmax, tmin0_redchisq, color=_colors[0], label="tmin=0")
    plt.plot(tmin1_tmax, tmin1_redchisq, color=_colors[1], label="tmin=1")
    plt.plot(tmin2_tmax, tmin2_redchisq, color=_colors[2], label="tmin=2")
    plt.tight_layout()
    plt.legend(fontsize="small")
    plt.xticks(np.arange(2,15))
    plt.xlabel(r"$t_{\textrm{max}}$")
    plt.ylabel(r"$\chi^2_{\textrm{red.}}$")
    plt.grid(True, alpha=0.3)
    plt.ylim(0,5)
    plt.savefig(plotdir / (f"lambda_fit_{name}.pdf"))
    plt.close()

    tmin0_me = [i["bootfit3"][:, 1] for i in tmin0_fits_]
    tmin1_me = [i["bootfit3"][:, 1] for i in tmin1_fits_]
    tmin2_me = [i["bootfit3"][:, 1] for i in tmin2_fits_]
    plt.figure(figsize=(5, 4))
    plt.errorbar(tmin0_tmax, np.average(tmin0_me, axis=1), np.std(tmin0_me, axis=1), color=_colors[0], label="tmin=0", capsize=4, elinewidth=1, fmt=_markers[0])
    plt.errorbar(tmin1_tmax, np.average(tmin1_me, axis=1), np.std(tmin1_me, axis=1), color=_colors[1], label="tmin=1", capsize=4, elinewidth=1, fmt=_markers[1])
    plt.errorbar(tmin2_tmax, np.average(tmin2_me, axis=1), np.std(tmin2_me, axis=1), color=_colors[2], label="tmin=2", capsize=4, elinewidth=1, fmt=_markers[2])
    plt.tight_layout()
    plt.legend(fontsize="small")
    plt.xticks(np.arange(2,15))
    plt.xlabel(r"$t_{\textrm{max}}$")
    plt.ylabel(r"Matrix element")
    plt.grid(True, alpha=0.3)
    # plt.ylim(0,5)
    plt.savefig(plotdir / (f"lambda_fit_"+name+"_me.pdf"))
    plt.close()

    return


def plot_fits_comb(data_, name, plotdir):
    tmin0_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 0 
    ]
    tmin1_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 1 
    ]
    tmin2_fits_ = [
        i
        for i in data_
        if i["lmb_range"][0] == 2 
    ]
    tmin0_tmax = [i["lmb_range"][-1] for i in tmin0_fits_]
    tmin1_tmax = [i["lmb_range"][-1] for i in tmin1_fits_]
    tmin2_tmax = [i["lmb_range"][-1] for i in tmin2_fits_]
    tmin0_redchisq = [i["redchisq3"] for i in tmin0_fits_]
    tmin1_redchisq = [i["redchisq3"] for i in tmin1_fits_]
    tmin2_redchisq = [i["redchisq3"] for i in tmin2_fits_]
    tmin0_me = [i["bootfit3"][:, 1] for i in tmin0_fits_]
    tmin1_me = [i["bootfit3"][:, 1] for i in tmin1_fits_]
    tmin2_me = [i["bootfit3"][:, 1] for i in tmin2_fits_]

    print(f"tmin0_tmax = {tmin0_tmax}")
    print(f"tmin1_tmax = {tmin1_tmax}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(tmin0_tmax, tmin0_redchisq, color=_colors[0], label=r"$t_{\textrm{min}}=0$")
    ax1.plot(tmin1_tmax, tmin1_redchisq, color=_colors[1], label=r"$t_{\textrm{min}}=1$")
    ax1.plot(tmin2_tmax, tmin2_redchisq, color=_colors[2], label=r"$t_{\textrm{min}}=2$")
    ax1.set_ylabel(r"$\chi^2_{\textrm{red.}}$")
    ax1.set_ylim(0,5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(2,15))
    ax1.set_xlabel(r"$t_{\textrm{max}}$")

    print(f"tmin0_tmax = {tmin0_tmax}")
    print(f"tmin0_me = {np.average(tmin0_me, axis=1)}")

    ax2 = ax1.twinx()
    ax2.errorbar(tmin0_tmax, np.average(tmin0_me, axis=1), np.std(tmin0_me, axis=1), color=_colors[0], label=r"$t_{\textrm{min}}=0$", capsize=4, elinewidth=1, fmt=_markers[0])
    ax2.errorbar(tmin1_tmax, np.average(tmin1_me, axis=1), np.std(tmin1_me, axis=1), color=_colors[1], label=r"$t_{\textrm{min}}=1$", capsize=4, elinewidth=1, fmt=_markers[1])
    ax2.errorbar(tmin2_tmax, np.average(tmin2_me, axis=1), np.std(tmin2_me, axis=1), color=_colors[2], label=r"$t_{\textrm{min}}=2$", capsize=4, elinewidth=1, fmt=_markers[2])
    ax2.set_ylabel(r"Matrix element")

    ax1.legend(fontsize="small")
    ax2.legend(fontsize="small")

    plt.savefig(plotdir / (f"lambda_fit_"+name+"_comb.pdf"))
    plt.close()

    return


def main():
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

    with open(datadir / (f"matrix_elements_loop_3pts.pkl"), "rb") as file_in:
        data_3pts = pickle.load(file_in)
    with open(datadir / (f"matrix_elements_loop_4pts.pkl"), "rb") as file_in:
        data_4pts = pickle.load(file_in)

    print("\n3 points")
    for i, elem in enumerate(data_3pts):
        print(elem["lmb_range"], "\t\t", elem["redchisq3"])
    print("\n4 points")
    for i, elem in enumerate(data_4pts):
        print(elem["lmb_range"], "\t\t", elem["redchisq3"])


    # plot_fits(data_3pts, "3points", plotdir)
    # plot_fits(data_4pts, "4points", plotdir)
    plot_fits_comb(data_3pts, "3points", plotdir)
    plot_fits_comb(data_4pts, "4points", plotdir)

if __name__ == "__main__":
    main()
