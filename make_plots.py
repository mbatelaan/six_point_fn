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



if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    nboot = 200
    nbin = 1

    # Read in the directory data from the yaml file
    config_file = "data_dir.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # TODO: Set up a defaults.yaml file for when there is no input file
    pickledir = Path(config["pickle_dir1"])
    pickledir2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    print(datadir / ("lambda_dep.pkl"))
    with open(datadir / ("lambda_dep.pkl"), "rb") as file_in:
        data = pickle.load(file_in)
    [lambdas, order0_fit, order1_fit, order2_fit, order3_fit] = data
    print(lambdas)
    # print(np.array(order0_fit))
    print(np.shape(order0_fit))
    print(np.shape(order0_fit[0]))
    print(np.shape(lambdas))

    order0_fit = np.einsum("ij,i->ij", order0_fit,lambdas**(-1))
    order1_fit = np.einsum("ij,i->ij", order1_fit,lambdas**(-1))
    order2_fit = np.einsum("ij,i->ij", order2_fit,lambdas**(-1))
    order3_fit = np.einsum("ij,i->ij", order3_fit,lambdas**(-1))

    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        lambdas,
        np.average(order0_fit, axis=1),
        np.std(order0_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.001,
        np.average(order1_fit, axis=1),
        np.std(order1_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.002,
        np.average(order2_fit, axis=1),
        np.std(order2_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        lambdas+0.003,
        np.average(order3_fit, axis=1),
        np.std(order3_fit, axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.legend(fontsize="x-small")
    pypl.xlim(-0.01, 0.22)
    # pypl.ylim(0, 0.2)
    pypl.xlabel("$\lambda$")
    pypl.ylabel("$\Delta E / \lambda$")
    # pypl.plot(lambdas, np.average(order0_fit, axis=1))
    # pypl.plot(lambdas, np.average(order1_fit, axis=1))
    # pypl.plot(lambdas, np.average(order2_fit, axis=1))
    # pypl.plot(lambdas, np.average(order3_fit, axis=1))
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / ("Energy_over_lambda.pdf"))
    # pypl.show()
    

