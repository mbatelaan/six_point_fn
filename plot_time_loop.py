import numpy as np
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

from common import read_pickle
from common import fit_value
from common import fit_value3
from common import read_correlators
from common import read_correlators2
from common import make_matrices
from common import gevp

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


def main():
    """Plot the chi-squared values and weights of a range of time fitting windows

    Read the data from a pickle file and plot it as a color plot on a matrix
    """

    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)  # Get the parameters for this lattice

    # Read in the directory data from the yaml file if one is given
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir_theta2.yaml"
    print("Reading directories from: ", config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    pickledir_k1 = Path(config["pickle_dir1"])
    pickledir_k2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    # with open(datadir / (f"time_window_loop_lambda.pkl"), "rb") as file_in:
    with open(datadir / (f"time_window_loop_nucl.pkl"), "rb") as file_in:
        fitlist_nucl = pickle.load(file_in)

    # with open(datadir / (f"time_window_loop_sigma.pkl"), "rb") as file_in:
    with open( "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma.pkl", "rb") as file_in:
        fitlist_sigma = pickle.load(file_in)

    with open(datadir / (f"time_window_loop_lambda_small.pkl"), "rb") as file_in:
        fitlist_small = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_lambda_large.pkl"), "rb") as file_in:
        fitlist_large = pickle.load(file_in)

    weights_nucl = np.array([i["weight"] for i in fitlist_nucl])
    high_weight_nucl = np.argmax(weights_nucl)
    # print(fitlist_nucl[high_weight_nucl]["redchisq"])
    nucl_t_range = np.arange(
        fitlist_nucl[high_weight_nucl]["x"][0],
        fitlist_nucl[high_weight_nucl]["x"][-1]+1,
    )
    print(f"nucl_t_range = {nucl_t_range}")
    print("redchisq", fitlist_nucl[high_weight_nucl]["redchisq"])
    print("maxweight",max(weights_nucl))
    
    weights_sigma = np.array([i["weight"] for i in fitlist_sigma])
    high_weight_sigma = np.argmax(weights_sigma)
    sigma_t_range = np.arange(
        fitlist_sigma[high_weight_sigma]["x"][0],
        fitlist_sigma[high_weight_sigma]["x"][-1]+1,
    )
    print(f"sigma_t_range = {sigma_t_range}")
    print(fitlist_sigma[high_weight_sigma]["redchisq"])
    print("maxweight",max(weights_sigma))

    weights_small = np.array([i["weight"] for i in fitlist_small])
    high_weight_small = np.argmax(weights_small)
    weights_large = np.array([i["weight"] for i in fitlist_large])
    high_weight_large = np.argmax(weights_large)
    ratio_t_range = np.arange(
        min(
            fitlist_small[high_weight_small]["x"][0],
            fitlist_large[high_weight_large]["x"][0],
        ),
        fitlist_large[high_weight_large]["x"][-1]+1,
    )
    print(f"ratio_t_range = {ratio_t_range}")
    print(fitlist_small[high_weight_small]["redchisq"])
    print("maxweight",max(weights_small))
    print(fitlist_large[high_weight_large]["redchisq"])
    print("maxweight",max(weights_large))

    print(f"small: {fitlist_small[high_weight_small]['x']}")
    print(f"large: {fitlist_large[high_weight_large]['x']}")
    exit()
    
    weights = np.array([i["weight"] for i in fitlist])
    x_coord = np.array([i["x"][0] for i in fitlist])
    y_coord = np.array([i["x"][-1] for i in fitlist])

    # print(weights)
    print(max(weights))
    argument_w = np.argmax(weights)
    print(argument_w)

    print(fitlist[argument_w]["x"][0])
    print(fitlist[argument_w]["x"][-1])
    print(fitlist[argument_w]["redchisq"])

    # Find the unique values of tmin and tmax to make a grid showing the reduced chi-squared values.
    unique_x = np.unique(x_coord)
    unique_y = np.unique(y_coord)
    min_x = np.min(x_coord)
    min_y = np.min(y_coord)
    plot_x = np.append(unique_x, unique_x[-1]+1)
    plot_y = np.append(unique_y, unique_y[-1]+1)

    print(f"\n\n{unique_x}")
    print(f"{min_x}")
    print(f"{unique_y}")
    print(f"{min_y}")

    matrix = np.zeros((len(unique_x), len(unique_y)))
    for i, x in enumerate(x_coord):
        matrix[x - min_x, y_coord[i] - min_y] = fitlist[i]["redchisq"]

    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu", vmin=0.0, vmax=2)
    # mat = plt.pcolormesh(unique_x, unique_y, matrix.T, cmap = 'GnBu', norm=colors.LogNorm(vmin=0.5, vmax=np.max(matrix)))
    plt.colorbar(mat, label=r"$\chi^2_{\textrm{dof}}$")
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$t_{\textrm{max}}$")
    plt.savefig(plotdir / (f"chisq_matrix_time.pdf"))
    plt.close()

    matrix = np.zeros((len(unique_x), len(unique_y)))
    for i, x in enumerate(x_coord):
        matrix[x - min_x, y_coord[i] - min_y] = fitlist[i]["weight"]

    print("len: ", np.shape(matrix))

    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(
        plot_x, plot_y, matrix.T, cmap="GnBu", vmin=0.0, vmax=np.max(matrix)
    )
    # mat = plt.pcolormesh(
    #     matrix.T, cmap="GnBu", vmin=0.0, vmax=np.max(matrix)
    # )
    # mat = plt.pcolormesh(unique_x, unique_y, matrix.T, cmap = 'GnBu', norm=colors.LogNorm(vmin=0.5, vmax=np.max(matrix)))
    plt.colorbar(mat, label="weight")
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$t_{\textrm{max}}$")
    plt.savefig(plotdir / (f"weights_matrix_time.pdf"))
    plt.close()


if __name__ == "__main__":
    main()
