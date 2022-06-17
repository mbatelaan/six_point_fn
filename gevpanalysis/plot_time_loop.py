import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
import matplotlib.pyplot as plt

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY

from analysis import stats
from analysis.formatting import err_brackets

from common import weighted_avg_1_2_exp
from common import weighted_avg

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


def plot_matrix(fitlist, plotdir, name):
    weights = np.array([i["weight"] for i in fitlist])
    x_coord = np.array([i["x"][0] for i in fitlist])
    y_coord = np.array([i["x"][-1] for i in fitlist])

    # print(max(weights))
    argument_w = np.argmax(weights)
    # print(argument_w)

    # print(fitlist[argument_w]["x"][0])
    # print(fitlist[argument_w]["x"][-1])
    # print(fitlist[argument_w]["redchisq"])

    # Find the unique values of tmin and tmax to make a grid showing the reduced chi-squared values.
    unique_x = np.unique(x_coord)
    unique_y = np.unique(y_coord)
    min_x = np.min(x_coord)
    min_y = np.min(y_coord)
    plot_x = np.append(unique_x, unique_x[-1] + 1)
    plot_y = np.append(unique_y, unique_y[-1] + 1)

    # print(f"\n\n{unique_x}")
    # print(f"{min_x}")
    # print(f"{unique_y}")
    # print(f"{min_y}")

    # matrix = np.zeros((len(unique_x), len(unique_y)))
    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(x_coord):
        matrix[x - min_x, y_coord[i] - min_y] = fitlist[i]["redchisq"]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(plot_x, plot_y, matrix.T, cmap="RdBu", vmin=0.0, vmax=2)
    plt.colorbar(mat, label=r"$\chi^2_{\textrm{dof}}$")
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$t_{\textrm{max}}$")
    plt.savefig(plotdir / (f"chisq_matrix_time_" + name + ".pdf"), metadata=_metadata)
    plt.close()

    # matrix = np.zeros((len(unique_x), len(unique_y)))
    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(x_coord):
        matrix[x - min_x, y_coord[i] - min_y] = fitlist[i]["weight"]
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(
        plot_x, plot_y, matrix.T, cmap="GnBu"  # , vmin=0.0, vmax=np.max(matrix)
    )
    plt.colorbar(mat, label="weight")
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$t_{\textrm{max}}$")
    plt.savefig(plotdir / (f"weights_matrix_time_" + name + ".pdf"), metadata=_metadata)
    plt.close()

    # matrix = np.zeros((len(unique_x), len(unique_y)))
    matrix = np.full((len(unique_x), len(unique_y)), np.nan)
    for i, x in enumerate(x_coord):
        matrix[x - min_x, y_coord[i] - min_y] = np.average(fitlist[i]["param"][:, 1])
    plt.figure(figsize=(5, 4))
    mat = plt.pcolormesh(
        plot_x, plot_y, matrix.T, cmap="GnBu"  # , vmin=0.4, vmax=np.max(matrix)
    )
    plt.colorbar(mat, label="energy")
    plt.xlabel(r"$t_{\textrm{min}}$")
    plt.ylabel(r"$t_{\textrm{max}}$")
    plt.savefig(plotdir / (f"energy_matrix_time_" + name + ".pdf"), metadata=_metadata)
    plt.close()
    return


def main():
    """Plot the chi-squared values and weights of a range of time fitting windows

    Read the data from a pickle file and plot it as a color plot on a matrix
    """
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    pars = params(0)  # Get the parameters for this lattice

    # Read in the directory data from the yaml file if one is given
    if len(sys.argv) == 2:
        config_file = Path(PROJECT_BASE_DIRECTORY) / Path("config/") / Path(sys.argv[1])
    else:
        config_file = Path(PROJECT_BASE_DIRECTORY) / Path("config/data_dir_theta7.yaml")
    print("Reading directories from: ", config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Set parameters to defaults defined in another YAML file
    with open(Path(PROJECT_BASE_DIRECTORY) / Path("config/defaults.yaml")) as f:
        defaults = yaml.safe_load(f)
    for key, value in defaults.items():
        config.setdefault(key, value)

    pickledir_k1 = Path(config["pickle_dir1"])
    pickledir_k2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    with open(datadir / (f"time_window_loop_nucl_1exp.pkl"), "rb") as file_in:
        fitlist_nucl_1exp = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_nucl_2exp.pkl"), "rb") as file_in:
        fitlist_nucl_2exp = pickle.load(file_in)

    with open(
        "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_1exp.pkl",
        "rb",
    ) as file_in:
        fitlist_sigma_1exp = pickle.load(file_in)
    with open(
        "/scratch/usr/hhpmbate/chroma_3pt/32x64/b5p50kp121040kp120620/six_point_fn_qmax/analysis/data/time_window_loop_sigma_2exp.pkl",
        "rb",
    ) as file_in:
        fitlist_sigma_2exp = pickle.load(file_in)

    with open(datadir / (f"time_window_loop_nucldivsigma_1exp.pkl"), "rb") as file_in:
        fitlist_nucldivsigma_1exp = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_nucldivsigma_2exp.pkl"), "rb") as file_in:
        fitlist_nucldivsigma_2exp = pickle.load(file_in)

    with open(datadir / (f"time_window_loop_lambda_small.pkl"), "rb") as file_in:
        fitlist_small = pickle.load(file_in)
    with open(datadir / (f"time_window_loop_lambda_large.pkl"), "rb") as file_in:
        fitlist_large = pickle.load(file_in)

    # ===== Nucleon data =====
    weights_nucl = np.array([i["weight"] for i in fitlist_nucl_1exp])
    weights_nucl_2 = np.array([i["weight"] for i in fitlist_nucl_2exp])
    high_weight_nucl = np.argmax(weights_nucl)
    # print(fitlist_nucl_1exp[high_weight_nucl]["redchisq"])
    nucl_t_range = np.arange(
        fitlist_nucl_1exp[high_weight_nucl]["x"][0],
        fitlist_nucl_1exp[high_weight_nucl]["x"][-1] + 1,
    )
    print(f"\nnucl_t_range = {nucl_t_range}")
    # print("redchisq", fitlist_nucl_1exp[high_weight_nucl]["redchisq"])
    # print("maxweight",max(weights_nucl))
    print("max: redchisq = ", fitlist_nucl_1exp[high_weight_nucl]["redchisq"])
    print("max: range = ", fitlist_nucl_1exp[high_weight_nucl]["x"])

    # ===== Sigma data =====
    weights_sigma = np.array([i["weight"] for i in fitlist_sigma_1exp])
    high_weight_sigma = np.argmax(weights_sigma)
    sigma_t_range = np.arange(
        fitlist_sigma_1exp[high_weight_sigma]["x"][0],
        fitlist_sigma_1exp[high_weight_sigma]["x"][-1] + 1,
    )
    print(f"\nsigma_t_range = {sigma_t_range}")
    # print(fitlist_sigma_1exp[high_weight_sigma]["redchisq"])
    # print("maxweight",max(weights_sigma))
    print("max: redchisq = ", fitlist_sigma_1exp[high_weight_sigma]["redchisq"])
    print("max: range = ", fitlist_sigma_1exp[high_weight_sigma]["x"])
    # print("max-1: redchisq = ", fitlist_sigma_1exp[high_weight_sigma-1]["redchisq"])
    # print("max-1: range = ", fitlist_sigma_1exp[high_weight_sigma-1]["x"])
    # print("max-2: redchisq = ", fitlist_sigma_1exp[high_weight_sigma-2]["redchisq"])
    # print("max-2: range = ", fitlist_sigma_1exp[high_weight_sigma-2]["x"])
    # print("max-3: redchisq = ", fitlist_sigma_1exp[high_weight_sigma-3]["redchisq"])
    # print("max-3: weight = ", fitlist_sigma_1exp[high_weight_sigma-3]["weight"])
    # print("max-3: range = ", fitlist_sigma_1exp[high_weight_sigma-3]["x"])

    weights_small = np.array([i["weight"] for i in fitlist_small])
    high_weight_small = np.argmax(weights_small)
    weights_large = np.array([i["weight"] for i in fitlist_large])
    high_weight_large = np.argmax(weights_large)
    ratio_t_range = np.arange(
        min(
            fitlist_small[high_weight_small]["x"][0],
            fitlist_large[high_weight_large]["x"][0],
        ),
        fitlist_large[high_weight_large]["x"][-1] + 1,
    )
    print(f"\nratio_t_range = {ratio_t_range}")
    # print(fitlist_small[high_weight_small]["redchisq"])
    # print("maxweight",max(weights_small))
    # print(fitlist_large[high_weight_large]["redchisq"])
    # print("maxweight",max(weights_large))

    print(f"small: {fitlist_small[high_weight_small]['x']}")
    print(f"large: {fitlist_large[high_weight_large]['x']}")
    print("max: redchisq = ", fitlist_small[high_weight_small]["redchisq"])
    print("max: range = ", fitlist_small[high_weight_small]["x"])
    # print("max-1: redchisq = ", fitlist_small[high_weight_small-1]["redchisq"])
    # print("max-1: range = ", fitlist_small[high_weight_small-1]["x"])
    # print("max-2: redchisq = ", fitlist_small[high_weight_small-2]["redchisq"])
    # print("max-2: range = ", fitlist_small[high_weight_small-2]["x"])
    # print("max-3: redchisq = ", fitlist_small[high_weight_small-3]["redchisq"])
    # print("max-3: range = ", fitlist_small[high_weight_small-3]["x"])

    print("\nmax: redchisq = ", fitlist_large[high_weight_large]["redchisq"])
    print("max: range = ", fitlist_large[high_weight_large]["x"])

    plot_matrix(fitlist_nucl_1exp, plotdir, "nucl_1exp")
    plot_matrix(fitlist_nucl_2exp, plotdir, "nucl_2exp")
    plot_matrix(fitlist_sigma_1exp, plotdir, "sigma_1exp")
    plot_matrix(fitlist_sigma_2exp, plotdir, "sigma_2exp")

    weighted_avg(
        fitlist_nucl_1exp,
        fitlist_nucl_2exp,
        plotdir,
        "nucl",
        tmax_choice=config["tmax_nucl"],
    )
    weighted_avg(
        fitlist_sigma_1exp,
        fitlist_sigma_2exp,
        plotdir,
        "sigma",
        tmax_choice=config["tmax_sigma"],
    )

    # ===== Nucleon divided by sigma data =====
    weights_nucldivsigma = np.array([i["weight"] for i in fitlist_nucldivsigma_1exp])
    weights_nucldivsigma_2 = np.array([i["weight"] for i in fitlist_nucldivsigma_2exp])
    high_weight_nucldivsigma = np.argmax(weights_nucldivsigma)
    nucldivsigma_t_range = np.arange(
        fitlist_nucldivsigma_1exp[high_weight_nucldivsigma]["x"][0],
        fitlist_nucldivsigma_1exp[high_weight_nucldivsigma]["x"][-1] + 1,
    )
    print(f"\nnucldivsigma_t_range = {nucldivsigma_t_range}")
    print(
        "max: redchisq = ",
        fitlist_nucldivsigma_1exp[high_weight_nucldivsigma]["redchisq"],
    )
    print("max: range = ", fitlist_nucldivsigma_1exp[high_weight_nucldivsigma]["x"])

    weighted_avg(
        fitlist_nucldivsigma_1exp,
        fitlist_nucldivsigma_2exp,
        plotdir,
        "nucldivsigma",
        tmax_choice=config["tmax_nucl"],
        tminmin_2exp=2,
        tminmax_2exp=2,
        tminmin_1exp=1,
        tminmax_1exp=15,
    )


if __name__ == "__main__":
    main()
