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
from gevpanalysis.common import read_pickle
from gevpanalysis.common import transfer_pickle_data
from gevpanalysis.params import params

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff

_colors = [
    (0, 0, 0),
    (0.9, 0.6, 0),
    (0.35, 0.7, 0.9),
    (0, 0.6, 0.5),
    (0.95, 0.9, 0.25),
    (0, 0.45, 0.7),
    (0.8, 0.4, 0),
    (0.8, 0.6, 0.7),
]

_markers = ["s", "^", "o", ".", "p", "v", "P", ",", "*"]


def main():
    """
    Read the FH correlation functions and plot them to show how the slope changes with higher orders.
    """
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)

    # Read in the analysis data from the yaml file if one is given
    qmax_config = read_config("qmax")
    qmax_datadir = Path(qmax_config["analysis_dir"]) / Path("data")
    if len(sys.argv) == 2:
        config = read_config(sys.argv[1])
    else:
        config = read_config("qmax")

    # Set parameters to defaults defined in another YAML file
    defaults = read_config("defaults")
    for key, value in defaults.items():
        config.setdefault(key, value)

    # Set the directories for reading data, saving data and saving plots
    pickledir_k1 = Path(config["pickle_dir1"])
    pickledir_k2 = Path(config["pickle_dir2"])
    plotdir = PROJECT_BASE_DIRECTORY / Path("data/plots") / Path(config["name"])
    # datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
    datadir = Path(
        "/home/mischa/Documents/PhD/analysis_results/six_point_fn_all/data/pickles"
    ) / Path(config["name"])
    correlator_dir = datadir / Path("correlator_data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    correlator_dir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    # Define the destination file names
    write_file_nucleon_lmb0 = datadir / Path(
        f"correlator_data/corr_UU_lmb0_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb0 = datadir / Path(
        f"correlator_data/corr_SS_lmb0_{mom_strings[1]}.pkl"
    )

    # Define the Sigma file names
    write_file_sigma_lmb1 = datadir / Path(
        f"correlator_data/corr_SU_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_sigma_lmb3 = datadir / Path(
        f"correlator_data/corr_SU_lmb3_{mom_strings[1]}.pkl"
    )
    # write_file_sigma_lmb2 = datadir / Path(
    #     f"correlator_data/corr_SS_lmb2_{mom_strings[1]}.pkl"
    # )
    # write_file_sigma_lmb4 = datadir / Path(
    #     f"correlator_data/corr_SS_lmb4_{mom_strings[1]}.pkl"
    # )
    # shutil.copyfile(filelist_SU1, write_file_sigma_lmb1)

    # Define the nucleon file names
    write_file_nucleon_lmb1 = datadir / Path(
        f"correlator_data/corr_US_lmb1_{mom_strings[1]}.pkl"
    )
    write_file_nucleon_lmb3 = datadir / Path(
        f"correlator_data/corr_US_lmb3_{mom_strings[1]}.pkl"
    )
    # write_file_nucleon_lmb2 = datadir / Path(
    #     f"correlator_data/corr_UU_lmb2_{mom_strings[1]}.pkl"
    # )
    # write_file_nucleon_lmb4 = datadir / Path(
    #     f"correlator_data/corr_UU_lmb4_{mom_strings[1]}.pkl"
    # )

    G2_nucleon_unpert = read_pickle(
        write_file_nucleon_lmb0, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_sigma_unpert = read_pickle(
        write_file_sigma_lmb0, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_sigma_lmb1 = read_pickle(write_file_sigma_lmb1, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigma_lmb3 = read_pickle(write_file_sigma_lmb3, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucleon_lmb1 = read_pickle(
        write_file_nucleon_lmb1, nboot=pars.nboot, nbin=pars.nbin
    )
    G2_nucleon_lmb3 = read_pickle(
        write_file_nucleon_lmb3, nboot=pars.nboot, nbin=pars.nbin
    )

    lmb = 0.08
    ratio1 = lmb * G2_sigma_lmb1[:, :, 0] / G2_sigma_unpert[:, :, 0]
    ratio2 = (
        lmb * G2_sigma_lmb1[:, :, 0] + lmb**3 * G2_sigma_lmb3[:, :, 0]
    ) / G2_sigma_unpert[:, :, 0]
    ratio3 = lmb * G2_nucleon_lmb1[:, :, 0] / G2_nucleon_unpert[:, :, 0]
    ratio4 = (
        lmb * G2_nucleon_lmb1[:, :, 0] + lmb**3 * G2_nucleon_lmb3[:, :, 0]
    ) / G2_nucleon_unpert[:, :, 0]
    print(f"{np.average(ratio1, axis=0)=}")
    print(f"{np.average(ratio2, axis=0)=}")

    slope1 = (ratio1[:, 1:] - ratio1[:, :-1]) / lmb
    slope2 = (ratio2[:, 1:] - ratio2[:, :-1]) / lmb
    slope3 = (ratio3[:, 1:] - ratio3[:, :-1]) / lmb
    slope4 = (ratio4[:, 1:] - ratio4[:, :-1]) / lmb

    effmass1 = stats.bs_effmass(G2_sigma_lmb1[:, :, 0], time_axis=1)

    # ======================================================================
    # 3pt
    threept_fn1 = lmb * G2_sigma_lmb1[:, :, 0]
    threept_fn2 = lmb * G2_sigma_lmb1[:, :, 0] + lmb**3 * G2_sigma_lmb3[:, :, 0]
    plt.figure()
    plt.errorbar(
        np.arange(np.shape(threept_fn1)[1]),
        np.average(threept_fn1, axis=0),
        np.std(threept_fn1, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="threept_fn1",
    )
    plt.errorbar(
        np.arange(np.shape(threept_fn2)[1]),
        np.average(threept_fn2, axis=0),
        np.std(threept_fn2, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_markers[1],
        markerfacecolor="none",
        label="threept_fn2",
    )

    plt.legend()
    # plt.ylim(-1 * lmb, 30 * lmb)
    # plt.ylim(0.15, 0.6)
    plt.xlim(-1, 20)
    # plt.semilogy()
    plt.savefig(plotdir / Path("3pt_sigma.pdf"))
    # plt.show()
    plt.close()

    # ======================================================================
    # Effmass
    plt.figure()
    plt.errorbar(
        np.arange(np.shape(effmass1)[1]),
        np.average(effmass1, axis=0),
        np.std(effmass1, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="effmass1",
    )
    # plt.errorbar(
    #     np.arange(np.shape(ratio2)[1]),
    #     np.average(ratio2, axis=0),
    #     np.std(ratio2, axis=0),
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[1],
    #     fmt=_markers[1],
    #     markerfacecolor="none",
    #     label="ratio2",
    # )

    plt.legend()
    # plt.ylim(-1 * lmb, 30 * lmb)
    plt.ylim(0.15, 0.6)
    plt.xlim(-1, 20)
    plt.savefig(plotdir / Path("effmass_sigma.pdf"))
    # plt.show()
    plt.close()

    # ======================================================================
    # Ratio
    plt.figure()
    plt.errorbar(
        np.arange(np.shape(ratio1)[1]),
        np.average(ratio1, axis=0),
        np.std(ratio1, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="ratio1",
    )
    plt.errorbar(
        np.arange(np.shape(ratio2)[1]),
        np.average(ratio2, axis=0),
        np.std(ratio2, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_markers[1],
        markerfacecolor="none",
        label="ratio2",
    )

    plt.legend()
    # plt.ylim(-1, 30)
    plt.ylim(-1 * lmb, 30 * lmb)
    # plt.ylim(-0.3, 2)
    # plt.ylim(-100, 300)
    plt.xlim(-1, 25)
    plt.savefig(plotdir / Path("ratio_sigma.pdf"))
    # plt.show()
    plt.close()

    plt.figure()
    plt.errorbar(
        np.arange(np.shape(slope1)[1]),
        np.average(slope1, axis=0),
        np.std(slope1, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="slope1",
    )
    plt.errorbar(
        np.arange(np.shape(slope2)[1]),
        np.average(slope2, axis=0),
        np.std(slope2, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_markers[1],
        markerfacecolor="none",
        label="slope2",
    )

    plt.legend()
    # plt.ylim(-1, 30)
    # plt.ylim(-1 * lmb, 30 * lmb)
    # plt.ylim(-0.04, 0.2)
    plt.ylim(-0.0 / lmb, 0.15 / lmb)
    # plt.ylim(-0.3, 2)
    # plt.ylim(-100, 300)
    plt.xlim(-1, 25)
    plt.savefig(plotdir / Path("slope_sigma.pdf"))
    # plt.show()
    plt.close()

    # ======================================================================
    # Nucleon
    # ----------------------------------------------------------------------
    plt.figure()
    plt.errorbar(
        np.arange(np.shape(ratio3)[1]),
        np.average(ratio3, axis=0),
        np.std(ratio3, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="ratio3",
    )
    plt.errorbar(
        np.arange(np.shape(ratio4)[1]),
        np.average(ratio4, axis=0),
        np.std(ratio4, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_markers[1],
        markerfacecolor="none",
        label="ratio4",
    )

    plt.legend()
    # plt.ylim(-1, 30)
    plt.ylim(-1 * lmb, 30 * lmb)
    # plt.ylim(-0.3, 2)
    # plt.ylim(-100, 300)
    plt.xlim(-1, 25)
    plt.savefig(plotdir / Path("ratio_nucleon.pdf"))
    # plt.show()
    plt.close()

    plt.figure()
    plt.errorbar(
        np.arange(np.shape(slope3)[1]),
        np.average(slope3, axis=0),
        np.std(slope3, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_markers[0],
        markerfacecolor="none",
        label="slope3",
    )
    plt.errorbar(
        np.arange(np.shape(slope4)[1]),
        np.average(slope4, axis=0),
        np.std(slope4, axis=0),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_markers[1],
        markerfacecolor="none",
        label="slope4",
    )

    plt.legend()
    # plt.ylim(-0.0, 0.15)
    plt.ylim(-0.0 / lmb, 0.15 / lmb)
    plt.xlim(-1, 25)
    plt.savefig(plotdir / Path("slope_nucleon.pdf"))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
