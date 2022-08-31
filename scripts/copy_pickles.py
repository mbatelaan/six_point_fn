import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
from os.path import exists
import matplotlib.pyplot as plt

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY
from gevpanalysis.util import find_file
from gevpanalysis.util import read_config
from gevpanalysis.common import read_pickle
from gevpanalysis.common import transfer_pickle_data
from gevpanalysis.params import params


def main():
    """Diagonalise correlation matrices to calculate an energy shift for various lambda values"""
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)
    _metadata["Keywords"] = f"{pars.__dict__}"

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
    datadir = PROJECT_BASE_DIRECTORY / Path("data/pickles") / Path(config["name"])
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist2" in config and config["onlytwist2"]:
        transfer_pickle_data(pars, pickledir_k1, pickledir_k2, datadir)
    elif "qmax" in config and config["qmax"]:
        transfer_pickle_data(pars, pickledir_k1, pickledir_k2, datadir)


if __name__ == "__main__":
    main()
