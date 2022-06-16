import numpy as np
from pathlib import Path
import pickle
import yaml
import sys
from os.path import exists
import scipy.optimize as syopt
import matplotlib.pyplot as plt
from matplotlib import rcParams

from gevpanalysis.definitions import PROJECT_BASE_DIRECTORY

from gevpanalysis.common import read_pickle
from gevpanalysis.common import fit_value
from gevpanalysis.common import fit_value3
from gevpanalysis.common import read_correlators
from gevpanalysis.common import read_correlators2
from gevpanalysis.common import read_correlators3
from gevpanalysis.common import read_correlators4
from gevpanalysis.common import read_correlators5_complex
from gevpanalysis.common import read_correlators6
from gevpanalysis.common import make_matrices
from gevpanalysis.common import normalize_matrices
from gevpanalysis.common import gevp
from gevpanalysis.common import gevp_bootstrap
from gevpanalysis.params import params

from analysis import stats
from analysis.bootstrap import bootstrap
from analysis.formatting import err_brackets
from analysis import fitfunc as ff


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
    """Diagonalise correlation matrices to calculate an energy shift for various lambda values"""
    # Plotting setup
    mystyle = Path(PROJECT_BASE_DIRECTORY) / Path("gevpanalysis/mystyle.txt")
    plt.style.use(mystyle.as_posix())

    # Get the parameters for this lattice ensemble (kp121040kp120620)
    pars = params(0)

    # Read in the analysis data from the yaml file if one is given
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = Path(PROJECT_BASE_DIRECTORY) / Path("config/data_dir_theta7.yaml")
        # config_file = "config/data_dir_theta2_fix.yaml"
    print(f"Reading directories from: {config_file}\n")
    with open(config_file) as f:
        config = yaml.safe_load(f)


if __name__ == "__main__":
    main()
