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
from common import read_correlators3
from common import read_correlators4
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
    """ Save the summed correlators as bootstraps in pickle files for one fixed value of lamdba 

    This is for the summer students so that they can practice diagonalising a matrix and fitting to a correlator to get the energy shift due to the feynman-hellmann modification.
    """

    pars = params(0) # Get the parameters for this lattice

    # Read in the directory data from the yaml file if one is given
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
        config_file = "data_dir_theta2.yaml"
    print("Reading directories from: ", config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Set things to defaults defined in another YAML file
    with open("defaults.yaml") as f:
        defaults = yaml.safe_load(f)
    for key, value in defaults.items():
        config.setdefault(key, value)
            
    pickledir_k1 = Path(config["pickle_dir1"])
    pickledir_k2 = Path(config["pickle_dir2"])
    plotdir = Path(config["analysis_dir"]) / Path("plots")
    datadir = Path(config["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist" in config and config["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(pars, pickledir_k1, pickledir_k2, mom_strings)
    elif "qmax" in config and config["qmax"]:
        G2_nucl, G2_sigm = read_correlators4(pars, pickledir_k1, pickledir_k2, mom_strings)
        print("qmax")
    else:
        print("else")
        G2_nucl, G2_sigm = read_correlators(pars, pickledir_k1, pickledir_k2, mom_strings)

    lambdas = np.array([0, 0.08])
    
    matrix4list = []
    for i, lmb_val in enumerate(lambdas):
        print(f"Lambda = {lmb_val}\n")
        # Construct a correlation matrix for each order in lambda(skipping order 0)
        matrix_1, matrix_2, matrix_3, matrix_4 = make_matrices(
            G2_nucl, G2_sigm, lmb_val
        )
        matrix4list.append(matrix_4)

    # Save the fit data to a pickle file
    all_data = {
        "nucleon_nucleon_l0": matrix4list[0][0][0],
        "sigma_sigma_l0": matrix4list[0][1][1],
        "nucleon_nucleon_lp08": matrix4list[1][0][0],
        "sigma_sigma_lp08": matrix4list[1][1][1],
        "nucleon_sigma_lp08": matrix4list[1][1][0],
        "sigma_nucleon_lp08": matrix4list[1][0][1],
    }
    with open(datadir / (f"summer_student_data.pkl"), "wb") as file_out:
        pickle.dump(all_data, file_out)

if __name__ == "__main__":
    main()

