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

def plotting_script_unpert(
    correlator1, correlator2, ratio, fitvals1, fitvals2, fitvals, t_range12, t_range, plotdir, name="", show=False
):
    """ Plot the effective mass and ratio of the unperturbed nucleon correlators with TBC implemented in two different ways
    """

    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(correlator1)[1])
    efftime = time[:-spacing] + 0.5
    correlator1 = stats.bs_effmass(correlator1, time_axis=1, spacing=1) 
    correlator2 = stats.bs_effmass(correlator2, time_axis=1, spacing=1) 
    effratio = stats.bs_effmass(ratio, time_axis=1, spacing=1) 
    yavg_1 = np.average(correlator1, axis=0)
    ystd_1 = np.std(correlator1, axis=0)
    yavg_2 = np.average(correlator2, axis=0)
    ystd_2 = np.std(correlator2, axis=0)
    yavg_ratio = np.average(ratio, axis=0)
    ystd_ratio = np.std(ratio, axis=0)
    yavg_effratio = np.average(effratio, axis=0)
    ystd_effratio = np.std(effratio, axis=0)

    f, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    f.tight_layout()
    axs[0].errorbar(
        efftime[:xlim],
        yavg_1[:xlim],
        ystd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    axs[0].errorbar(
        efftime[:xlim],
        yavg_2[:xlim],
        ystd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
    )
    axs[0].plot(t_range12, len(t_range12) * [np.average(fitvals1[:,1])], color=_colors[0])
    axs[0].fill_between(
        t_range12,
        np.average(fitvals1[:,1]) - np.std(fitvals1[:,1]),
        np.average(fitvals1[:,1]) + np.std(fitvals1[:,1]),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N^{{\textrm{{FermAct}}}}(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals1[:,1]),np.std(fitvals1[:,1]))}",
        label=rf"$E_N^{{\textrm{{FermAct}}}}(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals1[:,1]),np.std(fitvals1[:,1]))}, $A_N^{{\textrm{{FermAct}}}}$ = {err_brackets(np.average(fitvals1[:,0]),np.std(fitvals1[:,0]))}",
    )
    axs[0].plot(t_range12, len(t_range12) * [np.average(fitvals2[:,1])], color=_colors[1])
    axs[0].fill_between(
        t_range12,
        np.average(fitvals2[:,1]) - np.std(fitvals2[:,1]),
        np.average(fitvals2[:,1]) + np.std(fitvals2[:,1]),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{N}}^{{\textrm{{Gauge}}}}(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals2[:,1]),np.std(fitvals2[:,1]))}, $A_{{N}}^{{\textrm{{Gauge}}}}$ = {err_brackets(np.average(fitvals2[:,0]),np.std(fitvals2[:,0]))}",
    )

    axs[0].legend(fontsize="xx-small")

    axs[1].errorbar(
        time[:xlim],
        yavg_ratio[:xlim],
        ystd_ratio[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        # label=r"$G_{N}/G_{\Sigma}$",
    )
    axs[1].plot(t_range, len(t_range) * [np.average(fitvals)], color=_colors[0])
    axs[1].fill_between(
        t_range,
        np.average(fitvals) - np.std(fitvals),
        np.average(fitvals) + np.std(fitvals),
        alpha=0.3,
        color=_colors[2],
        label=rf"Fit = {err_brackets(np.average(fitvals),np.std(fitvals))}",
    )

    axs[1].axhline(y=1, color="k", alpha=0.6, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim))
    axs[0].set_ylabel(r"$\textrm{Effective energy}$")
    axs[1].set_ylabel(r"$G_N^{\textrm{FermAct}}(\mathbf{p}')/G_{N}^{\textrm{Gauge}}(\mathbf{p}')$")
    plt.xlabel("$t/a$")
    axs[1].legend(fontsize="x-small")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("unpert_ratio" + name + ".pdf"))
    if show:
        plt.show()
    plt.close()
    return

def main():
    """Investigate whether changing the twisted boundary conditions to the gauge field has an effect on the overlap factor of the correlator of the nucleon. 

    This could be the case since setting the TBC in the gauge fields allows us to take the momentum into consideration when inverting from a smeared source.
    """
    plt.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0) # Get the parameters for this lattice

    config_file1 = "data_dir_theta2.yaml"
    config_file2 = "data_dir_twisted_gauge5.yaml"
    with open(config_file1) as f:
        config1 = yaml.safe_load(f)
    with open(config_file2) as f:
        config2 = yaml.safe_load(f)
    pickledir_k1 = Path(config1["pickle_dir1"])
    pickledir_k2 = Path(config1["pickle_dir2"])
    pickledir_k1_2 = Path(config2["pickle_dir1"])
    pickledir_k2_2 = Path(config2["pickle_dir2"])
    plotdir = Path(config1["analysis_dir"]) / Path("plots")
    datadir = Path(config1["analysis_dir"]) / Path("data")
    plotdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # Read the correlator data from the pickle files
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]
    if "onlytwist" in config1 and config1["onlytwist"]:
        G2_nucl, G2_sigm = read_correlators2(pars, pickledir_k1, pickledir_k2, mom_strings)
    else:
        G2_nucl, G2_sigm = read_correlators(pars, pickledir_k1, pickledir_k2, mom_strings)

    if "onlytwist" in config2 and config2["onlytwist"]:
        G2_nucl2, G2_sigm2 = read_correlators2(pars, pickledir_k1_2, pickledir_k2_2, mom_strings)
    else:
        G2_nucl2, G2_sigm2 = read_correlators(pars, pickledir_k1_2, pickledir_k2_2, mom_strings)

    lambdas = np.linspace(0,0.05,30)
    t_range = np.arange(config1["t_range0"], config1["t_range1"])
    time_choice = config1["time_choice"]
    delta_t = config1["delta_t"]
    plotting = True

    order0_fit = np.zeros((len(lambdas), pars.nboot))
    order1_fit = np.zeros((len(lambdas), pars.nboot))
    order2_fit = np.zeros((len(lambdas), pars.nboot))
    order3_fit = np.zeros((len(lambdas), pars.nboot))
    red_chisq_list = np.zeros((4, len(lambdas)))
    
    aexp_function = ff.initffncs("Aexp")

    # Fit to the energy gap
    fit_range = np.arange(5,17)
    fit_range12 = np.arange(5,17)
    ratio_unpert = G2_nucl[0][:, :, 0] / G2_nucl2[0][:, :, 0]
    # ratio_unpert = G2_nucl[0][:, :, 0] / G2_sigm[0][:,:,0]
    bootfit1, redchisq1 = fit_value3(G2_nucl[0][:,:,0], fit_range12, aexp_function, norm=1)
    bootfit2, redchisq2 = fit_value3(G2_nucl2[0][:,:,0], fit_range12, aexp_function, norm=1)
    bootfit_ratio, redchisq_ratio = fit_value(ratio_unpert, fit_range)

    # diff = bootfit1[:,1]-bootfit2[:,1]
    plotting_script_unpert(
        G2_nucl[0][:, :, 0],
        G2_nucl2[0][:,:,0],
        ratio_unpert,
        bootfit1, #[:, 1],
        bootfit2, #[:, 1],
        bootfit_ratio[:, 0],
        fit_range12,
        fit_range,
        plotdir,
        name="_unpert_ratio_overlap",
        show=False,
    )

if __name__ == "__main__":
    main()

