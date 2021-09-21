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
# _colors = ["r", "g", "b", "k", "y", "m", "k", "k"]
_markers = ["s", "o", "^", "*", "v", ">", "<", "s", "s"]
# sys.stdout = open("output.txt", "wt")
# From the theta tuning:
m_N = 0.4179255
m_S = 0.4641829


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


def gevp(corr_matrix, time_choice=10, delta_t=1, name="", show=None):
    """Solve the GEVP for a given correlation matrix

    corr_matrix has the matrix indices as the first two, then the bootstrap and then the time index
    time_choice is the timeslice on which the GEVP will be set
    delta_t is the size of the time evolution which will be used to solve the GEVP
    """
    # time_choice = 10
    # delta_t = 1
    mat_0 = np.average(corr_matrix[:, :, :, time_choice], axis=2)
    mat_1 = np.average(corr_matrix[:, :, :, time_choice + delta_t], axis=2)

    # wl, vl = np.linalg.eig(mat_0.T)
    # wr, vr = np.linalg.eig(mat_0)
    wl, vl = np.linalg.eig(np.matmul(mat_1, np.linalg.inv(mat_0)).T)
    wr, vr = np.linalg.eig(np.matmul(np.linalg.inv(mat_0), mat_1))
    # print(wl, vl)
    # print(wr, vr)

    Gt1 = np.einsum("i,ijkl,j->kl", vl[:, 0], corr_matrix, vr[:, 0])
    # print(np.shape(Gt1))
    Gt2 = np.einsum("i,ijkl,j->kl", vl[:, 1], corr_matrix, vr[:, 1])
    # print(np.shape(Gt2))

    if show:
        stats.ploteffmass(Gt1, "eig_1" + name, plotdir, show=True)
        stats.ploteffmass(Gt2, "eig_2" + name, plotdir, show=True)

    # print(f"{np.shape(mat)=}")
    # print(mat)
    # wl, vl = np.linalg.eig(mat.T)
    # wr, vr = np.linalg.eig(mat)
    # print(wl)
    # print(vl)
    # print(wr, vr)
    return Gt1, Gt2


def plotting_script_diff_2(
    diffG1, diffG2, diffG3, diffG4, fitvals, t_range, lmb_val, name="", show=False
):
    spacing = 2
    xlim = 15
    time = np.arange(0, np.shape(diffG1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = pypl.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

    yeffavg_1 = np.average(diffG1, axis=0)
    yeffstd_1 = np.std(diffG1, axis=0)
    axs.errorbar(
        efftime[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^1)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[0])], color=_colors[0])
    axs.fill_between(
        t_range,
        np.average(fitvals[0]) - np.std(fitvals[0]),
        np.average(fitvals[0]) + np.std(fitvals[0]),
        alpha=0.3,
        color=_colors[0],
    )
    yeffavg_2 = np.average(diffG2, axis=0)
    yeffstd_2 = np.std(diffG2, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.2,
        yeffavg_2[:xlim],
        yeffstd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^2)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[1])], color=_colors[1])
    axs.fill_between(
        t_range,
        np.average(fitvals[1]) - np.std(fitvals[1]),
        np.average(fitvals[1]) + np.std(fitvals[1]),
        alpha=0.3,
        color=_colors[1],
    )
    yeffavg_3 = np.average(diffG3, axis=0)
    yeffstd_3 = np.std(diffG3, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.4,
        yeffavg_3[:xlim],
        yeffstd_3[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^3)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[2])], color=_colors[2])
    axs.fill_between(
        t_range,
        np.average(fitvals[2]) - np.std(fitvals[2]),
        np.average(fitvals[2]) + np.std(fitvals[2]),
        alpha=0.3,
        color=_colors[2],
    )
    yeffavg_4 = np.average(diffG4, axis=0)
    yeffstd_4 = np.std(diffG4, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.6,
        yeffavg_4[:xlim],
        yeffstd_4[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^4)$",
    )
    axs.plot(t_range, len(t_range) * [np.average(fitvals[3])], color=_colors[3])
    axs.fill_between(
        t_range,
        np.average(fitvals[3]) - np.std(fitvals[3]),
        np.average(fitvals[3]) + np.std(fitvals[3]),
        alpha=0.3,
        color=_colors[3],
    )

    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    pypl.ylabel(r"$\Delta E_{\textrm{eff}}/\lambda$")
    pypl.xlabel("$t/a$")
    pypl.legend(fontsize="x-small")
    pypl.title("$\lambda=" + str(lmb_val) + "$")
    pypl.savefig(plotdir / ("diff_G" + name + ".pdf"))
    if show:
        pypl.show()
    pypl.close()
    return


def fit_value(diffG, t_range):
    """Fit a constant function to the diffG correlator

    diffG is a correlator with tht bootstraps on the first index and the time on the second
    t_range is an array of time values to fit over
    the function will return an array of fit parameters for each bootstrap
    """
    diffG_avg = np.average(diffG, axis=0)[t_range]
    covmat = np.diag(np.std(diffG, axis=0)[t_range] ** 2)
    # covmat = np.cov(diffG[t_range])
    popt_avg, pcov_avg = curve_fit(ff.constant, t_range, diffG_avg, sigma=covmat)
    bootfit = []
    for iboot, values in enumerate(diffG):
        popt, pcov = curve_fit(ff.constant, t_range, values[t_range], sigma=covmat)
        bootfit.append(popt)
    bootfit = np.array(bootfit)
    # print(popt_avg)
    # print(np.average(bootfit))
    return bootfit

def read_correlators(pars, pickledir, mom_strings):
    """Read the pickle files which contain the correlator data

    The script will check the folders for available files and pick out the files with the highest number of configurations.
    """
    ### ----------------------------------------------------------------------
    ### find the highest number of configurations available
    files = (
         pickledir
        / Path(
            "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/p+1+0+0/"
        )
    ).glob("barspec_nucleon_rel_[0-9]*cfgs.pickle")
    # Strip the conf number from the file names
    conf_num_list = np.array([int("".join(filter(str.isdigit, l.name))) for l in list(files)])
    print(conf_num_list)
    # conf_num_list = [50] # hard code a choice
    conf_num = conf_num_list[np.argmax(conf_num_list)]
    barspec_name = "/barspec_nucleon_rel_" + str(conf_num) + "cfgs.pickle"


    ### ----------------------------------------------------------------------
    G2_nucl = []
    G2_sigm = []
    ### ----------------------------------------------------------------------
    ### Unperturbed correlators
    unpertfile_nucleon_pos = pickledir / Path(
        "baryon_qcdsf_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    unpertfile_sigma = pickledir2 / Path(
        "baryon_qcdsf/barspec/32x64/unpreconditioned_slrc/kp121040kp120620/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    G2_unpert_qp100_nucl = read_pickle(unpertfile_nucleon_pos, nboot=pars.nboot, nbin=pars.nbin)
    G2_unpert_q000_sigma = read_pickle(unpertfile_sigma, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_unpert_qp100_nucl)
    G2_sigm.append(G2_unpert_q000_sigma)

    ### ----------------------------------------------------------------------
    ### SU & SS
    filelist_SU1 = pickledir2 / Path(
        "baryon-3pt_SU_lmb_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_SU3 = pickledir2 / Path(
        "baryon-3pt_SU_lmb3_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_SS2 = pickledir2 / Path(
        "baryon-3pt_SS_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )
    filelist_SS4 = pickledir2 / Path(
        "baryon-3pt_SS_lmb4_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[1]
        + barspec_name
    )

    G2_q100_SU_lmb = read_pickle(filelist_SU1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb2 = read_pickle(filelist_SS2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_SU_lmb3 = read_pickle(filelist_SU3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_SS_lmb4 = read_pickle(filelist_SS4, nboot=pars.nboot, nbin=pars.nbin)
    G2_sigm.append(G2_q100_SU_lmb)
    G2_sigm.append(G2_q000_SS_lmb2)
    G2_sigm.append(G2_q100_SU_lmb3)
    G2_sigm.append(G2_q000_SS_lmb4)

    ### ----------------------------------------------------------------------
    ### US & UU
    filelist_US1 = pickledir / Path(
        "baryon-3pt_US_lmb_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )
    filelist_UU2 = pickledir / Path(
        "baryon-3pt_UU_lmb2_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )
    filelist_US3 = pickledir / Path(
        "baryon-3pt_US_lmb3_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp120620/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[0]
        + barspec_name
    )
    filelist_UU4 = pickledir / Path(
        "baryon-3pt_UU_lmb4_TBC/barspec/32x64/unpreconditioned_slrc_slrc/kp121040kp121040/lp0lp0__lp0lp0/sh_gij_p21_90-sh_gij_p21_90/"
        + mom_strings[2]
        + barspec_name
    )

    G2_q000_US_lmb = read_pickle(filelist_US1, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb2 = read_pickle(filelist_UU2, nboot=pars.nboot, nbin=pars.nbin)
    G2_q000_US_lmb3 = read_pickle(filelist_US3, nboot=pars.nboot, nbin=pars.nbin)
    G2_q100_UU_lmb4 = read_pickle(filelist_UU4, nboot=pars.nboot, nbin=pars.nbin)
    G2_nucl.append(G2_q000_US_lmb)
    G2_nucl.append(G2_q100_UU_lmb2)
    G2_nucl.append(G2_q000_US_lmb3)
    G2_nucl.append(G2_q100_UU_lmb4)

    return G2_nucl, G2_sigm

def makematrices(G2_nucl, G2_sigm, lmb_val):
    matrix_1 = np.array(
        [
            [G2_nucl[0][:, :, 0], lmb_val * G2_nucl[1][:, :, 0]],
            [lmb_val * G2_sigm[1][:, :, 0], G2_sigm[0][:, :, 0]],
        ]
    )
    matrix_2 = np.array(
        [
            [
                G2_nucl[0][:, :, 0]
                + lmb_val ** 2 * G2_nucl[2][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0],
                G2_sigm[0][:, :, 0]
                + lmb_val ** 2 * G2_sigm[2][:, :, 0],
            ],
        ]
    )
    matrix_3 = np.array(
        [
            [
                G2_nucl[0][:, :, 0]
                + lmb_val ** 2 * G2_nucl[2][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0]
            + lmb_val ** 3 * G2_nucl[3][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0]
                + lmb_val ** 3 * G2_sigm[3][:, :, 0],
                G2_sigm[0][:, :, 0]
                + lmb_val ** 2 * G2_sigm[2][:, :, 0],
            ],
        ]
    )
    matrix_4 = np.array(
        [
            [
                G2_nucl[0][:, :, 0]
                + (lmb_val ** 2) * G2_nucl[2][:, :, 0]
                + (lmb_val ** 4) * G2_nucl[4][:, :, 0],
                lmb_val * G2_nucl[1][:, :, 0]
                + (lmb_val ** 3) * G2_nucl[3][:, :, 0],
            ],
            [
                lmb_val * G2_sigm[1][:, :, 0]
                + (lmb_val ** 3) * G2_sigm[3][:, :, 0],
                G2_sigm[0][:, :, 0]
                + (lmb_val ** 2) * G2_sigm[2][:, :, 0]
                + (lmb_val ** 4) * G2_sigm[4][:, :, 0],
            ],
        ]
    )

    return matrix_1, matrix_2, matrix_3, matrix_4
    
    
if __name__ == "__main__":
    pypl.rc("font", size=18, **{"family": "sans-serif", "serif": ["Computer Modern"]})
    pypl.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    pars = params(0)
    # nboot = 500  # 700
    nbin = 1  # 10

    # Read in the directory data from the yaml file
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
    else:
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

    momenta = ["mass"]
    mom_strings = ["p-1+0+0", "p+0+0+0", "p+1+0+0"]

    G2_nucl, G2_sigm = read_correlators(pars, pickledir, mom_strings)

    lambdas = np.linspace(0,0.16,30)[1:]
    plotting = True
    lmb_val = 0.06 #0.16
    # for lmb_val in lambdas:
    # time_choice_range = np.arange(2,15)
    time_choice_range = np.arange(2,15)
    # delta_t = 2
    delta_t_range = np.arange(1,4)
    t_range = np.arange(4, 9)


    order0_fit = np.zeros((len(time_choice_range),len(delta_t_range), pars.nboot))
    order1_fit = np.zeros((len(time_choice_range),len(delta_t_range), pars.nboot))
    order2_fit = np.zeros((len(time_choice_range),len(delta_t_range), pars.nboot))
    order3_fit = np.zeros((len(time_choice_range),len(delta_t_range), pars.nboot))

    for i,time_choice in enumerate(time_choice_range):
        for j,delta_t in enumerate(delta_t_range):

            # Construct a correlation matrix for each order in lambda (skipping order 0)
            matrix_1, matrix_2, matrix_3, matrix_4 = makematrices(G2_nucl, G2_sigm, lmb_val)
            
            ### ----------------------------------------------------------------------
            print(f"\n HERE {time_choice}\n")
            Gt1_1, Gt2_1 = gevp(matrix_1, time_choice, delta_t, name="_test", show=False)
            effmassdata_1 = stats.bs_effmass(Gt1_1, time_axis=1, spacing=1)
            effmassdata_2 = stats.bs_effmass(Gt2_1, time_axis=1, spacing=1)
            diffG1 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
            bootfit1 = fit_value(diffG1, t_range)
            order0_fit[i,j] = bootfit1[:, 0]
            # order0_fit.append(bootfit1[:, 0])
            
            Gt1_2, Gt2_2 = gevp(matrix_2, time_choice, delta_t, name="_test", show=False)
            effmassdata_1 = stats.bs_effmass(Gt1_2, time_axis=1, spacing=1)
            effmassdata_2 = stats.bs_effmass(Gt2_2, time_axis=1, spacing=1)
            diffG2 = np.abs(effmassdata_1 - effmassdata_2) / 2  # / lmb_val
            bootfit2 = fit_value(diffG2, t_range)
            order1_fit[i,j] = bootfit2[:, 0]
            # order1_fit.append(bootfit2[:, 0])
            
            Gt1_3, Gt2_3 = gevp(matrix_3, time_choice, delta_t, name="_test", show=False)
            effmassdata_1_3 = stats.bs_effmass(Gt1_3, time_axis=1, spacing=1)
            effmassdata_2_3 = stats.bs_effmass(Gt2_3, time_axis=1, spacing=1)
            diffG3 = np.abs(effmassdata_1_3 - effmassdata_2_3) / 2  # / lmb_val
            bootfit3 = fit_value(diffG3, t_range)
            order2_fit[i,j] = bootfit3[:, 0]
            # order2_fit.append(bootfit3[:, 0])
            
            Gt1_4, Gt2_4 = gevp(matrix_4, time_choice, delta_t, name="_test", show=False)
            effmassdata_1_4 = stats.bs_effmass(Gt1_4, time_axis=1, spacing=1)
            effmassdata_2_4 = stats.bs_effmass(Gt2_4, time_axis=1, spacing=1)
            diffG4 = np.abs(effmassdata_1_4 - effmassdata_2_4) / 2  # / lmb_val
            bootfit4 = fit_value(diffG4, t_range)
            order3_fit[i,j] = bootfit4[:, 0]
            # order3_fit.append(bootfit4[:, 0])
            
            if plotting:
                plotting_script_diff_2(
                    diffG1,
                    diffG2,
                    diffG3,
                    diffG4,
                    [bootfit1, bootfit2, bootfit3, bootfit4],
                    t_range,
                    lmb_val,
                    name="_l" + str(lmb_val) + "_time_choice"+str(time_choice),
                    show=False,
                )
                
    print(f"\n\n\n END of LOOP \n\n")


    all_data = {
        "lambdas" : np.array([lmb_val]),
        "order0_fit" : order0_fit, 
        "order1_fit" : order1_fit,
        "order2_fit" : order2_fit,
        "order3_fit" : order3_fit,
        "time_choice" : time_choice_range,
        "delta_t" : delta_t_range
    }

    with open(datadir / (f"gevp_time_dep_l{lmb_val}.pkl"), "wb") as file_out:
        pickle.dump(all_data, file_out)

    # with open(datadir / ("fit_data_time_choice"+str(time_choice_range[0])+"-"+str(time_choice_range[-1])+".pkl"), "wb") as file_out:
    #     pickle.dump([time_choice_range, delta_t_range, order0_fit, order1_fit,order2_fit,order3_fit],file_out)
    #     # pickle.dump(np.array([lambdas,order0_fit, order1_fit,order2_fit,order3_fit],dtype=object),file_out)
    
    pypl.figure(figsize=(6, 6))
    pypl.errorbar(
        time_choice_range,
        np.average(order0_fit[:,0,:], axis=1),
        np.std(order0_fit[:,0,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^1)$",
        color=_colors[0],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.001,
        np.average(order1_fit[:,0,:], axis=1),
        np.std(order1_fit[:,0,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^2)$",
        color=_colors[1],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.002,
        np.average(order2_fit[:,0,:], axis=1),
        np.std(order2_fit[:,0,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^3)$",
        color=_colors[2],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )
    pypl.errorbar(
        time_choice_range+0.003,
        np.average(order3_fit[:,0,:], axis=1),
        np.std(order3_fit[:,0,:], axis=1),
        fmt="s",
        label=r"$\mathcal{O}(\lambda^4)$",
        color=_colors[3],
        capsize=4,
        elinewidth=1,
        markerfacecolor="none",
    )

    pypl.legend(fontsize="x-small")
    # pypl.xlim(-0.01, 0.22)
    # pypl.ylim(0, 0.2)
    pypl.xlabel("time choice")
    pypl.ylabel("$\Delta E$")
    pypl.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    pypl.savefig(plotdir / (f"time_choice_dep_l{lmb_val}.pdf"))
    # pypl.show()
