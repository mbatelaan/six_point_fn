import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt


def plotting_script_all(
    corr_matrix,
    corr_matrix1,
    corr_matrix2,
    corr_matrix3,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[1][1], axis=0)
    ystd = np.std(corr_matrix[1][1], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0)$",
    )
    yavg = np.average(corr_matrix1[1][1], axis=0)
    ystd = np.std(corr_matrix1[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix3[1][1], axis=0)
    ystd = np.std(corr_matrix3[1][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.4,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{\Sigma\Sigma}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )

    yavg = np.average(corr_matrix[1][0], axis=0)
    ystd = np.std(corr_matrix[1][0], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{\Sigma N}(t),\ \mathcal{O}(\lambda^1)$",
    )
    yavg = np.average(corr_matrix2[1][0], axis=0)
    ystd = np.std(corr_matrix2[1][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{\Sigma N}(t),\ \mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )

    plt.semilogy()
    plt.legend(fontsize="x-small")
    # plt.ylabel(r"$G_{nn}(t;\vec{p}=(1,0,0))$")
    # plt.title("$\lambda=0.04$")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    # plt.xlabel(r"$\textrm{t/a}$")
    plt.xlabel(r"$t$")
    plt.ylim(1e-5, 3e2)
    metadata_ = _metadata
    # metadata_["Keywords"] = f"lmb={lmb_val}"
    plt.savefig(plotdir / ("comp_plot_all_SS_" + name + ".pdf"), metadata=metadata_)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_all_N(
    corr_matrix,
    corr_matrix1,
    corr_matrix2,
    corr_matrix3,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr_matrix[0][0])[1])

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    yavg = np.average(corr_matrix[0][0], axis=0)
    ystd = np.std(corr_matrix[0][0], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0)$",
    )
    yavg = np.average(corr_matrix1[0][0], axis=0)
    ystd = np.std(corr_matrix1[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2)$",
    )
    yavg = np.average(corr_matrix3[0][0], axis=0)
    ystd = np.std(corr_matrix3[0][0], axis=0)
    axs.errorbar(
        time[:xlim] + 0.4,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="s",
        markerfacecolor="none",
        label=r"$G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )

    yavg = np.average(corr_matrix[0][1], axis=0)
    ystd = np.std(corr_matrix[0][1], axis=0)
    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{N\Sigma}(t),\ \mathcal{O}(\lambda^1)$",
    )
    yavg = np.average(corr_matrix2[0][1], axis=0)
    ystd = np.std(corr_matrix2[0][1], axis=0)
    axs.errorbar(
        time[:xlim] + 0.2,
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt="^",
        markerfacecolor="none",
        label=r"$G_{N\Sigma}(t),\ \mathcal{O}(\lambda^1) + \mathcal{O}(\lambda^3)$",
    )

    plt.semilogy()
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t$")
    plt.ylim(1e-5, 3e2)
    metadata_ = _metadata
    # metadata_["lambda"] = lmb_val
    plt.savefig(plotdir / ("comp_plot_all_NN_" + name + ".pdf"), metadata=metadata_)
    if show:
        plt.show()
    plt.close()
    return


def plot_real_imag(
    corr_matrix3,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr_matrix3[0][0])[1])

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

    yavg = np.average(np.real(corr_matrix3[0][0]), axis=0)
    ystd = np.std(np.real(corr_matrix3[0][0]), axis=0)

    yavgi = np.average(np.imag(corr_matrix3[0][0]), axis=0)
    ystdi = np.std(np.imag(corr_matrix3[0][0]), axis=0)

    axs.errorbar(
        time[:xlim],
        yavg[:xlim],
        ystd[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"real $G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )
    axs.errorbar(
        time[:xlim],
        yavgi[:xlim],
        ystdi[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="o",
        markerfacecolor="none",
        label=r"imag $G_{NN}(t),\ \mathcal{O}(\lambda^0) + \mathcal{O}(\lambda^2) + \mathcal{O}(\lambda^4)$",
    )

    plt.semilogy()
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t$")
    plt.ylim(1e-5, 3e2)
    metadata_ = _metadata
    # metadata_["lambda"] = lmb_val
    plt.savefig(plotdir / ("comp_real_imag_" + name + ".pdf"), metadata=metadata_)
    if show:
        plt.show()
    plt.close()
    return


def plot_real_imag_gevp(
    corr1,
    corr2,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 25
    time = np.arange(0, np.shape(corr1)[1])
    efftime = time[:-spacing] + 0.5
    # effmass_corr1 = stats.bs_effmass(np.real(corr1), time_axis=1, spacing=1)
    # effmass_corr2 = stats.bs_effmass(np.real(corr2), time_axis=1, spacing=1)
    # effmass_corr1i = stats.bs_effmass(np.imag(corr1), time_axis=1, spacing=1)
    # effmass_corr2i = stats.bs_effmass(np.imag(corr2), time_axis=1, spacing=1)
    yeffavg_1 = np.average(np.real(corr1), axis=0)
    yeffstd_1 = np.std(np.real(corr1), axis=0)
    yeffavg_1i = np.average(np.imag(corr1), axis=0)
    yeffstd_1i = np.std(np.imag(corr1), axis=0)
    yeffavg_2 = np.average(np.real(corr2), axis=0)
    yeffstd_2 = np.std(np.real(corr2), axis=0)
    yeffavg_2i = np.average(np.imag(corr2), axis=0)
    yeffstd_2i = np.std(np.imag(corr2), axis=0)

    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    axs.errorbar(
        time[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"real corr1",
    )
    axs.errorbar(
        time[:xlim],
        yeffavg_1i[:xlim],
        yeffstd_1i[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="o",
        markerfacecolor="none",
        label=r"imag corr1",
    )

    axs.errorbar(
        time[:xlim],
        yeffavg_2[:xlim],
        yeffstd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt="x",
        markerfacecolor="none",
        label=r"real corr2",
    )
    axs.errorbar(
        time[:xlim],
        yeffavg_2i[:xlim],
        yeffstd_2i[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt="^",
        markerfacecolor="none",
        label=r"imag corr2",
    )

    plt.semilogy()
    plt.legend(fontsize="x-small")
    plt.xlabel(r"$t$")
    plt.ylim(1e-10, 3e2)
    metadata_ = _metadata
    # metadata_["lambda"] = lmb_val
    plt.savefig(plotdir / ("comp_real_imag_gevp_" + name + ".pdf"), metadata=metadata_)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_diff_2(
    diffG1,
    diffG2,
    diffG3,
    diffG4,
    fitvals,
    t_range,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(diffG1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

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
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim), ylim=(-0.05, 0.25))
    # plt.ylabel(r"$\Delta E_{\textrm{eff}}$")
    # plt.ylabel(r"$(R_{\lambda}(t,\vec{q}))_{\textrm{eff}}$")
    plt.ylabel(r"$\textrm{eff. energy}\left[(R_{\lambda}(t,\vec{q})\right]$")
    plt.xlabel("$t$")
    plt.legend(fontsize="x-small")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("diff_G" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_gevp_corr(
    corr1,
    corr2,
    fit1,
    fit2,
    redchisq1,
    redchisq2,
    t_range,
    lmb_val,
    plotdir,
    name="",
    show=False,
):
    spacing = 2
    xlim = 20
    time = np.arange(0, np.shape(corr1)[1])
    efftime = time[:-spacing] + 0.5
    f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)

    effmass_corr1 = stats.bs_effmass(corr1, time_axis=1, spacing=1)
    effmass_corr2 = stats.bs_effmass(corr2, time_axis=1, spacing=1)
    yeffavg_1 = np.average(effmass_corr1, axis=0)
    yeffstd_1 = np.std(effmass_corr1, axis=0)
    axs.errorbar(
        efftime[:xlim],
        yeffavg_1[:xlim],
        yeffstd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^4)$, state 1",
    )
    axs.plot(
        t_range,
        len(t_range) * [np.average(fit1[:, 1])],
        color=_colors[0],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {redchisq1:.2f}$",
    )
    axs.fill_between(
        t_range,
        np.average(fit1[:, 1]) - np.std(fit1[:, 1]),
        np.average(fit1[:, 1]) + np.std(fit1[:, 1]),
        alpha=0.3,
        color=_colors[0],
    )
    yeffavg_2 = np.average(effmass_corr2, axis=0)
    yeffstd_2 = np.std(effmass_corr2, axis=0)
    axs.errorbar(
        efftime[:xlim] + 0.2,
        yeffavg_2[:xlim],
        yeffstd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        label=r"$\mathcal{O}(\lambda^4)$, state 2",
    )
    axs.plot(
        t_range,
        len(t_range) * [np.average(fit2[:, 1])],
        color=_colors[1],
        label=rf"$\chi^2_{{\textrm{{dof}}}} = {redchisq2:.2f}$",
    )
    axs.fill_between(
        t_range,
        np.average(fit2[:, 1]) - np.std(fit2[:, 1]),
        np.average(fit2[:, 1]) + np.std(fit2[:, 1]),
        alpha=0.3,
        color=_colors[1],
    )

    axs.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.setp(axs, xlim=(0, xlim), ylim=(0, 1))
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.05, 0.25))
    plt.ylabel(r"$E_{\textrm{eff}}$")
    plt.xlabel("$t$")
    plt.legend(fontsize="x-small")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.grid(True, alpha=0.3)
    plt.savefig(plotdir / ("gevp_corr_" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return


def plotting_script_unpert(
    correlator1,
    correlator2,
    ratio,
    fitvals1,
    fitvals2,
    fitvals,
    fitvals_effratio,
    nucl_t_range,
    sigma_t_range,
    ratio_t_range,
    plotdir,
    redchisqs,
    name="",
    show=False,
):
    spacing = 2
    xlim = 28
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

    plt.figure(figsize=(5, 5))
    plt.errorbar(
        efftime[:xlim],
        yavg_effratio[:xlim],
        ystd_effratio[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
    )
    plt.plot(
        ratio_t_range,
        len(ratio_t_range) * [np.average(fitvals_effratio)],
        color=_colors[0],
    )
    plt.fill_between(
        ratio_t_range,
        np.average(fitvals_effratio) - np.std(fitvals_effratio),
        np.average(fitvals_effratio) + np.std(fitvals_effratio),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}",
        label=rf"$\Delta E(\lambda=0)$ = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}",
    )

    plt.legend(fontsize="x-small")
    # plt.ylabel(r"$\textrm{eff. energy}[G_n(\mathbf{p}')/G_{\Sigma}(\mathbf{0})]$")
    plt.ylabel(r"$\textrm{eff. energy}[G_n(\mathbf{0})/G_{\Sigma}(\mathbf{0})]$")
    plt.xlabel(r"$t/a$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    plt.ylim(-0.1, 0.1)
    plt.savefig(plotdir / ("unpert_effmass.pdf"), metadata=_metadata)

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        efftime[:xlim],
        yavg_1[:xlim],
        ystd_1[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[0]:.2f}"
    )
    plt.errorbar(
        efftime[:xlim],
        yavg_2[:xlim],
        ystd_2[:xlim],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt="s",
        markerfacecolor="none",
        # label=f"{redchisqs[1]:.2f}"
    )

    fit_energy_nucl = fitvals1["param"][:, 1]
    fit_redchisq_nucl = fitvals1["redchisq"]
    plt.plot(
        nucl_t_range,
        len(nucl_t_range) * [np.average(fit_energy_nucl)],
        color=_colors[0],
    )
    plt.fill_between(
        nucl_t_range,
        np.average(fit_energy_nucl) - np.std(fit_energy_nucl),
        np.average(fit_energy_nucl) + np.std(fit_energy_nucl),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{0}}) = {err_brackets(np.average(fitvals1),np.std(fitvals1))}$; $\chi^2_{{\textrm{{dof}}}} = {redchisqs[0]:.2f}$",
        label=rf"$E_N(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_nucl),np.std(fit_energy_nucl))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_nucl:.2f}$",
    )

    fit_energy_sigma = fitvals2["param"][:, 1]
    fit_redchisq_sigma = fitvals2["redchisq"]
    plt.plot(
        sigma_t_range,
        len(sigma_t_range) * [np.average(fit_energy_sigma)],
        color=_colors[1],
    )
    plt.fill_between(
        sigma_t_range,
        np.average(fit_energy_sigma) - np.std(fit_energy_sigma),
        np.average(fit_energy_sigma) + np.std(fit_energy_sigma),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{\Sigma}}(\mathbf{{0}}) = {err_brackets(np.average(fit_energy_sigma),np.std(fit_energy_sigma))}$; $\chi^2_{{\textrm{{dof}}}} = {fit_redchisq_sigma:.2f}$",
    )
    # plt.plot(
    #     1000,
    #     1,
    #     label=rf"$\Delta E = {err_brackets(np.average(fitvals_effratio),np.std(fitvals_effratio))}$",
    # )
    plt.legend(fontsize="x-small")
    plt.ylabel(r"$\textrm{Effective energy}$")
    plt.xlabel(r"$t/a$")
    plt.axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(0, 2))
    plt.xlim(0, xlim)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig(plotdir / ("unpert_energies.pdf"), metadata=_metadata)

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
        # label=r"$N$",
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
        # label=r"$\Sigma$",
    )
    axs[0].plot(
        nucl_t_range,
        len(nucl_t_range) * [np.average(fit_energy_nucl)],
        color=_colors[0],
    )
    axs[0].fill_between(
        nucl_t_range,
        np.average(fit_energy_nucl) - np.std(fit_energy_nucl),
        np.average(fit_energy_nucl) + np.std(fit_energy_nucl),
        alpha=0.3,
        color=_colors[0],
        # label=rf"$E_N(\mathbf{{p}}')$ = {err_brackets(np.average(fitvals1),np.std(fitvals1))}",
        label=rf"$E_N(\mathbf{{0}})$ = {err_brackets(np.average(fit_energy_nucl),np.std(fit_energy_nucl))}",
    )
    axs[0].plot(
        sigma_t_range,
        len(sigma_t_range) * [np.average(fit_energy_sigma)],
        color=_colors[1],
    )
    axs[0].fill_between(
        sigma_t_range,
        np.average(fit_energy_sigma) - np.std(fit_energy_sigma),
        np.average(fit_energy_sigma) + np.std(fit_energy_sigma),
        alpha=0.3,
        color=_colors[1],
        label=rf"$E_{{\Sigma}}(\mathbf{{0}})$ = {err_brackets(np.average(fit_energy_sigma),np.std(fit_energy_sigma))}",
    )
    axs[0].plot(
        1000,
        1,
        label=rf"$\Delta E$ = {err_brackets(np.average(fit_energy_sigma-fit_energy_nucl),np.std(fit_energy_sigma-fit_energy_nucl))}",
    )
    axs[0].legend(fontsize="x-small")

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
    axs[1].plot(
        ratio_t_range, len(ratio_t_range) * [np.average(fitvals)], color=_colors[0]
    )
    axs[1].fill_between(
        ratio_t_range,
        np.average(fitvals) - np.std(fitvals),
        np.average(fitvals) + np.std(fitvals),
        alpha=0.3,
        color=_colors[2],
        label=rf"Fit = ${err_brackets(np.average(fitvals),np.std(fitvals))}$; $\chi^2_{{\textrm{{dof}}}} = {redchisqs[2]:.2f}$",
    )

    # axs[0].axhline(y=0, color="k", alpha=0.3, linewidth=0.5)
    # plt.setp(axs, xlim=(0, xlim), ylim=(-0.4, 0.4))
    plt.setp(axs, xlim=(0, xlim), ylim=(0, 2))
    axs[0].set_ylabel(r"$\textrm{Effective energy}$")
    # axs[1].set_ylabel(r"$G_n(\mathbf{p}')/G_{\Sigma}(\mathbf{0})$")
    axs[1].set_ylabel(r"$G_n(\mathbf{0})/G_{\Sigma}(\mathbf{0})$")
    plt.xlabel("$t/a$")
    axs[1].legend(fontsize="x-small")
    # plt.title("$\lambda=" + str(lmb_val) + "$")
    plt.savefig(plotdir / ("unpert_ratio" + name + ".pdf"), metadata=_metadata)
    if show:
        plt.show()
    plt.close()
    return
