import matplotlib.pyplot as plt
import numpy as np
import mplcursors

def plot_impedance_results(frequencies, Z, Z_fit, major_ticks):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    label_fs = 14
    tick_fs = 11
    legend_fs = 11
    panel_fs = 13

    # Nyquist Plot
    axs[0, 0].scatter(Z.real, -Z.imag, label='Data', s=45, color="darkcyan")
    axs[0, 0].plot(Z_fit.real, -Z_fit.imag, '-', label='Model Fit', color="coral", linewidth=2.5)
    axs[0, 0].set_xlabel(r'$Z_{re}$ / $\Omega$', fontsize=label_fs)
    axs[0, 0].set_ylabel(r'-$Z_{im}$ / $\Omega$', fontsize=label_fs)
    axs[0, 0].legend(fontsize=legend_fs)
    axs[0, 0].set_xlim(left=min(0, np.min(Z.real) * 1.05))
    axs[0, 0].set_ylim(bottom=min(0, np.min(-Z.imag) * 1.05))
    axs[0, 0].xaxis.set_major_locator(plt.MultipleLocator(major_ticks))
    axs[0, 0].yaxis.set_major_locator(plt.MultipleLocator(major_ticks))
    axs[0, 0].grid(which="both", linestyle='--', linewidth=0.8, alpha=0.7)
    axs[0, 0].tick_params(axis='both', labelsize=tick_fs)
    axs[0, 0].text(-0.10, -0.18, 'a', transform=axs[0, 0].transAxes, fontsize=panel_fs, va='top', ha='left')

    # Bode Magnitude Plot
    axs[0, 1].scatter(frequencies, np.abs(Z), label='Data |Z|', s=45, color="darkcyan")
    axs[0, 1].plot(frequencies, np.abs(Z_fit), '-', label='Model Fit', color="coral", linewidth=2.5)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Frequency / Hz', fontsize=label_fs)
    axs[0, 1].set_ylabel(r'$|Z|$ / $\Omega$', fontsize=label_fs)
    axs[0, 1].legend(fontsize=legend_fs)
    axs[0, 1].grid(which="both", linestyle='--', linewidth=0.8, alpha=0.7)
    axs[0, 1].tick_params(axis='both', labelsize=tick_fs)
    axs[0, 1].text(-0.10, -0.18, 'b', transform=axs[0, 1].transAxes, fontsize=panel_fs, va='top', ha='left')

    # Bode Phase Plot
    axs[1, 0].scatter(frequencies, -np.angle(Z, deg=True), label='Data Phase', s=45, color="darkcyan")
    axs[1, 0].plot(frequencies, -np.angle(Z_fit, deg=True), '-', label='Model Fit', color="coral", linewidth=2.5)
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel('Frequency / Hz', fontsize=label_fs)
    axs[1, 0].set_ylabel(r'-$\Phi(\omega)$ / °', fontsize=label_fs)
    axs[1, 0].legend(fontsize=legend_fs)
    axs[1, 0].grid(which="both", linestyle='--', linewidth=0.8, alpha=0.7)
    axs[1, 0].tick_params(axis='both', labelsize=tick_fs)
    axs[1, 0].text(-0.10, -0.18, 'c', transform=axs[1, 0].transAxes, fontsize=panel_fs, va='top', ha='left')

    # Residual Plot
    res_meas_real = (Z - Z_fit).real
    res_meas_imag = -(Z - Z_fit).imag

    axs[1, 1].scatter(frequencies, res_meas_real, label=r"$Z_{re}$ residual", s=40, color="slateblue")
    axs[1, 1].scatter(frequencies, res_meas_imag, label=r"$-Z_{im}$ residual", s=40, color="mediumvioletred")
    axs[1, 1].plot(frequencies, res_meas_real, linestyle='-', color="slateblue", linewidth=2)
    axs[1, 1].plot(frequencies, res_meas_imag, linestyle='-', color="mediumvioletred", linewidth=2)
    axs[1, 1].axhline(0, linestyle='--', linewidth=1, color='black', alpha=0.6)
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Frequency / Hz', fontsize=label_fs)
    axs[1, 1].set_ylabel(r'$\Delta$ / $\Omega$', fontsize=label_fs)
    axs[1, 1].legend(fontsize=legend_fs)
    axs[1, 1].grid(which="both", linestyle='--', linewidth=0.8, alpha=0.7)
    axs[1, 1].tick_params(axis='both', labelsize=tick_fs)
    axs[1, 1].text(-0.10, -0.18, 'd', transform=axs[1, 1].transAxes, fontsize=panel_fs, va='top', ha='left')

    for ax in axs.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(0)

    return fig

def plot_impedance_results_zoomable(frequencies, Z, Z_fit, major_ticks):
    # Use the TkAgg backend for interactive zoom functionality
    plt.switch_backend('tkagg')

    fig, axs = plt.subplots(2, 2, figsize=(6, 4))

    # Nyquist Plot
    scatter_nyquist = axs[0, 0].scatter(Z.real, -Z.imag, label='Data', s=10, color="darkcyan")
    axs[0, 0].plot(Z_fit.real, -Z_fit.imag, '-', label='Model Fit', color="coral")
    axs[0, 0].set_xlabel(r'$Z_{re}$ / $\Omega$', fontsize=8)
    axs[0, 0].set_ylabel(r'-$Z_{im}$ / $\Omega$', fontsize=8)
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].grid(which="both", linestyle='--', linewidth=0.5)
    axs[0, 0].text(-0.2, 1.05, 'a', transform=axs[0, 0].transAxes, fontsize=12, va='top', ha='left')

    # Bode Magnitude Plot
    scatter_bode_mag = axs[0, 1].scatter(frequencies, np.abs(Z), label='Data |Z|', s=10, color="darkcyan")
    axs[0, 1].plot(frequencies, np.abs(Z_fit), '-', label='Model Fit', color="coral")
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Frequency / Hz', fontsize=8)
    axs[0, 1].set_ylabel(r'|Z| / $\Omega$', fontsize=8)
    axs[0, 1].legend(fontsize=7)
    axs[0, 1].grid(which="both", linestyle='--', linewidth=0.5)
    axs[0, 1].text(-0.2, 1.05, 'b', transform=axs[0, 1].transAxes, fontsize=12, va='top', ha='left')

    # Bode Phase Plot
    scatter_bode_phase = axs[1, 0].scatter(frequencies, -np.angle(Z, deg=True), label='Data Phase', s=10, color="darkcyan")
    axs[1, 0].plot(frequencies, -np.angle(Z_fit, deg=True), '-', label='Model Fit', color="coral")
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_xlabel('Frequency / Hz', fontsize=8)
    axs[1, 0].set_ylabel(r'-$\Phi$ ($\omega$) / °', fontsize=8)
    axs[1, 0].legend(fontsize=7)
    axs[1, 0].grid(which="both", linestyle='--', linewidth=0.5)
    axs[1, 0].text(-0.2, 1.05, 'c', transform=axs[1, 0].transAxes, fontsize=12, va='top', ha='left')

    # Residual Plot
    res_meas_real = (Z - Z_fit).real
    res_meas_imag = (Z - Z_fit).imag
    scatter_residual = axs[1, 1].scatter(frequencies, res_meas_real, label=r"$Z_{re}$ residual", s=10, color="slateblue")
    axs[1, 1].scatter(frequencies, res_meas_imag, label=r"$-Z_{im}$ residual", s=10, color="mediumvioletred")
    axs[1, 1].plot(frequencies, res_meas_real, linestyle='-', color="slateblue")
    axs[1, 1].plot(frequencies, res_meas_imag, linestyle='-', color="mediumvioletred")
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_xlabel('Frequency / Hz', fontsize=8)
    axs[1, 1].set_ylabel(r'$\Delta$', fontsize=8)
    axs[1, 1].legend(fontsize=7)
    axs[1, 1].grid(which="both", linestyle='--', linewidth=0.5)
    axs[1, 1].text(-0.2, 1.05, 'd', transform=axs[1, 1].transAxes, fontsize=12, va='top', ha='left')

    # Adding cursors for interactive exploration
    mplcursors.cursor(scatter_nyquist, hover=2).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Re: {sel.target[0]:.2f}\n-Im: {sel.target[1]:.2f}"
        )
    )
    mplcursors.cursor(scatter_bode_mag, hover=2).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Frequency: {sel.target[0]:.2e} Hz\n|Z|: {sel.target[1]:.2f}"
        )
    )
    mplcursors.cursor(scatter_bode_phase, hover=2).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Frequency: {sel.target[0]:.2e} Hz\nPhase: {sel.target[1]:.2f}°"
        )
    )
    mplcursors.cursor(scatter_residual, hover=2).connect(
        "add", lambda sel: sel.annotation.set_text(
            f"Frequency: {sel.target[0]:.2e} Hz\nResidual: {sel.target[1]:.2f}%"
        )
    )

    fig.tight_layout()

    # Show plot with interactive zooming
    plt.show()
