import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

# =========================
# TURBO-LIKE COLORS
# =========================
TURBO = plt.get_cmap("turbo")

DATA_COLOR = TURBO(0.15)   # blue-cyan
FIT_COLOR  = TURBO(0.78)   # orange-yellow
RES_REAL   = TURBO(0.05)   # deep purple
RES_IMAG   = TURBO(0.92)   # red-orange

# =========================
# GLOBAL STYLE
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def _style_axes(ax, tick_fs=9):
    ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.grid(which="minor", linestyle=":", linewidth=0.45, alpha=0.30)
    ax.tick_params(axis="both", labelsize=tick_fs, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _safe_limits(arr, pad=0.05):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]

    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if amin == amax:
        delta = 1.0 if amin == 0 else abs(amin) * 0.1
        amin -= delta
        amax += delta

    span = amax - amin
    amin -= span * pad
    amax += span * pad

    return amin, amax


def _apply_nyquist_format(ax, Z, Z_fit, major_ticks, label_fs, legend_fs, title_fs, panel_fs):
    # plot fit first so data sits on top
    ax.plot(Z_fit.real, -Z_fit.imag, "-", linewidth=2.4, label="Model fit",
            color=FIT_COLOR, zorder=2)
    ax.scatter(Z.real, -Z.imag, s=26, label="Data",
               color=DATA_COLOR, zorder=3)

    # use BOTH data and fit to determine limits
    all_x = np.concatenate([Z.real, Z_fit.real])
    all_y = np.concatenate([-Z.imag, -Z_fit.imag])

    ax.set_xlim(*_safe_limits(all_x, pad=0.08))
    ax.set_ylim(*_safe_limits(all_y, pad=0.08))

    # keep Nyquist physically correct without shrinking weirdly
    ax.set_aspect("equal", adjustable="box")

    if major_ticks is not None and major_ticks > 0:
        ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
        ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))

    ax.set_xlabel(r"$Z^\prime$ / $\Omega$", fontsize=label_fs)
    ax.set_ylabel(r"$-Z^{\prime\prime}$ / $\Omega$", fontsize=label_fs)
    ax.set_title("Nyquist", fontsize=title_fs)

    # legend order: Data first, then Model fit
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=legend_fs, frameon=False, loc="best")

    _style_axes(ax)
    ax.text(-0.12, 1.05, "a", transform=ax.transAxes,
            fontsize=panel_fs, fontweight="bold")


def _apply_bode_mag_format(ax, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs):
    ax.plot(frequencies, np.abs(Z_fit), "-", linewidth=2.4, label="Model fit",
            color=FIT_COLOR, zorder=2)
    ax.scatter(frequencies, np.abs(Z), s=22, label="Data",
               color=DATA_COLOR, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_box_aspect(0.92)

    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$|Z|$ / $\Omega$", fontsize=label_fs)
    ax.set_title("Bode magnitude", fontsize=title_fs)

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=legend_fs, frameon=False, loc="best")

    _style_axes(ax)
    ax.text(-0.12, 1.05, "b", transform=ax.transAxes,
            fontsize=panel_fs, fontweight="bold")


def _apply_bode_phase_format(ax, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs):
    phase_data = -np.angle(Z, deg=True)
    phase_fit = -np.angle(Z_fit, deg=True)

    ax.plot(frequencies, phase_fit, "-", linewidth=2.4, label="Model fit",
            color=FIT_COLOR, zorder=2)
    ax.scatter(frequencies, phase_data, s=22, label="Data",
               color=DATA_COLOR, zorder=3)

    ax.set_xscale("log")
    ax.set_box_aspect(0.92)

    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$-\phi$ / °", fontsize=label_fs)
    ax.set_title("Bode phase", fontsize=title_fs)

    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=legend_fs, frameon=False, loc="best")

    _style_axes(ax)
    ax.text(-0.12, 1.05, "c", transform=ax.transAxes,
            fontsize=panel_fs, fontweight="bold")


def _apply_residual_format(ax, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs):
    res_real = (Z - Z_fit).real
    res_imag = -(Z - Z_fit).imag

    ax.plot(frequencies, res_real, "-", linewidth=1.5, color=RES_REAL, zorder=2)
    ax.plot(frequencies, res_imag, "-", linewidth=1.5, color=RES_IMAG, zorder=2)

    ax.scatter(frequencies, res_real, s=18, label=r"$Z^\prime$ residual",
               color=RES_REAL, zorder=3)
    ax.scatter(frequencies, res_imag, s=18, label=r"$-Z^{\prime\prime}$ residual",
               color=RES_IMAG, zorder=3)

    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax.set_xscale("log")
    ax.set_box_aspect(0.92)

    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$\Delta$ / $\Omega$", fontsize=label_fs)
    ax.set_title("Residuals", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False, loc="best")

    _style_axes(ax)
    ax.text(-0.12, 1.05, "d", transform=ax.transAxes,
            fontsize=panel_fs, fontweight="bold")


def plot_impedance_results(frequencies, Z, Z_fit, major_ticks=None):
    """
    Publication-style 2x2 plot with improved spacing and Nyquist scaling.
    """
    frequencies = np.asarray(frequencies)
    Z = np.asarray(Z)
    Z_fit = np.asarray(Z_fit)

    fig = plt.figure(figsize=(10.2, 7.0), dpi=150)
    gs = fig.add_gridspec(
        2, 2,
        left=0.08, right=0.98, bottom=0.09, top=0.94,
        wspace=0.18, hspace=0.34
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    label_fs = 12
    legend_fs = 10
    title_fs = 11
    panel_fs = 13

    _apply_nyquist_format(ax1, Z, Z_fit, major_ticks, label_fs, legend_fs, title_fs, panel_fs)
    _apply_bode_mag_format(ax2, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)
    _apply_bode_phase_format(ax3, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)
    _apply_residual_format(ax4, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)

    return fig


def plot_impedance_results_zoomable(frequencies, Z, Z_fit, major_ticks=None):
    """
    Streamlit-friendly fixed-size figure with tighter layout.
    """
    frequencies = np.asarray(frequencies)
    Z = np.asarray(Z)
    Z_fit = np.asarray(Z_fit)

    fig = plt.figure(figsize=(9.6, 6.8), dpi=150)
    gs = fig.add_gridspec(
        2, 2,
        left=0.08, right=0.98, bottom=0.10, top=0.93,
        wspace=0.16, hspace=0.30
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    label_fs = 10
    legend_fs = 9
    title_fs = 10
    panel_fs = 12

    _apply_nyquist_format(ax1, Z, Z_fit, major_ticks, label_fs, legend_fs, title_fs, panel_fs)
    _apply_bode_mag_format(ax2, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)
    _apply_bode_phase_format(ax3, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)
    _apply_residual_format(ax4, frequencies, Z, Z_fit, label_fs, legend_fs, title_fs, panel_fs)

    return fig
