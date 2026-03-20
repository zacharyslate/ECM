import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


def _style_axes(ax, tick_fs=10):
    ax.grid(which="major", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.grid(which="minor", linestyle=":", linewidth=0.45, alpha=0.35)
    ax.tick_params(axis="both", labelsize=tick_fs, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _safe_limits(arr, pad=0.05, force_zero_min=False):
    arr = np.asarray(arr, dtype=float)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if amin == amax:
        delta = 1.0 if amin == 0 else abs(amin) * 0.1
        amin -= delta
        amax += delta

    span = amax - amin
    amin -= span * pad
    amax += span * pad

    if force_zero_min:
        amin = min(0, amin)

    return amin, amax


def plot_impedance_results(frequencies, Z, Z_fit, major_ticks=500):
    """
    Publication-style 2x2 impedance summary:
    a) Nyquist
    b) Bode magnitude
    c) Bode phase
    d) Residuals
    """
    frequencies = np.asarray(frequencies)
    Z = np.asarray(Z)
    Z_fit = np.asarray(Z_fit)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    label_fs = 13
    tick_fs = 10
    legend_fs = 10
    panel_fs = 13
    title_fs = 11

    # -------------------------
    # a) Nyquist
    # -------------------------
    ax = axs[0, 0]
    ax.scatter(Z.real, -Z.imag, s=32, label="Data")
    ax.plot(Z_fit.real, -Z_fit.imag, "-", linewidth=2.0, label="Model fit")

    xlim = _safe_limits(Z.real, pad=0.06, force_zero_min=True)
    ylim = _safe_limits(-Z.imag, pad=0.06, force_zero_min=True)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # b) Bode magnitude
    # -------------------------
    ax = axs[0, 1]
    ax.scatter(frequencies, np.abs(Z), s=28, label="Data")
    ax.plot(frequencies, np.abs(Z_fit), "-", linewidth=2.0, label="Model fit")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$|Z|$ / $\Omega$", fontsize=label_fs)
    ax.set_title("Bode magnitude", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # c) Bode phase
    # -------------------------
    ax = axs[1, 0]
    phase_data = -np.angle(Z, deg=True)
    phase_fit = -np.angle(Z_fit, deg=True)

    ax.scatter(frequencies, phase_data, s=28, label="Data")
    ax.plot(frequencies, phase_fit, "-", linewidth=2.0, label="Model fit")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$-\phi$ / °", fontsize=label_fs)
    ax.set_title("Bode phase", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # d) Residuals
    # -------------------------
    ax = axs[1, 1]

    res_real_pct = 100 * (Z.real - Z_fit.real) / np.maximum(np.abs(Z.real), 1e-30)
    res_imag_pct = 100 * ((-Z.imag) - (-Z_fit.imag)) / np.maximum(np.abs(-Z.imag), 1e-30)

    ax.scatter(frequencies, res_real_pct, s=24, label=r"$Z^\prime$ residual")
    ax.scatter(frequencies, res_imag_pct, s=24, label=r"$-Z^{\prime\prime}$ residual")
    ax.plot(frequencies, res_real_pct, "-", linewidth=1.6)
    ax.plot(frequencies, res_imag_pct, "-", linewidth=1.6)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel("Residual / %", fontsize=label_fs)
    ax.set_title("Residuals", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    return fig


def plot_impedance_results_zoomable(frequencies, Z, Z_fit, major_ticks=500):
    """
    Streamlit-friendly version optimized for on-screen viewing.
    """
    frequencies = np.asarray(frequencies)
    Z = np.asarray(Z)
    Z_fit = np.asarray(Z_fit)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    label_fs = 11
    tick_fs = 9
    legend_fs = 9
    panel_fs = 12
    title_fs = 10

    # -------------------------
    # a) Nyquist
    # -------------------------
    ax = axs[0, 0]
    ax.scatter(Z.real, -Z.imag, s=20, label="Data")
    ax.plot(Z_fit.real, -Z_fit.imag, "-", linewidth=1.8, label="Model fit")

    xlim = _safe_limits(Z.real, pad=0.06, force_zero_min=True)
    ylim = _safe_limits(-Z.imag, pad=0.06, force_zero_min=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # b) Bode magnitude
    # -------------------------
    ax = axs[0, 1]
    ax.scatter(frequencies, np.abs(Z), s=18, label="Data")
    ax.plot(frequencies, np.abs(Z_fit), "-", linewidth=1.8, label="Model fit")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$|Z|$ / $\Omega$", fontsize=label_fs)
    ax.set_title("Bode magnitude", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # c) Bode phase
    # -------------------------
    ax = axs[1, 0]
    phase_data = -np.angle(Z, deg=True)
    phase_fit = -np.angle(Z_fit, deg=True)

    ax.scatter(frequencies, phase_data, s=18, label="Data")
    ax.plot(frequencies, phase_fit, "-", linewidth=1.8, label="Model fit")

    ax.set_xscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel(r"$-\phi$ / °", fontsize=label_fs)
    ax.set_title("Bode phase", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "c", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    # -------------------------
    # d) Residuals
    # -------------------------
    ax = axs[1, 1]
    res_real_pct = 100 * (Z.real - Z_fit.real) / np.maximum(np.abs(Z.real), 1e-30)
    res_imag_pct = 100 * ((-Z.imag) - (-Z_fit.imag)) / np.maximum(np.abs(-Z.imag), 1e-30)

    ax.scatter(frequencies, res_real_pct, s=16, label=r"$Z^\prime$ residual")
    ax.scatter(frequencies, res_imag_pct, s=16, label=r"$-Z^{\prime\prime}$ residual")
    ax.plot(frequencies, res_real_pct, "-", linewidth=1.4)
    ax.plot(frequencies, res_imag_pct, "-", linewidth=1.4)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="black", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency / Hz", fontsize=label_fs)
    ax.set_ylabel("Residual / %", fontsize=label_fs)
    ax.set_title("Residuals", fontsize=title_fs)
    ax.legend(fontsize=legend_fs, frameon=False)
    _style_axes(ax, tick_fs=tick_fs)
    ax.text(-0.12, 1.04, "d", transform=ax.transAxes, fontsize=panel_fs, fontweight="bold")

    return fig
