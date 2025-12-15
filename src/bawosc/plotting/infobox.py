import numpy as np
from matplotlib.offsetbox import AnchoredText

def add_baw_info_box(
    ax,
    baw,
    *,
    show_n=False,
    show_d=False,
    show_eta=False,
    show_omega_lambda=False,
    show_f_lambda=True,
    show_Q=True,
    show_gamma_lambda=False,
    show_tau=True,
    show_k_lambda=False,
    show_drive=False,
    show_detuning=True,
    show_duration=False,
    f_drive_hz=None,
    f_start_hz=None,
    f_end_hz=None,
    T=None,
    loc="upper right",
    fontsize=11,
    alpha=0.85,
    boxstyle="round,pad=0.3",
    sep=" | ",
):
    """
    Compact info box for your BAWMode object + optional drive info.
    Preserves your formatting and toggles.
    """
    parts = [r"$\bf{BAW:}$"]

    omega_l = baw.omega_lambda
    f_l = omega_l / (2*np.pi)

    if show_n:
        parts.append(rf"$n={baw.n}$")
    if show_d:
        parts.append(rf"$d={baw.d*1e3:.3g}\ \mathrm{{mm}}$")
    if show_eta:
        if np.isclose(baw.eta_x, baw.eta_y):
            parts.append(rf"$\eta_x=\eta_y={baw.eta_x:.3g}$")
        else:
            parts.append(rf"$\eta_x={baw.eta_x:.3g},\ \eta_y={baw.eta_y:.3g}$")

    if show_omega_lambda:
        parts.append(rf"$\omega_\lambda={omega_l:.6g}\ \mathrm{{rad/s}}$")
    if show_f_lambda:
        parts.append(rf"$\omega_\lambda/2\pi={f_l*1e-6:.6g}\ \mathrm{{MHz}}$")
    if show_Q:
        parts.append(rf"$Q={baw.Q:.2g}$")
    if show_gamma_lambda:
        parts.append(rf"$\gamma_\lambda={baw.gamma_lambda:.3g}\ \mathrm{{Hz}}$")
    if show_tau:
        parts.append(rf"$\tau={1/baw.gamma_lambda:.2e}\ \mathrm{{s}}$")
    if show_k_lambda:
        parts.append(rf"$k_\lambda={baw.k_lambda:.2g}$")

    if show_drive:
        parts.append(r"$\bf{Drive:}$")

        if (f_start_hz is not None) and (f_end_hz is not None):
            parts.append(
                rf"$f_d:{f_start_hz*1e-6:.6g}\!\rightarrow\!{f_end_hz*1e-6:.6g}\ \mathrm{{MHz}}$"
            )
            if show_duration and (T is not None):
                parts.append(rf"$T={T*1e6:.3g}\ \mu s$")

        elif f_drive_hz is not None:
            parts.append(rf"$f_d={f_drive_hz*1e-6:.6g}\ \mathrm{{MHz}}$")
            if show_detuning:
                df = f_drive_hz - f_l
                parts.append(rf"$\Delta f={df*1e-3:.2g}\ \mathrm{{kHz}}$")
            if show_duration and (T is not None):
                parts.append(rf"$T={T*1e6:.3g}\ \mu s$")

    text = sep.join(parts)

    at = AnchoredText(text, loc=loc, prop=dict(size=fontsize), frameon=True)
    at.patch.set_boxstyle(boxstyle)
    at.patch.set_alpha(alpha)
    at.patch.set_linewidth(0.8)
    ax.add_artist(at)
    return at
