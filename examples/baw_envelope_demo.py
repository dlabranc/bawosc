import numpy as np
import matplotlib.pyplot as plt

from bawosc import BAWMode, add_baw_info_box

t = np.linspace(0, 200e-6, 20001)

baw = BAWMode(omega_lambda=2*np.pi*5e6, Q=1e7, k_lambda=1e-2)

f_drive = 5e6
omega_d = 2*np.pi*f_drive
h = 1e-21 * np.cos(omega_d * t)

A_t, V_t, I_env, f_drive_t, F_env = baw.solve_current_envelope(
    t, h, omega_d_of_t=omega_d*np.ones_like(t), use_hilbert=True
)

fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(t*1e6, h)
ax[0].set_ylabel(r"$h_+(t)$")
ax[1].plot(t*1e6, f_drive_t)
ax[1].set_ylabel(r"$f_{\mathrm{drive}}(t)$")
ax[2].plot(t*1e6, np.real(I_env), label="Re(I_env)")
ax[2].plot(t*1e6, np.abs(I_env), label="|I_env|", lw=2)
ax[2].set_ylabel(r"$I_{\mathrm{env}}(t)$")
ax[2].set_xlabel(r"time [$\mu$s]")
ax[2].legend()

add_baw_info_box(ax[0], baw, show_f_lambda=True, show_Q=True, show_tau=True)

plt.tight_layout()
plt.show()
