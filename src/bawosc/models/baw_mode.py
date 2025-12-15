import numpy as np
from scipy.signal import hilbert
from math import erf, sqrt, pi

from .driven_oscillator import DrivenHarmonicOscillator

# ---------------------------------------------------------------
# BAW mode class
# ---------------------------------------------------------------
class BAWMode:
    """
    Represents a single bulk-acoustic-wave (BAW) mode coupled to a
    gravitational strain h_+(t).
    """

    def __init__(
        self,
        n=3,
        d=1e-3,
        eta_x=0.1,
        eta_y=0.1,
        omega_lambda=2 * np.pi * 5e6,
        Q=1e7,
        k_lambda=1e-2,
    ):
        self.n = n
        self.d = d
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.omega_lambda = omega_lambda
        self.Q = Q
        self.k_lambda = k_lambda

        self.gamma_lambda = omega_lambda / Q

    def baw_specs_text(self):
        lines = []
        for name, value in vars(self).items():
            if not name.startswith("_"):
                lines.append(f"{name}: {value}")
        return "\n".join(lines)

    def xi_lambda(self):
        n = self.n
        ex = self.eta_x
        ey = self.eta_y
        d  = self.d

        num = erf(sqrt(n) * ex) * erf(sqrt(n) * ey)
        den = erf(sqrt(2 * n) * ex) * erf(sqrt(2 * n) * ey)
        return (4.0 * d / (n * pi)) * (num / den)

    def solve_current(self, t, h_plus, use_green=True):
        dt = t[1] - t[0]
        xi = self.xi_lambda()

        h_dot  = np.gradient(h_plus, dt)
        h_ddot = np.gradient(h_dot, dt)

        f_drive = 0.5 * xi * h_ddot

        osc = DrivenHarmonicOscillator(
            m=1.0,
            gamma=self.gamma_lambda,
            omega0=self.omega_lambda,
        )

        if use_green:
            B = osc.solve_via_green(t, f_drive)
        else:
            B = osc.solve_direct_ode(t, f_drive)

        Bdot = np.gradient(B, dt)
        I = self.k_lambda * Bdot

        return B, Bdot, I, f_drive

    def solve_current_envelope(
        self,
        t,
        h_plus,
        omega_d_of_t=None,
        omega_c=None,
        A0=0.0 + 0.0j,
        A1=0.0 + 0.0j,
        use_hilbert=True,
        rescale=True,
    ):
        t = np.asarray(t, dtype=float)
        h_plus = np.asarray(h_plus, dtype=float)
        dt = t[1] - t[0]

        if omega_c is None:
            omega_c = self.omega_lambda

        if omega_d_of_t is None:
            omega_d_of_t = omega_c * np.ones_like(t)
        else:
            omega_d_of_t = np.asarray(omega_d_of_t, dtype=float)
            if omega_d_of_t.ndim == 0:
                omega_d_of_t = float(omega_d_of_t) * np.ones_like(t)
            if omega_d_of_t.shape != t.shape:
                raise ValueError("omega_d_of_t must be None, scalar, or same shape as t")

        omega0 = self.omega_lambda
        gamma  = self.gamma_lambda
        xi     = self.xi_lambda()

        if use_hilbert:
            a_t = np.abs(hilbert(h_plus))
        else:
            a_t = np.abs(h_plus)

        a_dot  = np.gradient(a_t, dt)
        a_ddot = np.gradient(a_dot, dt)
        omega_dot = np.gradient(omega_d_of_t, dt)

        theta = np.zeros_like(t)
        theta[1:] = np.cumsum(0.5 * (omega_d_of_t[1:] + omega_d_of_t[:-1]) * np.diff(t))

        c = np.cos(theta)
        s = np.sin(theta)

        h_ddot = (
            a_ddot * c
            - (2.0 * a_dot * omega_d_of_t + a_t * omega_dot) * s
            - a_t * (omega_d_of_t**2) * c
        )

        f_drive = 0.5 * xi * h_ddot

        if use_hilbert:
            f_plus = hilbert(f_drive)
        else:
            f_plus = f_drive.astype(complex)

        F_env = f_plus * np.exp(+1j * theta)

        scale = 1.0
        if rescale:
            scale = np.max(np.abs(F_env))
            if scale == 0.0:
                scale = 1.0
            F_env = F_env / scale
            A0 = A0 / scale
            A1 = A1 / scale

        osc = DrivenHarmonicOscillator(m=1.0, gamma=gamma, omega0=omega0)
        A_t, V_t = osc.solve_envelope_green(
            t=t,
            B_of_t=F_env,
            omega_d_of_t=omega_d_of_t,
            omega_c=omega_c,
            A0=A0,
            A1=A1,
            return_V=True,
            rtol=1e-8,
            atol=1e-11
        )

        if rescale:
            A_t = scale * A_t
            V_t = scale * V_t
            F_env = scale * F_env

        I_env = self.k_lambda * (V_t - 1j * omega_c * A_t)
        return A_t, V_t, I_env, f_drive, F_env

    def linear_chirp_strain(
        self,
        t,
        A0=1e-21,
        f_start=4.5e6,
        f_end=5.5e6,
        phi0=0.0,
        envelope=None,
    ):
        t = np.asarray(t)
        dt = t[1] - t[0]
        t0 = t[0]
        T = t[-1] - t0

        f_t = f_start + (f_end - f_start) * (t - t0) / T
        omega_t = 2 * np.pi * f_t

        phi = np.cumsum(omega_t) * dt
        phi -= phi[0]

        if envelope is None:
            env = np.ones_like(t)
        elif callable(envelope):
            env = np.asarray(envelope(t))
        else:
            env = np.asarray(envelope)
            if env.shape != t.shape:
                raise ValueError("envelope array must have the same shape as t")

        return A0 * env * np.cos(phi + phi0)
