import numpy as np
from scipy.integrate import simpson
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------
# Simple Driven Harmonic Oscillator Class
# ---------------------------------------------------------------
class DrivenHarmonicOscillator:
    """
    Driven harmonic oscillator:
        x'' + 2 γ x' + ω0^2 x = f(t)/m

    Solved using the causal Green’s function:
        G(t) = Θ(t) e^{-γ t} sin(ω_d t) / (m ω_d)
    """

    def __init__(self, m=1.0, gamma=0.2, omega0=1.0):
        self.m = m
        self.gamma = gamma
        self.omega0 = omega0
        self.omega_d = np.sqrt(abs(omega0**2 - (gamma**2)) )

    def green(self, t):
        """
        Causal Green's function G(t).
        """
        G = np.zeros_like(t)
        mask = t >= 0
        if self.omega0 > self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                * np.sin(self.omega_d * t[mask])
                / (self.m * self.omega_d)
            )
        elif self.omega0 < self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                * np.sinh(self.omega_d * t[mask])
                / (self.m * self.omega_d)
            )
        elif self.omega0 == self.gamma:
            G[mask] = (
                np.exp(-self.gamma * t[mask] )
                *  t[mask]
                / (self.m )
            )
        return G

    def solve_via_green(self, t, f_of_t):
        """
        Computes x(t) = ∫ G(t - t') f(t') dt'
        where f_of_t is an array f(t).
        """
        dt = t[1] - t[0]
        x = np.zeros_like(t)
        G = self.green  # alias

        for i in range(len(t)):
            tau = t[i] - t[: i + 1]      # only integrate where t' <= t
            integrand = G(tau) * f_of_t[: i + 1]
            x[i] = simpson(integrand, t[: i + 1])
        return x

    def solve_direct_ode(self, t, f_of_t):
        """
        Solves the same oscillator via ODE integration (verification).
        """

        def f_interp(t_query):
            # simple linear interpolation of forcing array
            return np.interp(t_query, t, f_of_t)

        def ode(t, y):
            x, v = y
            dxdt = v
            dvdt = (
                -2*self.gamma * v
                - self.omega0**2 * x
                + f_interp(t) / self.m
            )
            return [dxdt, dvdt]

        sol = solve_ivp(ode, (t[0], t[-1]), y0=[0, 0], t_eval=t, rtol=1e-8, atol=1e-10)
        return sol.y[0]


    def solve_envelope_ode(self, t, B_of_t, omega_d, omega_c=None, A0=0.0 + 0.0j):
        """
        Solve the rotating-frame envelope equation for A(t):

            2(γ - i ω_c) Ȧ(t)
            - (ω0^2 - ω_c^2 - 2 i γ ω_c) A(t)
            = (B(t)/m) * exp(-i ∫Δ(t) dt),

        with Δ(t) = ω_d(t) - ω_c.
        """
        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d = np.asarray(omega_d)

        if omega_c is None:
            omega_c = self.omega0

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        # interpolation of B(t) and ω_d(t)
        def B_interp(tq):
            return np.interp(tq, t, B_of_t)

        def omega_d_interp(tq):
            return np.interp(tq, t, omega_d)

        # precompute the complex coefficients
        num_coeff = (omega0**2 - omega_c**2 - 2j * gamma * omega_c)
        den_coeff = 2.0 * (gamma - 1j * omega_c)

        # real-valued ODE for Re(A), Im(A)
        def ode(tq, y):
            A = y[0] + 1j * y[1]

            Delta_t = omega_d_interp(tq) - omega_c

            drive = (B_interp(tq) / m) * np.exp(-1j * Delta_t * tq)

            dA_dt = (-num_coeff * A + drive) / den_coeff

            return [dA_dt.real, dA_dt.imag]

        # initial condition
        y0 = [A0.real, A0.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        return A_t

    def solve_envelope_ode_varfreq(self, t, B_of_t, omega_d_of_t, omega_c=None, A0=0.0 + 0.0j):
        """
        Solve the rotating-frame envelope equation for A(t) with time-dependent drive:

            2(γ - i ω_c) Ȧ(t)
            - (ω0^2 - ω_c^2 - 2 i γ ω_c) A(t)
            = (B(t)/m) * exp(-i φ(t)),

        with φ(t) = ∫_{t0}^{t} Δ(τ) dτ and Δ(t) = ω_d(t) - ω_c.
        """
        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t)

        if omega_c is None:
            omega_c = self.omega0  # or self.omega_lambda in your BAW code

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        # detuning array Δ(t) = ω_d(t) - ω_c
        Delta_of_t = omega_d_of_t - omega_c

        # phase φ(t) = ∫ Δ(t) dt computed on the grid (trapezoidal rule)
        dt = np.diff(t)
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)

        # precompute the complex coefficients
        num_coeff = (omega0**2 - omega_c**2 - 2j * gamma * omega_c)
        den_coeff = 2.0 * (gamma - 1j * omega_c)

        # interpolation of B(t) so solver can query at arbitrary times
        def B_interp(tq):
            return np.interp(tq, t, B_of_t)

        # interpolation of φ(t)
        def phi_interp(tq):
            return np.interp(tq, t, phi)

        # real-valued ODE for Re(A), Im(A)
        def ode(tq, y):
            A = y[0] + 1j * y[1]

            # RHS drive term: (B(t)/m) * exp(-i φ(t))
            drive = (B_interp(tq) / m) * np.exp(-1j * phi_interp(tq))

            dA_dt = (-num_coeff * A + drive) / den_coeff

            return [dA_dt.real, dA_dt.imag]

        # initial condition in real form
        y0 = [A0.real, A0.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        return A_t


    def solve_envelope_ode_second_order(self, t, B_of_t, omega_d_of_t,
                                    omega_c=None,
                                    A0=0.0 + 0.0j,
                                    A1=0.0 + 0.0j,
                                    return_V=False,
                                    rtol=1e-8,
                                    atol=1e-10,):
        r"""
        Second-order envelope equation for A(t) with A¨ term:

            A¨(t) + alpha Ȧ(t) + beta A(t) = (B(t)/m) * exp[-i φ(t)],

        where φ(t) = ∫ Δ(t) dt and Δ(t) = ω_d(t) - ω_c.
        """

        t = np.asarray(t)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t)

        if omega_c is None:
            omega_c = self.omega0

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        Delta_of_t = omega_d_of_t - omega_c

        # phase φ(t) = ∫ Δ(t) dt on the grid
        dt = np.diff(t)
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)

        def B_interp(tq):
            Br = np.interp(tq, t, B_of_t.real)
            Bi = np.interp(tq, t, B_of_t.imag)
            return Br + 1j*Bi

        def phi_interp(tq):
            return np.interp(tq, t, phi)

        def ode(tq, y):
            A = y[0] + 1j * y[1]
            V = y[2] + 1j * y[3]

            phase = np.exp(-1j * phi_interp(tq))
            drive = (B_interp(tq) / m) * phase

            dA_dt = V
            dV_dt = -2 * (gamma - 1j*omega_c) * V                     - (omega0**2 - omega_c**2 - 2j*omega_c*gamma) * A                     + drive

            return [dA_dt.real, dA_dt.imag, dV_dt.real, dV_dt.imag]

        y0 = [A0.real, A0.imag, A1.real, A1.imag]

        sol = solve_ivp(
            ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            rtol=rtol,
            atol=atol,
        )

        A_t = sol.y[0] + 1j * sol.y[1]
        V_t = sol.y[2] + 1j * sol.y[3]

        return (A_t, V_t) if return_V else A_t

    def solve_envelope_green(
        self, t, B_of_t, omega_d_of_t,
        omega_c=None,
        A0=0.0 + 0.0j,
        A1=0.0 + 0.0j,
        return_V=False,
        rtol=1e-8,
        atol=1e-10,
    ):
        r"""
        Fast second-order envelope solver via causal Green's function (FFT convolution).

        Notes
        -----
        - Works with complex B_of_t (envelope with quadrature).
        - Works with time-dependent omega_d_of_t via φ(t).
        - Requires *uniform* t for the FFT path; otherwise falls back to solve_ivp.
        """
        import numpy as np
        from scipy.integrate import solve_ivp  # fallback only

        t = np.asarray(t, dtype=float)
        B_of_t = np.asarray(B_of_t)
        omega_d_of_t = np.asarray(omega_d_of_t, dtype=float)

        if omega_c is None:
            omega_c = self.omega0

        gamma  = self.gamma
        omega0 = self.omega0
        m      = self.m

        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a 1D array with at least 2 points")
        if B_of_t.shape != t.shape:
            raise ValueError("B_of_t must have the same shape as t")
        if omega_d_of_t.shape != t.shape:
            raise ValueError("omega_d_of_t must have the same shape as t")

        dt = t[1] - t[0]
        uniform = np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-15)

        # ------------------------------------------------------------------
        # If grid is not uniform, fall back to your original ODE integrator.
        # ------------------------------------------------------------------
        if not uniform:
            Delta_of_t = omega_d_of_t - omega_c

            # trapezoidal phase on the grid (same convention as your original code)
            phi = np.zeros_like(t, dtype=float)
            phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * np.diff(t))

            def B_interp(tq):
                Br = np.interp(tq, t, B_of_t.real)
                Bi = np.interp(tq, t, B_of_t.imag)
                return Br + 1j * Bi

            def phi_interp(tq):
                return np.interp(tq, t, phi)

            def ode(tq, y):
                A = y[0] + 1j * y[1]
                V = y[2] + 1j * y[3]
                drive = (B_interp(tq) / m) * np.exp(-1j * phi_interp(tq))

                dA_dt = V
                dV_dt = (
                    -2 * (gamma - 1j * omega_c) * V
                    - (omega0**2 - omega_c**2 - 2j * omega_c * gamma) * A
                    + drive
                )
                return [dA_dt.real, dA_dt.imag, dV_dt.real, dV_dt.imag]

            y0 = [A0.real, A0.imag, A1.real, A1.imag]
            sol = solve_ivp(ode, (t[0], t[-1]), y0=y0, t_eval=t, rtol=rtol, atol=atol)
            A_t = sol.y[0] + 1j * sol.y[1]
            V_t = sol.y[2] + 1j * sol.y[3]
            return (A_t, V_t) if return_V else A_t

        # ------------------------------------------------------------------
        # FFT-Green fast path (uniform grid)
        # ------------------------------------------------------------------

        Delta_of_t = omega_d_of_t - omega_c
        phi = np.zeros_like(t, dtype=float)
        phi[1:] = np.cumsum(0.5 * (Delta_of_t[1:] + Delta_of_t[:-1]) * dt)

        # source term s(t) = (B(t)/m) e^{-i φ(t)}
        s = (B_of_t / m) * np.exp(-1j * phi)

        # Green's function for:
        #   A¨ + 2(γ - i ω_c) Ȧ + (ω0^2 - ω_c^2 - 2 i γ ω_c) A = δ(t)
        a = gamma - 1j * omega_c
        Omega = np.sqrt(omega0**2 - gamma**2 + 0j)

        tau = t - t[0]
        if np.abs(Omega) == 0:
            g = np.exp(-a * tau) * tau
            gdot = np.exp(-a * tau) * (1.0 - a * tau)
        else:
            g = np.exp(-a * tau) * (np.sin(Omega * tau) / Omega)
            gdot = np.exp(-a * tau) * (np.cos(Omega * tau) - (a / Omega) * np.sin(Omega * tau))

        # causal convolution via FFT
        N = t.size
        L = 1 << int(np.ceil(np.log2(2 * N - 1)))

        S = np.fft.fft(s, n=L)
        G = np.fft.fft(g, n=L)
        Gd = np.fft.fft(gdot, n=L)

        A_forced = dt * np.fft.ifft(S * G)[:N]
        V_forced = dt * np.fft.ifft(S * Gd)[:N]

        # homogeneous solution enforcing A(0)=A0, A'(0)=A1
        if np.abs(Omega) == 0:
            C = A0
            D = A1 + a * A0
            A_hom = np.exp(-a * tau) * (C + D * tau)
            V_hom = np.exp(-a * tau) * (D - a * (C + D * tau))
        else:
            C = A0
            D = (A1 + a * A0) / Omega
            cos = np.cos(Omega * tau)
            sin = np.sin(Omega * tau)
            exp = np.exp(-a * tau)

            A_hom = exp * (C * cos + D * sin)
            V_hom = exp * (-a * (C * cos + D * sin) + (-C * Omega * sin + D * Omega * cos))

        A_t = A_hom + A_forced
        V_t = V_hom + V_forced

        return (A_t, V_t) if return_V else A_t
