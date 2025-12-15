import numpy as np

def time_at_frequency(t, f_t, f_target):
    """
    Given arrays t and f_t (same length), return the time(s) at which
    f_t(t) = f_target, using linear interpolation.
    """
    t = np.asarray(t)
    f_t = np.asarray(f_t)
    f_target = np.asarray(f_target, dtype=float)

    idx = np.argsort(f_t)
    f_sorted = f_t[idx]
    t_sorted = t[idx]

    t_at = np.interp(f_target, f_sorted, t_sorted, left=np.nan, right=np.nan)

    if np.ndim(f_target) == 0:
        return float(t_at)
    return t_at

def chirp_with_window(t, f_start, f_end, t_start, t_end, envelope=None, phi0=0.0):
    """
    Linear chirp active only in [t_start, t_end].
    Returns signal and instantaneous frequency array.
    """
    if envelope is None:
        envelope = np.ones_like(t)

    dt = t[1] - t[0]
    f_t = np.zeros_like(t)

    mask = (t >= t_start) & (t <= t_end)
    dT = t_end - t_start

    f_t[mask] = f_start + (f_end - f_start) * (t[mask] - t_start) / dT

    f_t[(t < t_start)] = f_start
    f_t[(t > t_end)] = f_end

    omega_t = 2 * np.pi * f_t
    phase = phi0 + np.cumsum(omega_t) * dt

    signal = envelope * np.cos(phase)
    return signal, f_t

def chirp_force(t, A=1.0, omega_min=0.5, omega_max=5.0, t0=5.0, sigma=3.0):
    """
    Smooth linear chirp with Gaussian envelope.
    """
    alpha = (omega_max - omega_min) / (2 * t[-1])
    envelope = np.exp(-(t - t0)**2 / (2 * sigma**2))
    phase = omega_min * t + alpha * t**2
    return A * envelope * np.cos(phase)
