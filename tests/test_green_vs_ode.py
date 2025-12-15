import numpy as np
from bawosc import DrivenHarmonicOscillator

def test_green_vs_ode_simple_cosine():
    omega0 = 1.0
    gamma = 0.1
    m = 1.0

    t = np.linspace(0.0, 100.0, 20001)
    f = np.cos(2*omega0*t)

    osc = DrivenHarmonicOscillator(m=m, gamma=gamma, omega0=omega0)
    xg = osc.solve_via_green(t, f)
    xo = osc.solve_direct_ode(t, f)

    np.testing.assert_allclose(xg, xo, rtol=5e-3, atol=5e-3)
