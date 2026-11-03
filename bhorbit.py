"""Adaptive fourth-order Runge-Kutta integrator for one black hole of an equal-mass
binary, with an optional dynamical-friction drag.

Units: G = M = 1 (M = mass of one black hole), length unit 100 pc.  With the
other black hole at -r, the separation is 2r and

    d^2 r/dt^2 = -r / (4 r^3) - A v / (v^3 + B),

i.e. a Kepler problem with effective GM' = 1/4 plus a drag opposite to v.  The
step size follows Newman Sec. 8.4: two steps of h are compared with one step of
2h, the position error is eps = |r_1 - r_2| / 30, and the step is accepted if
eps < h * delta; the next step is h * (h delta / eps)^(1/4), at most 2h.
"""
import math

import numpy as np
from numba import njit

GM = 0.25          # effective G M' of the relative orbit of one hole
R_S = 1e-7         # Schwarzschild radius in units of 100 pc

# Convert code units to physical ones.  r_s = 2 G M / c^2 = 1e-7 * 100 pc fixes
# the black-hole mass, and then the velocity unit sqrt(G M / 100 pc) and the
# time unit (100 pc) / (velocity unit).
PC = 3.0857e18     # cm
C_LIGHT = 2.99792458e10
G_CGS = 6.6743e-8
MSUN = 1.98892e33
MYR = 3.15576e13   # s
L_UNIT = 100 * PC
M_BH = R_S * L_UNIT * C_LIGHT**2 / (2 * G_CGS)
V_UNIT = math.sqrt(G_CGS * M_BH / L_UNIT)
T_UNIT_MYR = L_UNIT / V_UNIT / MYR


@njit(cache=True)
def _deriv(x, y, vx, vy, A, B):
    r2 = x * x + y * y
    g = -GM / (r2 * math.sqrt(r2))
    v = math.sqrt(vx * vx + vy * vy)
    d = A / (v * v * v + B)
    return vx, vy, g * x - d * vx, g * y - d * vy


@njit(cache=True)
def _rk4(s, h, A, B):
    x, y, vx, vy = s[0], s[1], s[2], s[3]
    k1 = _deriv(x, y, vx, vy, A, B)
    k2 = _deriv(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1],
                vx + 0.5 * h * k1[2], vy + 0.5 * h * k1[3], A, B)
    k3 = _deriv(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1],
                vx + 0.5 * h * k2[2], vy + 0.5 * h * k2[3], A, B)
    k4 = _deriv(x + h * k3[0], y + h * k3[1], vx + h * k3[2], vy + h * k3[3], A, B)
    out = np.empty(4)
    for i in range(4):
        out[i] = s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
    return out


@njit(cache=True)
def osculating(s):
    """Energy, angular momentum and pericentre of the osculating Kepler orbit."""
    x, y, vx, vy = s[0], s[1], s[2], s[3]
    r = math.sqrt(x * x + y * y)
    E = 0.5 * (vx * vx + vy * vy) - GM / r
    L = x * vy - y * vx
    e = math.sqrt(max(0.0, 1.0 + 2.0 * E * L * L / (GM * GM)))
    return E, L, L * L / (GM * (1.0 + e))   # r_p = a (1 - e) without cancellation


@njit(cache=True)
def integrate(s0, A, B, delta, tmax, rstop, h0, nrec, max_steps):
    """Integrate from state s0 until t = tmax or r <= rstop.

    rstop is checked both on the accepted steps and, at every pericentre passage
    (r.v changing sign from - to +), on the osculating pericentre distance, so a
    pericentre that falls between two steps is not missed.

    Returns (t_end, reason, nsteps, traj, peri) where reason is 0 for tmax,
    1 for r <= rstop, 2 for max_steps and 3 if
    the step size fell below the round-off floor; traj holds (t, x, y, vx, vy) for every
    accepted step up to nrec rows; peri holds (t, r_p, E, L) at each pericentre.
    """
    s = s0.copy()
    t = 0.0
    h = h0
    traj = np.empty((nrec, 5))
    peri = np.empty((100000, 4))
    nrow = 0
    npk = 0
    if nrec > 0:
        traj[0, 0] = t
        traj[0, 1:] = s
        nrow = 1
    rdotv = s[0] * s[2] + s[1] * s[3]
    nsteps = 0
    reason = 2
    while nsteps < max_steps:
        if t + 2 * h > tmax:
            h = 0.5 * (tmax - t)
        s1 = _rk4(_rk4(s, h, A, B), h, A, B)
        s2 = _rk4(s, 2 * h, A, B)
        err = math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2) / 30.0
        rho = h * delta / err if err > 0 else 16.0
        if rho < 1.0:
            h *= rho**0.25
            if t + 2 * h == t:      # step below the resolution of t: delta is too
                reason = 3          # small for double precision
                break
            continue
        s = s1
        t += 2 * h
        nsteps += 1
        h *= min(rho**0.25, 2.0)
        if nrow < nrec:
            traj[nrow, 0] = t
            traj[nrow, 1:] = s
            nrow += 1
        r = math.sqrt(s[0] * s[0] + s[1] * s[1])
        new_rdotv = s[0] * s[2] + s[1] * s[3]
        hit = r <= rstop
        if rdotv < 0 <= new_rdotv:
            E, L, rp = osculating(s)
            if npk < peri.shape[0]:
                peri[npk, 0] = t
                peri[npk, 1] = rp
                peri[npk, 2] = E
                peri[npk, 3] = L
                npk += 1
            hit = hit or rp <= rstop
        rdotv = new_rdotv
        if hit:
            reason = 1
            break
        if t >= tmax * (1 - 1e-15):
            reason = 0
            break
    return t, reason, nsteps, traj[:nrow], peri[:npk]


def initial_state(v0):
    """BH at x = 1, y = 0 moving in +y with speed v0."""
    return np.array([1.0, 0.0, 0.0, v0])


V_CIRC = math.sqrt(GM / 1.0)   # circular speed at r = 1


def v_for_pericentre(rp, ra=1.0):
    """Apocentre speed at ra for a Kepler orbit with pericentre rp."""
    return math.sqrt(2 * GM * rp / (ra * (ra + rp)))
