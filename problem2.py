"""HW4 Problem 2: inspiral of a supermassive black-hole binary by dynamical friction.

One hole of an equal-mass binary obeys

    d^2 r/dt^2 = -G M r / (4 r^3) - A v / (v^3 + B),

(G = M = 1, length unit 100 pc, r_s = 1e-7), integrated with the adaptive RK4
scheme of bhorbit.py.  The hole starts at (1, 0) moving in +y; with friction its
speed is 0.8 of the circular speed v_c = 1/2.  The integration stops when r
reaches the Schwarzschild radius.

(a) choose delta from a pure Kepler orbit with pericentre r_s, followed for 10 orbits;
(b) the orbit for A = B = 1;
(c) the time to reach r_s against B/A for A, B in [0.5, 10];
(d) dependence on the initial speed and on A and B separately.
"""
import math
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bhorbit import (GM, R_S, V_CIRC, M_BH, MSUN, V_UNIT, T_UNIT_MYR,
                     integrate, initial_state, osculating, v_for_pericentre)
from plotstyle import use_science

use_science()

DELTA = 1e-10      # chosen in part (a)
H0 = 1e-3
TMAX = 1e5
MAX_STEPS = 10**9

print(f"units: r_s = 1e-7 x 100 pc -> M_BH = {M_BH / MSUN:.3e} Msun; "
      f"velocity unit {V_UNIT / 1e5:.2f} km/s; time unit {T_UNIT_MYR:.4f} Myr")


def merge_time(args):
    """Time (code units) to reach r_s, plus run statistics."""
    A, B, v0, delta = args
    t, reason, n, _, peri = integrate(initial_state(v0), A, B, delta, TMAX, R_S, H0,
                                      0, MAX_STEPS)
    if reason != 1:
        raise RuntimeError(f"A={A} B={B} v0={v0}: stopped with reason {reason} at t={t}")
    return t, n, len(peri)


# =============================================================================
# (a) tolerance from a pure Kepler orbit with r_peri = r_s
# =============================================================================
print("\n(a) Kepler orbit without friction, r_apo = 1, r_peri = r_s")
v_kep = v_for_pericentre(R_S)
a_kep = 0.5 * (1 + R_S)
T_kep = 2 * math.pi * math.sqrt(a_kep**3 / GM)
NORB = 10
s0 = initial_state(v_kep)
E0, L0, _ = osculating(s0)
print(f"    v0 = {v_kep:.6e} (= {v_kep / V_CIRC:.3e} v_c), e = {1 - 2 * R_S / (1 + R_S):.8f}, "
      f"period T = {T_kep:.6f}")
print(f"    pericentre speed {math.sqrt(2 * GM / R_S):.1f}, "
      f"pericentre passage time r_s/v_p = {R_S / math.sqrt(2 * GM / R_S):.1e}")

deltas = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
kepler = {}
for d in deltas:
    t, reason, n, traj, peri = integrate(s0, 0.0, 1.0, d, NORB * T_kep, 0.0, H0,
                                         2 * 10**6, MAX_STEPS)
    x, y, vx, vy = traj[:, 1:].T
    rdotv = x * vx + y * vy
    # apocentres: r.v changes sign from + to -; the run ends at the 10th one
    iapo = np.where((rdotv[:-1] > 0) & (rdotv[1:] <= 0))[0] + 1
    if len(iapo) < NORB:        # the last apocentre is the final step itself
        iapo = np.append(iapo, len(traj) - 1)
    Eapo = np.array([osculating(traj[i, 1:])[0] for i in iapo])
    Lapo = np.array([osculating(traj[i, 1:])[1] for i in iapo])
    rapo = np.hypot(x[iapo], y[iapo])
    kepler[d] = dict(n=n, Eapo=Eapo, Lapo=Lapo, rapo=rapo, rp=peri[:, 1], tp=peri[:, 0])
    print(f"    delta = {d:.0e}: {n / NORB:6.0f} steps/orbit; after {NORB} orbits "
          f"dE/E = {Eapo[-1] / E0 - 1:+.1e}, dL/L = {Lapo[-1] / L0 - 1:+.1e}, "
          f"r_apo - 1 = {rapo[-1] - 1:+.1e}, r_peri/r_s - 1 in "
          f"[{peri[:, 1].min() / R_S - 1:+.1e}, {peri[:, 1].max() / R_S - 1:+.1e}], "
          f"mean period error {np.diff(peri[:, 0]).mean() - T_kep:+.1e}")
t, reason, *_ = integrate(s0, 0.0, 1.0, 1e-12, T_kep, 0.0, H0, 0, MAX_STEPS)
print(f"    delta = 1e-12: the step size falls below the spacing of doubles near "
      f"t = {t:.4f} (reason {reason}), so delta >~ 1e-11 is the round-off floor")
print(f"    -> delta = {DELTA:.0e}: over {NORB} orbits the energy changes by "
      f"{abs(kepler[DELTA]['Eapo'][-1] / E0 - 1):.0e} and r_peri by "
      f"{np.max(np.abs(kepler[DELTA]['rp'] / R_S - 1)):.0e} (relative)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"wspace": 0.33})
cols = plt.cm.viridis(np.linspace(0, 0.9, len(deltas)))
for c, d in zip(cols, deltas):
    k = kepler[d]
    orb = np.arange(1, len(k["Eapo"]) + 1)
    axes[0].semilogy(orb, np.abs(k["Eapo"] / E0 - 1), "o-", ms=3, color=c,
                     label=rf"$10^{{{int(round(np.log10(d)))}}}$")
    axes[1].semilogy(np.arange(1, len(k["rp"]) + 1), np.abs(k["rp"] / R_S - 1) + 1e-16,
                     "o-", ms=3, color=c)
axes[0].set_xlabel("orbit")
axes[0].set_ylabel(r"$|E/E_0-1|$ at apocentre")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(deltas),
           bbox_to_anchor=(0.38, 0.98), title=r"tolerance $\delta$ per unit time",
           frameon=False)
axes[1].set_xlabel("pericentre passage")
axes[1].set_ylabel(r"$|r_{\rm peri}/r_s-1|$")
ax = axes[2]
steps = np.array([kepler[d]["n"] / NORB for d in deltas])
errE = np.array([abs(kepler[d]["Eapo"][-1] / E0 - 1) for d in deltas])
ax.loglog(deltas, errE, "o-", color="C0", label=r"$|\Delta E/E|$ after 10 orbits")
ax.loglog(deltas, [abs(kepler[d]["Lapo"][-1] / L0 - 1) for d in deltas], "s-", color="C1",
          label=r"$|\Delta L/L|$ after 10 orbits")
ax.axvline(DELTA, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"$\delta$")
ax.set_ylabel("relative error")
ax.set_ylim(1e-13, 1e3)
ax.legend(fontsize=7, loc="upper left")
ax2 = ax.twinx()
ax2.loglog(deltas, steps, "^--", color="C2", ms=4)
ax2.set_ylim(1e2, 1e6)
ax2.set_ylabel("steps per orbit", color="C2")
ax2.tick_params(axis="y", which="both", colors="C2")
ax2.grid(False)
fig.savefig("figs/problem2a_tolerance.png")

# =============================================================================
# (b) A = B = 1
# =============================================================================
print("\n(b) A = B = 1, v0 = 0.8 v_c")
v0 = 0.8 * V_CIRC
t_b, reason, n_b, traj, peri = integrate(initial_state(v0), 1.0, 1.0, DELTA, TMAX, R_S,
                                         H0, 2 * 10**6, MAX_STEPS)
tt, x, y, vx, vy = traj.T
r = np.hypot(x, y)
print(f"    reached r_s at t = {t_b:.6f} = {t_b * T_UNIT_MYR:.2f} Myr after {n_b} steps "
      f"and {len(peri)} pericentre passages")
osc = np.array([osculating(s) for s in traj[:, 1:]])
ecc = np.sqrt(np.maximum(0, 1 + 2 * osc[:, 0] * osc[:, 1]**2 / GM**2))
for i in [0, 4, 9, 19, len(peri) - 1]:
    print(f"    pericentre {i + 1:2d}: t = {peri[i, 0]:7.4f}, r_p = {peri[i, 1]:.3e}, "
          f"L = {peri[i, 3]:.3e}, E = {peri[i, 2]:+.4f}")
for d in [1e-9, 1e-11]:
    t_d = merge_time((1.0, 1.0, v0, d))[0]
    print(f"    check: delta = {d:.0e} gives t = {t_d:.6f} ({t_d / t_b - 1:+.1e})")

fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], wspace=0.25, hspace=0.1)
ax = fig.add_subplot(gs[:, 0])
ax.plot(x, y, lw=0.5, color="C0")
ax.plot(-x, -y, lw=0.5, color="C3", alpha=0.5)
ax.plot([1, -1], [0, 0], "o", color="k", ms=3)
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.15)
ax.set_xlabel(r"$x$ [100 pc]")
ax.set_ylabel(r"$y$ [100 pc]")
ax.set_title(r"$A=B=1$, $v_0=0.8\,v_c$ (red: the other BH)", fontsize=9)
ins = ax.inset_axes([0.12, 0.64, 0.42, 0.33])
ins.plot(x, y, lw=0.4, color="C0")
ins.set_xlim(-0.01, 0.35)
ins.set_ylim(-0.01, 0.17)
ins.set_aspect("equal")
ins.tick_params(labelsize=6)
ax.indicate_inset_zoom(ins, edgecolor="0.4")

ax = fig.add_subplot(gs[0, 1])
ax.plot(tt * T_UNIT_MYR, np.log10(r), lw=0.5, color="C0")
ax.plot(peri[:, 0] * T_UNIT_MYR, np.log10(peri[:, 1]), ".", ms=3, color="C3",
        label=r"pericentres")
ax.axhline(np.log10(R_S), color="k", ls=":", lw=0.8)
ax.set_ylabel(r"$\log_{10} r$ [100 pc]")
ax.set_xlim(0, t_b * T_UNIT_MYR * 1.01)
ax.legend(loc="lower left", fontsize=8)
ax.tick_params(labelbottom=False)
secax = ax.secondary_xaxis("top", functions=(lambda m: m / T_UNIT_MYR,
                                             lambda c: c * T_UNIT_MYR))
secax.set_xlabel("$t$ [code units]")

ax = fig.add_subplot(gs[1, 1], sharex=ax)
ax.plot(tt * T_UNIT_MYR, 1 - ecc, lw=0.8, color="C2", label=r"$1-e$")
ax.plot(tt * T_UNIT_MYR, -GM / (2 * osc[:, 0]), lw=0.8, color="C4",
        label=r"semi-major axis $a$")
ax.set_yscale("log")
ax.set_xlabel("$t$ [Myr]")
ax.legend(loc="lower left", fontsize=8)
fig.savefig("figs/problem2b_orbit.png")

# =============================================================================
# (c) time to reach r_s against B/A
# =============================================================================
print("\n(c) grid of A, B in [0.5, 10], v0 = 0.8 v_c")
NG = 12
grid = np.geomspace(0.5, 10, NG)
jobs = [(A, B, v0, DELTA) for A in grid for B in grid]
t0 = time.time()
with mp.get_context("fork").Pool() as pool:
    out = pool.map(merge_time, jobs, chunksize=1)
print(f"    {len(jobs)} runs in {time.time() - t0:.0f} s")
AA = np.array([j[0] for j in jobs])
BB = np.array([j[1] for j in jobs])
TT = np.array([o[0] for o in out])
NP = np.array([o[2] for o in out])
ratio = BB / AA
for q in np.unique(np.round(ratio, 6)):
    m = np.isclose(ratio, q)
    if m.sum() >= 3:
        print(f"    B/A = {q:6.3f}: {m.sum():2d} runs, t = {TT[m].min():8.3f}..{TT[m].max():8.3f} "
              f"(spread {TT[m].max() / TT[m].min() - 1:5.1%}), pericentres "
              f"{NP[m].min()}..{NP[m].max()}")
imin = np.argmin(TT)
print(f"    shortest: t = {TT[imin]:.3f} ({TT[imin] * T_UNIT_MYR:.1f} Myr) at B/A = "
      f"{ratio[imin]:.3f}; longest: t = {TT.max():.1f} ({TT.max() * T_UNIT_MYR:.0f} Myr) "
      f"at B/A = {ratio[np.argmax(TT)]:.1f}")
# limits: strong friction -> terminal-velocity infall v = (A/B)^-1 GM'/r^2, so
# t = A / (3 GM' B) = (4/3) A/B; weak friction -> t proportional to B/A.
od = ratio < 0.1
print(f"    overdamped limit (4/3) A/B vs measured for B/A < 0.1: "
      + ", ".join(f"{4 / 3 / q:.2f}/{tq:.2f}" for q, tq in zip(ratio[od], TT[od])))
ud = ratio > 4
slope = np.exp(np.mean(np.log(TT[ud] / ratio[ud])))
print(f"    weak-friction runs (B/A > 4): t / (B/A) = {np.min(TT[ud] / ratio[ud]):.2f}.."
      f"{np.max(TT[ud] / ratio[ud]):.2f}, geometric mean {slope:.2f}")
for (A, B) in [(0.5, 10.0)]:
    ta = merge_time((A, B, v0, DELTA))[0]
    tb_ = merge_time((A, B, v0, 1e-9))[0]
    print(f"    check A = {A}, B = {B}: delta 1e-10 -> {ta:.4f}, 1e-9 -> {tb_:.4f} "
          f"({tb_ / ta - 1:+.1e})")

fig, ax = plt.subplots(figsize=(6.5, 4.4))
sc = ax.scatter(ratio, TT * T_UNIT_MYR, c=np.log10(AA), cmap="viridis", s=14, zorder=3)
cb = fig.colorbar(sc, ax=ax, pad=0.12)
cb.set_label(r"$\log_{10} A$")
q = np.geomspace(0.04, 25, 200)
ax.plot(q, 4 / 3 / q * T_UNIT_MYR, "--", color="C3", lw=1,
        label=r"overdamped infall $t=\frac{4}{3}A/B$")
ax.plot(q, slope * q * T_UNIT_MYR, ":", color="C0", lw=1,
        label=rf"$t={slope:.1f}\,B/A$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.04, 25)
ax.set_ylim(4, 600)
ax.set_xlabel("$B/A$")
ax.set_ylabel(r"time to reach $r_s$ [Myr]")
ax.legend(loc="upper center", fontsize=8)
secax = ax.secondary_yaxis("right", functions=(lambda m: m / T_UNIT_MYR,
                                               lambda c: c * T_UNIT_MYR))
secax.set_ylabel("[code units]")
fig.savefig("figs/problem2c_ratio.png")

# =============================================================================
# (d) initial speed, and A, B separately
# =============================================================================
print("\n(d) dependence on v0 and on A, B separately")
fracs = np.round(np.linspace(0.1, 1.3, 13), 3)
pairs = [(2.5, 0.5), (10.0, 2.0), (0.5, 0.5), (10.0, 10.0), (0.5, 2.5), (2.0, 10.0)]
jobs = [(A, B, f * V_CIRC, DELTA) for (A, B) in pairs for f in fracs]
with mp.get_context("fork").Pool() as pool:
    out = pool.map(merge_time, jobs, chunksize=1)
Td = np.array([o[0] for o in out]).reshape(len(pairs), len(fracs))
print("    v0/v_c   " + " ".join(f"{f:7.2f}" for f in fracs))
for (A, B), row in zip(pairs, Td):
    print(f"    A={A:4.1f} B={B:4.1f} " + " ".join(f"{t:7.2f}" for t in row))
for k in range(0, len(pairs), 2):
    rel = Td[k + 1] / Td[k] - 1
    print(f"    B/A = {pairs[k][1] / pairs[k][0]:.1f}: A = {pairs[k + 1][0]} vs "
          f"{pairs[k][0]} changes t by {rel.min():+.1%} to {rel.max():+.1%}")

# A and B scaled together at fixed B/A, v0 = 0.8 v_c
lam = np.geomspace(0.5, 10, 9)
for q in [0.2, 1.0, 5.0]:
    tl = [merge_time((l, q * l, v0, DELTA))[0] for l in lam if 0.5 <= q * l <= 10]
    print(f"    B/A = {q}: A from 0.5 to 10 at fixed ratio -> t = "
          + ", ".join(f"{t:.2f}" for t in tl))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
styles = ["-", "--"]
for k, ((A, B), row) in enumerate(zip(pairs, Td)):
    ax.plot(fracs, row * T_UNIT_MYR, styles[k % 2], marker="os"[k % 2], ms=3,
            color=f"C{[3, 3, 0, 0, 2, 2][k]}", label=rf"$A={A:g},\ B={B:g}$")
ax.set_yscale("log")
ax.set_xlabel(r"$v_0/v_c$")
ax.set_ylabel(r"time to reach $r_s$ [Myr]")
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
fig.savefig("figs/problem2d_v0.png")
print("\nwrote figs/problem2a_tolerance.png, problem2b_orbit.png, problem2c_ratio.png, "
      "problem2d_v0.png")
