"""HW4 Problem 1 (Newman 8.18): the Brusselator with an adaptive Bulirsch-Stoer method.

    dx/dt = 1 - (b+1) x + a x^2 y,     dy/dt = b x - a x^2 y,

with a = 1, b = 3 and x = y = 0 at t = 0, solved to t = 20.  Each interval H is
integrated with the modified midpoint method for n = 1, 2, ..., 8 steps and
Richardson-extrapolated; if the error estimate is still above H*delta after
n = 8, the interval is cut in half and each half is done recursively.  The
target accuracy is delta = 1e-10 per unit time in both x and y.
"""
import numpy as np
import matplotlib.pyplot as plt

from plotstyle import use_science

use_science()

a, b = 1.0, 3.0
t0, t1 = 0.0, 20.0
delta = 1e-10
NMAX = 8


def f(r):
    x, y = r
    return np.array([1 - (b + 1) * x + a * x * x * y, b * x - a * x * x * y])


def modified_midpoint(r, H, n):
    """n modified-midpoint steps of size H/n starting from r."""
    h = H / n
    r1 = r + 0.5 * h * f(r)
    r2 = r + h * f(r1)
    for _ in range(n - 1):
        r1 += h * f(r2)
        r2 += h * f(r1)
    return 0.5 * (r1 + r2 + 0.5 * h * f(r2))


def bs_interval(r, H, delta):
    """Richardson extrapolation of the modified midpoint method over one interval.

    Returns (r(t+H), n) on success, or (None, NMAX) if the error estimate is still
    above H*delta with n = NMAX.
    """
    R_prev = [modified_midpoint(r, H, 1)]
    for n in range(2, NMAX + 1):
        R = [modified_midpoint(r, H, n)]
        for m in range(1, n):
            eps = (R[m - 1] - R_prev[m - 1]) / ((n / (n - 1)) ** (2 * m) - 1)
            R.append(R[m - 1] + eps)
        # eps is now the error estimate of R_{n,n-1}; require it in both x and y
        if np.max(np.abs(eps)) < H * delta:
            return R[n - 1], n
        R_prev = R
    return None, NMAX


def solve(r, t, H, delta, out):
    """Integrate from t to t+H, halving recursively; append (t_end, r_end, n) to out."""
    rnew, n = bs_interval(r, H, delta)
    if rnew is None:
        rmid = solve(r, t, H / 2, delta, out)
        return solve(rmid, t + H / 2, H / 2, delta, out)
    out.append((t + H, rnew, n, H))
    return rnew


def run(delta):
    out = []
    # The first tries (H = 20, 10, ...) blow up to inf/nan; nan fails the error
    # test, so those intervals are simply split.
    with np.errstate(over="ignore", invalid="ignore"):
        solve(np.array([0.0, 0.0]), t0, t1 - t0, delta, out)
    t = np.array([t0] + [o[0] for o in out])
    r = np.array([[0.0, 0.0]] + [o[1] for o in out])
    n = np.array([o[2] for o in out])
    H = np.array([o[3] for o in out])
    return t, r, n, H


# Fixed point and its stability: x* = 1, y* = b/a.  The Jacobian there has
# trace b - 1 - a and determinant a, so the fixed point is unstable (and the
# system settles on a limit cycle) for b > 1 + a.
J = np.array([[b - 1, a], [-b, -a]])
ev = np.linalg.eigvals(J)
print(f"fixed point (x, y) = (1, {b / a:g}); Jacobian eigenvalues {ev[0]:.4f}, {ev[1]:.4f}")
print(f"  b = {b:g} > 1 + a = {1 + a:g}: unstable spiral, solution approaches a limit cycle")

t, r, n, H = run(delta)
x, y = r.T
print(f"\ndelta = {delta:g} per unit time: {len(H)} intervals")
print(f"  interval widths from {H.min():.4g} (= 20/2^{np.log2(20 / H.min()):.0f}) "
      f"to {H.max():.4g} (= 20/2^{np.log2(20 / H.max()):.0f})")
print(f"  modified-midpoint steps per accepted interval: n = {n.min()}..{n.max()}, "
      f"mean {n.mean():.2f}")
print(f"  x(20) = {x[-1]:.12f}, y(20) = {y[-1]:.12f}")

# Accuracy check against a much tighter run.  The global error should be of
# order delta * (t1 - t0) = 2e-9 if the per-interval estimates are honest.
tr, rr, nr, Hr = run(1e-12)
print(f"\nreference run with delta = 1e-12: {len(Hr)} intervals, "
      f"x(20) = {rr[-1, 0]:.12f}, y(20) = {rr[-1, 1]:.12f}")
err = np.abs(r[-1] - rr[-1])
print(f"  |error| at t = 20: x {err[0]:.2e}, y {err[1]:.2e} "
      f"(target delta * 20 = {delta * 20:.0e})")

# Where are the short intervals?  Compare the interval width with the speed
# |dr/dt| in the (x, y) plane at the interval midpoints.
speed = np.array([np.hypot(*f(0.5 * (r[i] + r[i + 1]))) for i in range(len(H))])
fast = speed > np.median(speed)
print(f"\nmean interval width where |dr/dt| > median: {H[fast].mean():.3f}, "
      f"elsewhere: {H[~fast].mean():.3f}")
print(f"  correlation of log H with log |dr/dt|: "
      f"{np.corrcoef(np.log(H), np.log(speed))[0, 1]:.2f}")

# The accepted intervals are too coarse to draw a smooth curve through, so the
# plotted line comes from the same method on a fine fixed grid.
tf = np.linspace(t0, t1, 4001)
rf = np.zeros((len(tf), 2))
for i in range(1, len(tf)):
    rf[i] = solve(rf[i - 1], tf[i - 1], tf[i] - tf[i - 1], 1e-12, [])
print(f"\ndense curve (4000 fixed intervals, delta = 1e-12) at t = 20 differs from "
      f"the adaptive one by {np.max(np.abs(rf[-1] - r[-1])):.1e}")

# --- figure -------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
ax = axes[0]
ax.plot(tf, rf[:, 0], color="C0", lw=1.0, label="$x$")
ax.plot(tf, rf[:, 1], color="C3", lw=1.0, label="$y$")
ax.plot(t, x, ".", color="C0", ms=5)
ax.plot(t, y, ".", color="C3", ms=5)
ax.set_ylabel("concentration")
ax.set_ylim(0, 1.05 * rf.max())
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
          title=r"$a=1,\ b=3$, $\delta=10^{-10}$; dots mark the boundaries of the "
                f"{len(H)} intervals", alignment="left")

ax = axes[1]
ax.step(t, np.append(H, H[-1]), where="post", color="k", lw=1.0)
ax.set_yscale("log", base=2)
ax.set_ylabel("interval width $H$")
ax.set_xlabel("$t$")
ax.set_xlim(t0, t1)
ax2 = ax.twinx()
ax2.plot(tf, np.hypot(*np.array([f(ri) for ri in rf]).T), color="C2", lw=0.8)
ax2.set_yscale("log")
ax2.set_ylabel(r"$|d\mathbf{r}/dt|$", color="C2")
ax2.tick_params(axis="y", colors="C2")
ax2.grid(False)
fig.savefig("figs/problem1_brusselator.png")
print("\nwrote figs/problem1_brusselator.png")
