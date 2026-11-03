"""Shared SciencePlots setup for all three problems.

Uses real LaTeX text rendering when a working latex/dvipng toolchain and the
packages SciencePlots needs are present; otherwise falls back to matplotlib's
built-in mathtext so the scripts still run on a bare machine.
"""
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" styles)

_PACKAGES = ["siunitx", "amsmath", "type1cm"]


def _latex_available():
    if not (shutil.which("latex") and shutil.which("dvipng")):
        return False
    return all(subprocess.run(["kpsewhich", f"{p}.sty"], capture_output=True).stdout.strip()
               for p in _PACKAGES)


def use_science():
    if _latex_available():
        styles = ["science", "grid"]
    else:
        styles = ["science", "no-latex", "grid"]
        print("plotstyle: no LaTeX toolchain found, falling back to mathtext. "
              "On Arch: paru -S texlive-bin texlive-latexextra texlive-mathscience")
    plt.style.use(styles)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
    return styles


if __name__ == "__main__":
    print("latex available:", _latex_available())
