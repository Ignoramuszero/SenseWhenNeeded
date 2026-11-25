"""Sense When Needed - command line entry point.

Usage:
    python main.py prepare      # parse WISDM -> runs + replay sessions
    python main.py train        # train models + tune event threshold
    python main.py experiments  # run the 3-strategy comparison on test subjects
    python main.py figures      # generate experiment figures
    python main.py dashboard    # build the interactive HTML dashboard
    python main.py all          # everything above, in order
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

STEPS = {
    "prepare": "prepare_data.py",
    "train": "train.py",
    "experiments": "run_experiments.py",
    "figures": "make_figures.py",
    "dashboard": "build_dashboard.py",
    "ablation": "ablation.py",
}


def run(step: str) -> int:
    script = STEPS[step]
    print(f"==> {step}: {script}")
    return subprocess.call([sys.executable, str(SCRIPTS / script)], cwd=str(ROOT))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 0
    if args[0] == "all":
        order = ["prepare", "train", "experiments", "figures", "dashboard"]
    else:
        order = args
    for step in order:
        if step not in STEPS:
            print(f"unknown step: {step}")
            return 2
        code = run(step)
        if code != 0:
            print(f"step '{step}' failed with exit code {code}")
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
