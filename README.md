# dyn_svo

Code for the paper "State-Dependent Cooperation Weights for Wave-Damping Automated Vehicles:
A Pre-Registered Evaluation on Ring-Road Trajectories".

## Data

The simulations replay the trajectories of the Arizona ring-road experiments (ARED; Wu et al., 2019).
Place the files `expA.csv` to `expH.csv` in `<DATA_ROOT>/shared_data/arizona_ring/` and set the
environment variable `DYN_SVO_ROOT=<DATA_ROOT>`. `config/data_manifest.yaml` lists the md5 of every
file; the scripts refuse files that differ. Results are written to `<DATA_ROOT>/dyn_svo/results/`.

## Installation

Python 3.13 with the pinned packages of `requirements.txt`. The reinforcement-learning scripts also need
`config/requirements_rl.txt`, the theory scripts `config/requirements_theory.txt`. Run the test suite
from the repository root with `python -m pytest`.

## Layout

- `src/dyn_svo/`: simulator, human driver model, controllers (MPC, RL, classical baselines), metrics and theory checks.
- `scripts/`: command-line entry points; the docstring at the top of each script lists its commands.
- `config/`: all settings, including the decision constants of the evaluation (`config/decision_constants.yaml`).
- `tests/`: test suite.

`scripts/paper_macros.py` and `scripts/paper_figures.py` regenerate the numbers, tables and figures of the
paper from the result files listed in `config/paper_sources.yaml`.

## License

MIT, see `LICENSE`.
