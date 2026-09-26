# PRISM reproducibility code

This directory contains the code used to reproduce PRISM with EoH and EoH-S
on online bin packing (OBP), travelling salesman problem (TSP), and capacitated
vehicle routing problem (CVRP) tasks.

## Installation

Use Python 3.9--3.12 and install the local package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## LLM configuration

Training examples read credentials and endpoint settings from environment
variables. No credentials are stored in this repository.

```text
LLM_API_HOST   API host name without the URL scheme
LLM_API_KEY    API credential
LLM_MODEL      model identifier
```

Set the variables in your shell and then run one of the scripts under
`../examples/training/`.

## PRISM

The framework-independent PRISM module is in `prism/`. It contains:

- initialization on the complete training instance set;
- instance selection with leave-one-algorithm-out validation;
- selected-instance evaluation;
- mean feedback for EoH;
- CPI feedback with augmented simplex directions for EoH-S.

See `prism/README.md` and `../examples/prism_minimal.py` for the minimal API.

## Released assets

- `../datasets/`: training and test instances or generation scripts;
- `../heuristics/prism/`: final PRISM+EoH and PRISM+EoH-S heuristics;
- `../results/results_prism.xlsx`: per-instance train and test results.

The underlying AAD implementation is derived from LLM4AD. See `LICENSE` for
the applicable license terms.
