# Learning to relax

This repository contains code to reproduce the empirical results in [our paper](https://arxiv.org/abs/2310.02246) on learning-augmented linear system solvers, specifically successive over-relaxation (SOR) and Symmetric SOR-preconditioned conjugate gradient (CG).

## Python port

This repo now includes a Python port under `src/ltr/` plus Python equivalents of the MATLAB scripts under `scripts/` (as they are implemented).

### Conda setup (recommended, Python 3.11)

```bash
conda create -n banditos-py311 -c conda-forge python=3.11 numpy scipy matplotlib joblib pytest -y
conda activate banditos-py311
pip install -e .
```

### Run tests (Python)

```bash
pytest
```

### Install (Python)

```bash
python -m pip install -e .
```

### Run (Python)

```bash
python scripts/learning.py
```

## Script descriptions:
<tt>asymptotic.m</tt>: computes bounds on the performance of SOR across different relaxation parameters  
<tt>cg.m</tt>: computes bounds on the performance of CG across different relaxation parameters  
<tt>comparators.m</tt>: computes average SOR performance across distributions over linear systems  
<tt>contextual.m</tt>: evaluates contextual bandit algorithms for setting the SOR relaxation parameter  
<tt>degenerate.m</tt>: compares SOR performance on degenerate and stochastic target vectors  
<tt>h2d.m</tt>: evaluates different approaches for accelerating a heat equation simulation  
<tt>learning.m</tt>: compares several fixed settings of the SOR relaxation parameter with a bandit algorithm  

## Reference:
Mikhail Khodak, Edmond Chow, Maria-Florina Balcan, Ameet Talwalkar. [*Learning to relax: Setting solver parameters across a sequence of linear system instances*](https://openreview.net/forum?id=5t57omGVMw). ICLR 2024.
