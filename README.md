# Negative distance MMD Wasserstein gradient flow on the line
Implicit and explicit Euler schemes for Wasserstein gradient flow of the Maximum Mean Discrepancy (MMD) with respect to the negative distance kernel on the line.

Overview
---------------------------
This repository provides three scripts used to produce the numerical experiments for the paper

[Wasserstein Gradient Flows of MMD Functionals with Distance Kernel and  Cauchy Problems on Quantile Functions](https://arxiv.org/abs/2408.07498) by [Richard Duong](https://www.researchgate.net/profile/Richard-Duong), [Viktor Stein](https://viktorajstein.github.io/), [Robert Beinert](https://scholar.google.com/citations?user=D-RIm78AAAAJ&hl=en&oi=ao), [Johannes Hertrich](https://johertrich.github.io/), and [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/).

1. ```implicit_with_Dirac_at_zero_target``` recreates figure 21, illustrating the implicit Euler discretization of the MMD flow whose target is the Dirac measure at 0.
2. ```Euler_schemes``` recreates all other figures.

<!-- 2. ```discrete_Target``` recreates figures 14 and implements the explicit formula for the quantile functions of the MMD flow with discrete target measure. -->


If you use this code please cite this preprint, preferably like this:
```
@unpublished{DSBHS24,
 author = {Duong, Richard and Stein, Viktor and Beinert, Robert and Hertrich, Johannes and Steidl, Gabriele},
 title = {Wasserstein Gradient Flows of {MMD} Functionals with Distance Kernel and {C}auchy Problems on Quantile Functions},
 note = {ArXiv preprint},
 volume = {arXiv:2408.07498},
 year = {2024},
 month = {Aug},
 url = {https://arxiv.org/abs/2408.07498},
 doi = {10.48550/arXiv.2408.07498}
 }
```


Required packages
---------------------------
This script requires the following Python packages. We tested the code with Python 3.11.7 and the following package versions:

scipy 1.12.0
numpy 1.26.3
matplotlib 3.8.2
pypdf 4.3.1 (if you want to collate single pdf plots to one big pdf)

Usually code is also compatible with some later or earlier versions of those packages.
