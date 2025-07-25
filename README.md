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
@article{DRSS2025,
    title={Wasserstein gradient ﬂows of {MMD} functionals with distance kernels under {Sobolev} regularization}, 
    author={R. Duong and R. Rux and V. Stein and G. Steidl},
    journal = {Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences},
    volume = {383},
    number = {2298},
    pages = {20240243},
    year = {2025},
    doi = {10.1098/rsta.2024.0243}, 
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
