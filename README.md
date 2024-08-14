# MMD Wasserstein gradient flow on the line
Implicit and explicit Euler schemes for Wasserstein gradient flow of the Maximum Mean Discrepancy (MMD) with respect to the negative distance kernel on the line.

Overview
---------------------------
This repository provides three scripts used to produce the numerical experiments for the paper

[Wasserstein Gradient Flows of MMD Functionals with Distance Kernel and  Cauchy Problems on Quantile Functions](https://arxiv.org/abs/TODO) by [Richard Duong](https://www.researchgate.net/profile/Richard-Duong), [Viktor Stein](https://viktorajstein.github.io/), [Robert Beinert](https://scholar.google.com/citations?user=D-RIm78AAAAJ&hl=en&oi=ao), [Johannes Hertrich](https://johertrich.github.io/), and [Gabriele Steidl](https://page.math.tu-berlin.de/~steidl/).

1. ```implicit_with_Dirac_at_zero_target``` recreates figure ..., illustrating the implicit Euler discretization of the MMD flow whose target is the Dirac measure at 0.
2. ```discrete_Target``` recreates figures ... and ... and implements the explicit formula for the quantile functions of the MMD flow with discrete target measure.
3. ```Euler_schemes``` recreates all other figures.


If you use this code please cite this preprint, preferably like this:
```
@unpublished{NSSR24,
 author = {Duong, Richard and Stein, Viktor and Beinert, Robert and Hertrich, Johannes and Steidl, Gabriele},
 title = {Wasserstein Gradient Flows of {MMD} Functionals with Distance Kernel and {C}auchy Problems on Quantile Functions},
 note = {ArXiv preprint},
 volume = {arXiv:TODO},
 year = {2024},
 month = {Aug},
 url = {https://arxiv.org/abs/TODO},
 doi = {TODO}
 }
``
