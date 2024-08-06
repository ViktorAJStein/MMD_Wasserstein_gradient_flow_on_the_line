# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:45:29 2024

@author: Viktor Stein
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm


def soft_shrinkage(x, tau):
    return 1/2 * (np.abs(x - tau) + x - tau - np.abs(x + tau) + x + tau)


def approx_density(s, g):
    # approximate density by discrete gradients of the cdf
    g_mid = .5*(g[1:] + g[:-1])
    density = (s[1:] - s[:-1])/(g[1:] - g[:-1])
    return g_mid, density


def implicit_scheme_toDiracatZero(mu=stats.norm(-1, .5),
                                  tau=.01, M=9999, N=500,
                                  plot_densities=True,
                                  plot_quantiles=False):
    s = np.linspace(1/(M+1), M/(M+1), M)
    g = mu.ppf(s)
    g_values = []  # List to store g values
    for n in range(N+1):
        g = soft_shrinkage(g + 2 * tau * s - tau, tau)
        if not n % 50:
            if plot_quantiles:
                g_values.append(g.copy())
            if plot_densities:
                fig, ax = plt.subplots(layout='constrained')
                x, den = approx_density(s, g)
                plt.plot(x, den)
                plt.title(f'Iteration {n}')
                ax.set_ylim(0, 1)
                plt.show()
                plt.close()

    if plot_quantiles:
        fig, ax = plt.subplots(layout='constrained')
        colors = [cm.jet(x) for x in np.linspace(0, 1, len(g_values))]
        for i, g in enumerate(g_values):
            plt.plot(s, g, label=f'Iteration {i*25}', color=colors[i])
        plt.title(r'Implicit Euler scheme for $\nu = \delta_0$')
        plt.legend()
        plt.savefig('Implicit_Euler_scheme_Norm_to_Dirac.pdf',
                    dpi=200, bbox_inches='tight')
        plt.show()


implicit_scheme_toDiracatZero()
