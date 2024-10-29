import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
# from functools import partial
from pypdf import PdfWriter
import os
import functools
from scipy.optimize import root_scalar


def make_folder(name):
    try:
        os.mkdir(name)
        print(f"Folder '{name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}.")

# The following code to compute quantiles of mixture models is
# a modified version of https://www.jamesatkins.com/posts/
# quantile-function-of-mixture-distributions-in-python/


def _vectorize_float(f):
    vectorized = np.vectorize(f, otypes=[float], signature="(),()->()")

    @functools.wraps(f)
    def wrapper(*args):
        return vectorized(*args)

    return wrapper


class MixtureDistribution:
    def __init__(self, distributions, weights):
        self._distributions = list(distributions)
        self._weights = list(weights)

        if not (all(w >= 0
                    for w in self._weights) and sum(self._weights) == 1):
            raise ValueError("Invalid weight vector.")

        if len(self._distributions) != len(self._weights):
            raise ValueError("Mixtures and weights must have the same length.")

        if len(self._distributions) < 2:
            raise ValueError("Must have at least two component distributions.")

    @_vectorize_float
    def pdf(self, x):
        return sum(w * d.pdf(x)
                   for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def cdf(self, x):
        return sum(w * d.cdf(x)
                   for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def ppf(self, p):
        def objective(x):
            return self.cdf(x) - p

        ppf_values = [dist.ppf(p) for dist in self._distributions]
        min_ppf, max_ppf = min(ppf_values), max(ppf_values)

        # Expand the bracket if necessary
        bracket = [min_ppf, max_ppf]
        step = 1.0
        while objective(bracket[0]) * objective(bracket[1]) > 0:
            bracket[0] -= step
            bracket[1] += step
        # Use root_scalar to find the root within the bracket
        r = root_scalar(
            f=objective,
            bracket=bracket,
            method='brentq'
        )

        assert r.converged
        return r.root


def bisection(f, vals, x_init=None, tol=1e-5, lower_1_Lipschitz=True):
    '''
    compute f^{-1}(vals) via bisection

    Parameters
    ----------
    f : function handle
        function to be inverted
    vals : array of length n
        values for which f^{-1} should be computed.
    x_init : array of length N, optional
        Initialization. The default is None.
    tol : float, optional
        Tolerance for bisection. The default is 1e-5.
    lower_1_Lipschitz : bool, optional
        TODO: explain. The default is True.

    Returns
    -------
    mid : array of length n
        f^{-1}(s) for s in vals.

    '''
    if x_init is None:
        x_init = np.zeros_like(vals)
    f_x = f(x_init)
    low = x_init.copy()
    f_low = f_x.copy()
    high = x_init.copy()
    f_high = f_x.copy()

    if lower_1_Lipschitz:
        # find lower and upper bound for the bisection search
        # here we use that f is the identity plus something monotone.
        low[f_low > vals] = low[f_low > vals] - f_low[f_low > vals] + vals[f_low > vals]
        f_low[f_low > vals] = f(low[f_low > vals])
        high[f_high < vals] = high[f_high < vals] - f_high[f_high < vals] + vals[f_high < vals]
        f_high[f_high < vals] = f(high[f_high < vals])
    else:
        k = 0
        while np.any(f_high < vals):
            high[f_high < vals] = high[f_high < vals] + 2**k
            f_high[f_high < vals] = f(high[f_high < vals])
            k += 1
        k = 0
        while np.any(f_low > vals):
            low[f_low > vals] = low[f_low > vals]-2**k
            f_low[f_low > vals] = f(low[f_low > vals])
            k += 1

    # check if initialization procedure was correct
    assert np.all(f_high >= vals - tol)
    assert np.all(f_low <= vals + tol)

    mid = low
    while np.max(high-low) > tol:
        mid = .5*(high+low)
        f_mid = f(mid)
        leq = f_mid <= vals
        greater = np.logical_not(leq)
        low[leq] = mid[leq]
        f_low[leq] = f_mid[leq]
        high[greater] = mid[greater]
        f_high[greater] = f_mid[greater]

    return mid


def implicit_Euler(R_nu, s, h, g):
    '''
    Perform an implicit Euler step for the MMD functional

    Parameters
    ----------
    R_nu : function handle (compatible with vectorization)
        CDF of target measure nu.
    s : array of length n
        floats from (0, 1) to evaluate g on.
    g : array of length n
        current iterate, a quantile function, evaluated on s.
    h : float
        step size.

    Returns
    -------
    array of length
        the new iterate, evaluated on s.

    '''
    f = lambda x: x + 2*h*R_nu(x)
    return bisection(f, g+2*h*s, x_init=g)


def explicit_Euler(R_nu, s, h, g):
    return g - 2 * h * (R_nu(g) - s)


def approx_density(s, g):
    # approximate density by discrete gradients of the cdf
    g_mid = .5*(g[1:] + g[:-1])
    density = (s[1:] - s[:-1])/(g[1:] - g[:-1])
    return g_mid, density


plot_densities = True
plot_quantiles = True

if __name__ == '__main__':
    name = 'Norm0,.5_to_Norm0,2'
    n = 1001
    s = np.linspace(1/(n+1), n/(n+1), n)
    initial_measure = scipy.stats.norm(0, .5)
    target_measure = scipy.stats.norm(0, 2)
    # MixtureDistribution(
    #       [scipy.stats.norm(loc=-10), scipy.stats.norm(loc=10)],
    #       [1/2, 1/2])
    R_nu = target_measure.cdf
    initial_g = bisection(initial_measure.cdf, s, lower_1_Lipschitz=False)
    target_g = bisection(R_nu, s, lower_1_Lipschitz=False)
    initial_x, initial_y = approx_density(s, initial_g)
    target_x, target_y = approx_density(s, target_g)
    h = 1e-2  # step size

    implicit_step_g = functools.partial(implicit_Euler, R_nu, s, h)
    explicit_step_g = functools.partial(explicit_Euler, R_nu, s, h)
    # apply one step:
    g_impl = initial_g
    g_expl = initial_g
    N = int(1e4)
    folder_name = name+f"_{N}_{h}"
    make_folder(folder_name)
    M = 100
    for n in range(N+1):
        g_impl = implicit_step_g(g_impl)
        g_expl = explicit_step_g(g_expl)
        if not n % M:
            if plot_densities:
                fig, ax = plt.subplots(layout='constrained')
                impl_x, impl_y = approx_density(s, g_impl)
                expl_x, expl_y = approx_density(s, g_expl)
                plt.plot(initial_x, initial_y, label='initial')
                plt.plot(target_x, target_y, label='target')
                plt.plot(expl_x, expl_y, label='explicit')
                plt.plot(impl_x, impl_y, label='implicit')
                # print(((impl_y-expl_y)**2).sum())
                ax.set_ylim(bottom=0)
                plt.legend(loc='upper left')
                plt.title(fr'Iteration {n}')
                plt.savefig(f"{folder_name}/{name}_density_at_{n}_h={h}.pdf",
                            dpi=200, bbox_inches="tight")
                plt.show()
                plt.close()
            if plot_quantiles:
                plt.plot(s, initial_g, label='initial')
                plt.plot(s, target_g, label='target')
                plt.plot(s, g_expl, label='explicit')
                plt.plot(s, g_impl, label='implicit')
                plt.legend()
                plt.title(fr'Iteration {n}')
                plt.savefig(f"{folder_name}/{name}_quantile_at_{n}_h={h}.pdf",
                            dpi=200, bbox_inches="tight")
                plt.show()
                plt.close()
    densities = [f"{folder_name}/{name}_density_at_{n}_h={h}.pdf"
                 for n in range(N+1) if not n % M]
    merger = PdfWriter()
    for pdf in densities:
        merger.append(pdf)
    merger.write(f"{folder_name}/{name}_densities.pdf")
    merger.close()

    quantiles = [f"{folder_name}/{name}_quantile_at_{n}_h={h}.pdf"
                 for n in range(N+1) if not n % M]
    merger = PdfWriter()
    for pdf in quantiles:
        merger.append(pdf)
    merger.write(f"{folder_name}/{name}_quantiles.pdf")
    merger.close()
