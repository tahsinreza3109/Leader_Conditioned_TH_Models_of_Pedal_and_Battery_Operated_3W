"""Truncated log-location-scale headway distributions, shared by the CSTP notebooks.

Every family is written as an accelerated-failure-time (AFT) model,

    T = exp(mu) * Z,   mu = X @ beta,

so that a covariate shifts log T additively and exp(beta) is a time ratio.
The baseline Z carries the shape:

    family              shape parameters      baseline
    lognormal           sigma                 log Z ~ Normal(0, sigma)
    loglogistic         sigma                 log Z ~ Logistic(0, sigma)
    weibull             sigma                 log Z ~ smallest-extreme-value(0, sigma)
    gamma               k                     Z ~ Gamma(k, 1)
    gengamma            sigma, Q              Prentice (1974) generalized gamma, as in flexsurv:
                                              Q = 0 lognormal, Q = 1 Weibull, Q = sigma gamma
    exponential         (none)                Z ~ Exp(1)  (reference only)

All likelihoods are truncated to [lower, upper]: each observation contributes
log f(t) - log(F(upper) - F(lower)).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

FAMILIES = ["lognormal", "loglogistic", "weibull", "gamma", "gengamma", "exponential"]
LABELS = {"lognormal": "Lognormal", "loglogistic": "Log-logistic", "weibull": "Weibull",
          "gamma": "Gamma", "gengamma": "Generalized gamma", "exponential": "Exponential"}
N_SHAPE = {"lognormal": 1, "loglogistic": 1, "weibull": 1, "gamma": 1, "gengamma": 2, "exponential": 0}
SHAPE_NAMES = {"lognormal": ["sigma"], "loglogistic": ["sigma"], "weibull": ["sigma"], "gamma": ["k"],
               "gengamma": ["sigma", "Q"], "exponential": []}
_Q_ZERO = 1e-4          # |Q| below this: generalized gamma evaluated as its lognormal limit
_MIN_WINDOW_MASS = 1e-10  # a fit putting less probability than this inside [lower, upper] is rejected


# ----------------------------------------------------------------------------- shape transforms
def shape_from_free(family, free):
    """Unconstrained optimiser values -> natural shape parameters."""
    free = np.asarray(free, float)
    if family in ("lognormal", "loglogistic", "weibull"):
        return np.array([np.exp(free[0])])
    if family == "gamma":
        return np.array([np.exp(free[0])])
    if family == "gengamma":
        return np.array([np.exp(free[0]), free[1]])
    return np.array([])


def free_from_shape(family, shape):
    shape = np.asarray(shape, float)
    if family == "gengamma":
        return np.array([np.log(shape[0]), shape[1]])
    return np.log(shape) if len(shape) else np.array([])


# ----------------------------------------------------------------------------- log f, F, F^-1
def logpdf(x, mu, family, shape):
    x = np.asarray(x, float); lx = np.log(x)
    if family == "lognormal":
        s = shape[0]; w = (lx - mu) / s
        return stats.norm.logpdf(w) - np.log(s) - lx
    if family == "loglogistic":
        s = shape[0]; w = (lx - mu) / s
        return -w - 2 * np.logaddexp(0, -w) - np.log(s) - lx
    if family == "weibull":
        s = shape[0]; w = (lx - mu) / s
        return w - np.exp(w) - np.log(s) - lx
    if family == "gamma":
        k = shape[0]; z = x * np.exp(-mu)
        return (k - 1) * np.log(z) - z - special.gammaln(k) - mu
    if family == "gengamma":
        s, Q = shape; w = (lx - mu) / s
        if abs(Q) < _Q_ZERO:
            return stats.norm.logpdf(w) - np.log(s) - lx
        qi = 1 / Q**2
        return (-np.log(s) - lx + np.log(abs(Q)) + qi * np.log(qi)
                + qi * (Q * w - np.exp(Q * w)) - special.gammaln(qi))
    if family == "exponential":
        return -mu - x * np.exp(-mu)
    raise ValueError(family)


def cdf(x, mu, family, shape):
    x = np.asarray(x, float); lx = np.log(x)
    if family == "lognormal":
        return stats.norm.cdf((lx - mu) / shape[0])
    if family == "loglogistic":
        return special.expit((lx - mu) / shape[0])
    if family == "weibull":
        return -np.expm1(-np.exp((lx - mu) / shape[0]))
    if family == "gamma":
        return special.gammainc(shape[0], x * np.exp(-mu))
    if family == "gengamma":
        s, Q = shape; w = (lx - mu) / s
        if abs(Q) < _Q_ZERO:
            return stats.norm.cdf(w)
        qi = 1 / Q**2; g = qi * np.exp(Q * w)
        return special.gammainc(qi, g) if Q > 0 else special.gammaincc(qi, g)
    if family == "exponential":
        return -np.expm1(-x * np.exp(-mu))
    raise ValueError(family)


def ppf(u, mu, family, shape):
    u = np.asarray(u, float)
    if family == "lognormal":
        return np.exp(mu + shape[0] * stats.norm.ppf(u))
    if family == "loglogistic":
        return np.exp(mu + shape[0] * special.logit(u))
    if family == "weibull":
        return np.exp(mu + shape[0] * np.log(-np.log1p(-u)))
    if family == "gamma":
        return np.exp(mu) * special.gammaincinv(shape[0], u)
    if family == "gengamma":
        s, Q = shape
        if abs(Q) < _Q_ZERO:
            return np.exp(mu + s * stats.norm.ppf(u))
        qi = 1 / Q**2
        g = special.gammaincinv(qi, u) if Q > 0 else special.gammaincinv(qi, 1 - u)
        return np.exp(mu + s * np.log(g / qi) / Q)
    if family == "exponential":
        return -np.exp(mu) * np.log1p(-u)
    raise ValueError(family)


# ----------------------------------------------------------------------------- truncated model
def trunc_cdf(x, mu, family, shape, lower, upper):
    Fa, Fb = cdf(lower, mu, family, shape), cdf(upper, mu, family, shape)
    return (cdf(x, mu, family, shape) - Fa) / (Fb - Fa)


def trunc_pdf(x, mu, family, shape, lower, upper):
    Fa, Fb = cdf(lower, mu, family, shape), cdf(upper, mu, family, shape)
    return np.exp(logpdf(x, mu, family, shape)) / (Fb - Fa)


def trunc_sample(rng, mu, family, shape, lower, upper):
    """One draw per element of mu, from the truncated distribution (inverse-CDF method)."""
    mu = np.asarray(mu, float)
    Fa, Fb = cdf(lower, mu, family, shape), cdf(upper, mu, family, shape)
    u = Fa + rng.uniform(size=mu.shape) * (Fb - Fa)
    return np.clip(ppf(u, mu, family, shape), lower, upper)


def negloglik(theta, x, X, family, lower, upper):
    p = X.shape[1]
    beta, shape = theta[:p], shape_from_free(family, theta[p:])
    mu = X @ beta
    with np.errstate(all="ignore"):
        lf = logpdf(x, mu, family, shape)
        Z = cdf(upper, mu, family, shape) - cdf(lower, mu, family, shape)
        if not np.all(np.isfinite(Z)) or np.min(Z) < _MIN_WINDOW_MASS:
            return 1e12                 # degenerate: (almost) no probability inside the window
        val = -(lf.sum() - np.log(Z).sum())
    return val if np.isfinite(val) else 1e12


def _starts(x, X, family):
    beta0, *_ = np.linalg.lstsq(X, np.log(x), rcond=None)
    s0 = max(np.std(np.log(x) - X @ beta0), 0.05)
    if family in ("lognormal", "loglogistic", "weibull"):
        sc = {"lognormal": 1.0, "loglogistic": 0.55, "weibull": 0.78}[family]
        return [np.r_[beta0, np.log(s0 * sc)]]
    if family == "gamma":
        k0 = max(1 / s0**2, 0.5)
        b = beta0.copy(); b[0] -= np.log(k0)
        return [np.r_[b, np.log(k0)]]
    if family == "gengamma":
        # moderate-Q starts plus high-|Q| / small-sigma starts, so ridge modes in small samples are not missed
        return ([np.r_[beta0, np.log(s0), q] for q in (0.0, 0.5, 1.0, -0.5)]
                + [np.r_[beta0, np.log(0.1), q] for q in (3.0, 6.0, 10.0, -3.0)])
    return [beta0.copy()]


def fit(x, family, X=None, lower=0.5, upper=5.0, start=None):
    """Maximum-likelihood fit of the truncated AFT model. Returns a dict."""
    x = np.asarray(x, float)
    X = np.ones((len(x), 1)) if X is None else np.asarray(X, float)
    starts = [np.asarray(start, float)] if start is not None else _starts(x, X, family)
    best = None
    for s0 in starts:
        r = optimize.minimize(negloglik, s0, args=(x, X, family, lower, upper), method="L-BFGS-B")
        if not r.success or best is None or r.fun < best.fun:
            r2 = optimize.minimize(negloglik, r.x, args=(x, X, family, lower, upper), method="Nelder-Mead",
                                   options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-11})
            r = r2 if r2.fun < r.fun else r
        if best is None or r.fun < best.fun:
            best = r
    p = X.shape[1]; k = p + N_SHAPE[family]; n = len(x); ll = -best.fun
    return {"family": family, "theta": best.x, "beta": best.x[:p], "shape": shape_from_free(family, best.x[p:]),
            "loglik": ll, "k": k, "n": n, "AIC": 2 * k - 2 * ll, "AICc": 2 * k - 2 * ll + 2 * k * (k + 1) / max(n - k - 1, 1),
            "BIC": k * np.log(n) - 2 * ll, "converged": bool(np.isfinite(ll)), "X": X, "lower": lower, "upper": upper}


def covariance(res, x):
    """Inverse observed information for the free parameter vector (numerical Hessian)."""
    from statsmodels.tools.numdiff import approx_hess3
    H = approx_hess3(res["theta"], negloglik, args=(np.asarray(x, float), res["X"], res["family"], res["lower"], res["upper"]))
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full_like(H, np.nan)


# ----------------------------------------------------------------------------- goodness of fit
def pit(x, res):
    """Probability-integral transform under the fitted truncated model (one value per observation)."""
    mu = res["X"] @ res["beta"]
    return trunc_cdf(np.asarray(x, float), mu, res["family"], res["shape"], res["lower"], res["upper"])


def uniform_gof(u):
    """KS D, Anderson-Darling A^2 and Cramer-von Mises W^2 of values that should be Uniform(0,1)."""
    u = np.clip(np.sort(np.asarray(u, float)), 1e-12, 1 - 1e-12); n = len(u); i = np.arange(1, n + 1)
    ks = max((i / n - u).max(), (u - (i - 1) / n).max())
    ad = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
    cvm = 1 / (12 * n) + np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
    return np.array([ks, ad, cvm])


def bootstrap_gof(x, res, B=999, seed=0):
    """Refitted parametric bootstrap p-values for KS, AD and CvM (covariates held fixed).

    Returns (observed statistics, p-values, number of valid replicates)."""
    rng = np.random.default_rng(seed)
    obs = uniform_gof(pit(x, res)); mu = res["X"] @ res["beta"]
    exceed = np.zeros(3); valid = 0
    for _ in range(B):
        xb = trunc_sample(rng, mu, res["family"], res["shape"], res["lower"], res["upper"])
        try:
            rb = fit(xb, res["family"], res["X"], res["lower"], res["upper"], start=res["theta"])
        except Exception:
            continue
        if not np.isfinite(rb["loglik"]):
            continue
        exceed += uniform_gof(pit(xb, rb)) >= obs; valid += 1
    return obs, (1 + exceed) / (1 + valid), valid
