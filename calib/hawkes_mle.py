#!/usr/bin/env python3
"""
hawkes_mle.py

Maximum-likelihood estimation for Hawkes processes with exponential kernels.
Improvements over naive implementation:
 - O(n) log-likelihood using recursive updates (R_k recursion).
 - Optional numba acceleration.
 - Simplified tick backend (if tick is installed).
 - Hessian via numdifftools if available, fallback to finite differences.
 - Bootstrap with tqdm progress bar.

Author: <Abhishek Tiwari>
"""

from __future__ import annotations

import warnings
from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv

# optional libraries
try:
    import tick.hawkes as tick_hawkes  # type: ignore
    _HAS_TICK = True
except Exception:
    _HAS_TICK = False

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    import numdifftools as nd
    _HAS_NUMDIFF = True
except Exception:
    _HAS_NUMDIFF = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# small nicety: provide a no-op njit if numba not present
if not _HAS_NUMBA:
    def njit(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator


# -------------------------
# Core: O(n) log-likelihood
# -------------------------
@njit(cache=True)
def _hawkes_loglik_recursive(params: np.ndarray, times: np.ndarray) -> float:
    """
    O(n) log-likelihood for univariate Hawkes with K exponential components.

    params layout: [mu, alpha_1, ..., alpha_K, beta_1, ..., beta_K]

    Uses recursive accumulator:
      R_k(i) = exp(-beta_k * (t_i - t_{i-1})) * (1 + R_k(i-1))
    so that lambda(t_i) = mu + sum_k alpha_k * R_k(i)
    """
    mu = params[0]
    if mu <= 0.0:
        return -1e30  # penalize invalid params in optimizer

    total_len = len(params)
    K = (total_len - 1) // 2
    alphas = params[1:1 + K]
    betas = params[1 + K:1 + 2 * K]

    n = times.shape[0]
    if n == 0:
        return 0.0

    # initialize R_k
    R = np.zeros(K, dtype=np.float64)
    ll = 0.0

    t_prev = 0.0
    for i in range(n):
        t_i = times[i]
        if i == 0:
            # For first event: R_k = 0 => lambda = mu
            lam = mu
            # but we still update R for next event as 1 (since previous history includes this event)
            # We'll do that after incrementing ll
        else:
            dt = t_i - t_prev
            lam = mu
            for k in range(K):
                R[k] = np.exp(-betas[k] * dt) * (1.0 + R[k])
                lam += alphas[k] * R[k]
        # ensure lambda positive
        if lam <= 0.0:
            # extremely unlikely if parameters positive; penalize
            return -1e30
        ll += np.log(lam)
        # incorporate the current event into R for the next iteration:
        # (Note: above recursion already added +1 when updating; for first event we didn't update - do it now)
        if i == 0:
            for k in range(K):
                R[k] = 1.0  # because the event just occured
        t_prev = t_i

    # compensator (integrated intensity)
    T = times[-1]
    comp = mu * T
    for k in range(K):
        beta_k = betas[k]
        alpha_k = alphas[k]
        # compute sum_j (1 - exp(-beta (T - t_j)))
        # vectorized in loops for numba compatibility
        s = 0.0
        for j in range(n):
            s += (1.0 - np.exp(-beta_k * (T - times[j])))
        comp += alpha_k / beta_k * s

    return ll - comp


# wrapper for Python (non-numba) to facilitate use of numdifftools later
def _hawkes_loglik(params: np.ndarray, times: np.ndarray) -> float:
    if _HAS_NUMBA:
        return float(_hawkes_loglik_recursive(params, times))
    else:
        # pure numpy version (same recursion)
        mu = params[0]
        if mu <= 0.0:
            return -1e30
        total_len = len(params)
        K = (total_len - 1) // 2
        alphas = params[1:1 + K]
        betas = params[1 + K:1 + 2 * K]
        n = len(times)
        if n == 0:
            return 0.0
        R = np.zeros(K, dtype=float)
        ll = 0.0
        t_prev = 0.0
        for i in range(n):
            t_i = times[i]
            if i == 0:
                lam = mu
            else:
                dt = t_i - t_prev
                lam = mu
                for k in range(K):
                    R[k] = np.exp(-betas[k] * dt) * (1.0 + R[k])
                    lam += alphas[k] * R[k]
            if lam <= 0.0:
                return -1e30
            ll += np.log(lam)
            if i == 0:
                R[:] = 1.0
            t_prev = t_i
        T = times[-1]
        comp = mu * T
        for k in range(K):
            beta_k = betas[k]
            alpha_k = alphas[k]
            s = 0.0
            for j in range(n):
                s += (1.0 - np.exp(-beta_k * (T - times[j])))
            comp += alpha_k / beta_k * s
        return ll - comp


# -------------------------
# Fitting backends
# -------------------------
def _fit_manual(times: np.ndarray,
                kernel: str = "exp",
                init_guess: Optional[Dict[str, np.ndarray]] = None
                ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Fit using scipy.optimize. Supports single or multi-exp kernel.
    kernel: 'exp' or 'exp-K'
    """
    if kernel == "exp":
        K = 1
    elif kernel.startswith("exp-"):
        try:
            K = int(kernel.split("-")[1])
        except Exception:
            raise ValueError("kernel must be 'exp' or 'exp-K'")
    else:
        raise ValueError("kernel must be 'exp' or 'exp-K'")

    # initial guesses
    if init_guess:
        mu0 = float(np.asarray(init_guess.get("mu"))[0])
        alphas0 = np.asarray(init_guess.get("alpha", np.full(K, 0.3)), dtype=float)
        betas0 = np.asarray(init_guess.get("beta", np.full(K, 1.0)), dtype=float)
        if len(alphas0) != K or len(betas0) != K:
            raise ValueError("init_guess alphas/betas length mismatch")
    else:
        mu0 = max(1e-3, 0.1)
        alphas0 = np.full(K, 0.3)
        betas0 = np.full(K, 1.0)

    x0 = np.concatenate([[mu0], alphas0, betas0])

    # bounds
    bounds = [(1e-9, None)] + [(0.0, None)] * K + [(1e-6, None)] * K

    def negll(x):
        return -_hawkes_loglik(x, times)

    res = minimize(negll, x0, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=2000, disp=False))

    if not res.success:
        warnings.warn(f"Optimizer did not converge: {res.message}")

    mu_hat = res.x[0]
    alphas_hat = res.x[1:1 + K]
    betas_hat = res.x[1 + K:1 + 2 * K]

    # Hessian estimation for standard errors
    x_opt = res.x.copy()
    try:
        if _HAS_NUMDIFF:
            # use numdifftools to compute Hessian of negll
            H_fun = nd.Hessian(lambda x: negll(np.asarray(x)))
            hess = np.asarray(H_fun(x_opt))
        else:
            # fallback finite-difference Hessian (symmetric)
            eps = 1e-5
            npar = len(x_opt)
            hess = np.zeros((npar, npar), dtype=float)
            f0 = negll(x_opt)
            for i in range(npar):
                x_ip = x_opt.copy(); x_ip[i] += eps
                x_im = x_opt.copy(); x_im[i] -= eps
                f_ip = negll(x_ip)
                f_im = negll(x_im)
                hess[i, i] = (f_ip - 2.0 * f0 + f_im) / (eps ** 2)
                for j in range(i+1, npar):
                    x_ipp = x_opt.copy(); x_ipp[i] += eps; x_ipp[j] += eps
                    x_ipm = x_opt.copy(); x_ipm[i] += eps; x_ipm[j] -= eps
                    x_imp = x_opt.copy(); x_imp[i] -= eps; x_imp[j] += eps
                    x_imm = x_opt.copy(); x_imm[i] -= eps; x_imm[j] -= eps
                    f_ipp = negll(x_ipp); f_ipm = negll(x_ipm)
                    f_imp = negll(x_imp); f_imm = negll(x_imm)
                    hess[i, j] = (f_ipp - f_ipm - f_imp + f_imm) / (4.0 * eps * eps)
                    hess[j, i] = hess[i, j]
    except Exception:
        hess = None

    if hess is None or hess.size == 0:
        stderr = np.full_like(x_opt, np.nan)
    else:
        try:
            cov = np.linalg.pinv(hess)
            stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except Exception:
            stderr = np.full_like(x_opt, np.nan)

    params = dict(mu=np.array([mu_hat]), alpha=alphas_hat, beta=betas_hat)
    stderr_dict = dict(mu=np.array([stderr[0]]),
                       alpha=stderr[1:1 + K],
                       beta=stderr[1 + K:1 + 2 * K])
    return params, stderr_dict


def _fit_tick(times: np.ndarray,
              kernel: str = "exp",
              init_guess: Optional[Dict[str, np.ndarray]] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Fit using tick library if available. Only supports single-exponential (K=1).
    """
    if not _HAS_TICK:
        raise RuntimeError("tick library not available")
    if kernel != "exp":
        raise ValueError("tick backend supports only single exponential kernel")

    # tick signature: fit HawkesExpKern or similar inference API
    # create learner and fit to the single sequence
    learner = tick_hawkes.HawkesExpKern().fit([times])
    # learner exposes baseline_, alpha_, decay_ or similar depending on version
    try:
        mu = learner.baseline[0]
        alpha = learner.alpha[0, 0]
        beta = learner.decay  # sometimes scalar or array
        if hasattr(beta, "__len__"):
            beta = float(beta[0])
    except Exception:
        # different tick versions have slightly different attributes
        mu = float(np.asarray(learner.baseline_)[0])
        alpha = float(np.asarray(learner.adjacency_)[0, 0])
        beta = float(np.asarray(learner.decay_)[0])
    params = dict(mu=np.array([mu]), alpha=np.array([alpha]), beta=np.array([beta]))
    # tick may provide approximate std errs; if not, leave NaN
    stderr = dict(mu=np.array([np.nan]), alpha=np.array([np.nan]), beta=np.array([np.nan]))
    return params, stderr


# -------------------------
# Public API
# -------------------------
def hawkes_mle(times: np.ndarray,
               kernel: str = "exp",
               init_guess: Optional[Dict[str, np.ndarray]] = None,
               backend: str = "auto"
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Estimate Hawkes parameters via MLE.

    Parameters
    ----------
    times : np.ndarray
        Sorted event times.
    kernel : str
        'exp' or 'exp-K' for K-exponential mixture (e.g., 'exp-2').
    init_guess : dict, optional
        Keys 'mu', 'alpha', 'beta' with np.ndarray values.
    backend : str
        'auto', 'tick', 'manual'. Auto prefers tick if available and K=1.

    Returns
    -------
    params : dict
        Estimated parameters.
    stderr : dict
        Approximate standard errors.
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        raise ValueError("times array is empty")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted ascending")

    if kernel == "exp":
        K = 1
    elif kernel.startswith("exp-"):
        K = int(kernel.split("-")[1])
    else:
        raise ValueError("kernel must be 'exp' or 'exp-K'")

    if backend == "auto":
        if _HAS_TICK and K == 1:
            backend = "tick"
        else:
            backend = "manual"

    if backend == "tick":
        return _fit_tick(times, kernel, init_guess)
    elif backend == "manual":
        return _fit_manual(times, kernel, init_guess)
    else:
        raise ValueError("backend must be 'auto', 'tick' or 'manual'")


# -------------------------
# Bootstrap confidence intervals
# -------------------------
def bootstrap_confidence_intervals(times: np.ndarray,
                                   kernel: str = "exp",
                                   B: int = 200,
                                   alpha: float = 0.05,
                                   seed: Optional[int] = None
                                   ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Non-parametric bootstrap for parameter uncertainty (simple resample-with-replacement).
    For Hawkes processes this is only approximate; block-bootstrap or parametric bootstrap
    may be preferred in serious applications.
    """
    rng = np.random.default_rng(seed)
    n = len(times)
    estimates = []
    iterator = range(B)
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc="bootstrap", unit="it")
    for _ in iterator:
        idx = rng.integers(0, n, size=n)
        t_boot = np.sort(times[idx])
        try:
            params, _ = hawkes_mle(t_boot, kernel=kernel, backend="manual")
            flat = np.concatenate([params['mu'], np.atleast_1d(params['alpha']), np.atleast_1d(params['beta'])])
            estimates.append(flat)
        except Exception:
            continue
    estimates = np.array(estimates)
    if estimates.shape[0] < max(10, B // 5):
        raise RuntimeError("too many bootstrap fits failed or too few successful fits")
    low = np.percentile(estimates, 100 * alpha / 2, axis=0)
    high = np.percentile(estimates, 100 * (1 - alpha / 2), axis=0)
    K = (len(low) - 1) // 2
    return dict(mu=(low[:1], high[:1]),
                alpha=(low[1:1 + K], high[1:1 + K]),
                beta=(low[1 + K:], high[1 + K:]))


# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    import os
    import sys
    # make sure sim module on path if running from package root
    root = os.path.join(os.path.dirname(__file__), "..")
    if root not in sys.path:
        sys.path.append(root)
    try:
        from sim.simulate_hawkes import simulate_hawkes_ogata
    except Exception:
        try:
            from simulate_hawkes import simulate_hawkes_ogata
        except Exception as e:
            raise RuntimeError("simulate_hawkes not found; run tests after adding sim/ to PYTHONPATH") from e

    T = 2000.0
    mu_true, alpha_true, beta_true = 0.1, 0.5, 1.0
    times, _ = simulate_hawkes_ogata(T, mu_true, np.array([alpha_true]), np.array([beta_true]), seed=42)
    print(f"Simulated {len(times)} events.")
    params, stderr = hawkes_mle(times, kernel="exp", backend="manual")
    print("Estimated (manual backend):")
    print("  mu   = %.4f ± %.4f  (true %.4f)" % (params['mu'][0], stderr['mu'][0], mu_true))
    print("  alpha= %.4f ± %.4f  (true %.4f)" % (params['alpha'][0], stderr['alpha'][0], alpha_true))
    print("  beta = %.4f ± %.4f  (true %.4f)" % (params['beta'][0], stderr['beta'][0], beta_true))