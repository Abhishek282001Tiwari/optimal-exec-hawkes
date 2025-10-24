#!/usr/bin/env python3
"""
propagator_estimation.py

Estimate the transient propagator kernel G(τ) from empirical price changes and
signed order-flow using

  ΔP_t ≈ Σ_{s<t} G(t-s) ε_s q_s   (permanent impact removed)

Improvements vs. naive version:
 - Vectorized design matrix construction via sliding-window (O(n) memory for X view)
 - Optional ridge regularization for stability
 - Toeplitz-based Wiener–Hopf deconvolution (fast, theory-aligned)
 - Robust multi-exponential fit with scaled initial guesses and clipping
 - Bootstrap confidence intervals (block-resampling)
 - Improved plotting with CI shading and optional log-scale

Author: <you>
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz, solve_toeplitz
from numpy.lib.stride_tricks import sliding_window_view
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# ------------------------
# Helper utilities
# ------------------------
def _signed_volume_series(df: pd.DataFrame, side_col: str, size_col: str) -> np.ndarray:
    """
    Compute ε_t * q_t aligned with price differences (one shorter than price series).
    Returns array of length len(df)-1 corresponding to ΔP indexing.
    """
    # signed sizes per event
    signed = (df[side_col] * df[size_col]).values
    # align: if price has n entries, dp = price.diff()[1:] -> length n-1
    # we align eps_q with dp by taking midpoints or simple shift; here we use previous trade
    eps_q = signed[:-1]
    return eps_q


# ------------------------
# A. Non-parametric regression (vectorized, ridge)
# ------------------------
def estimate_propagator_regression(
        df: pd.DataFrame,
        price_col: str = "price",
        side_col: str = "side",
        size_col: str = "size",
        dt: float = 1.0,
        max_lag: int = 60,
        ridge: float = 0.0,
        clip_eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate G(τ) at lags τ = dt, 2dt, ..., max_lag*dt via regularized OLS.

    Model (indexed in discrete ticks aligned to trades):
      ΔP_i = Σ_{k=1}^{max_lag} G_k * (ε q)_{i-k} + η_i

    Returns
    -------
    lags : array (max_lag,)
    G_hat : array (max_lag,)
    stderr : array (max_lag,)  -- approximate std errors from residuals (OLS formula)
    """
    # compute price changes (aligned)
    price = df[price_col].values
    dp = np.diff(price)  # length n-1
    if len(dp) <= max_lag:
        raise ValueError("series too short for chosen max_lag")

    eps_q = _signed_volume_series(df, side_col, size_col)  # length n-1, aligned with dp

    # build sliding windows of past max_lag signed volumes
    # sliding_window_view produces shape (n-1 - max_lag + 1, max_lag)
    X_full = sliding_window_view(eps_q, window_shape=max_lag)
    # we need rows corresponding to dp[max_lag:]
    y = dp[max_lag:]
    X = X_full[:-0 or None]  # same shape: (len(y), max_lag)

    # defensive clipping to avoid degenerate design
    X = np.clip(X, -np.finfo(float).max, np.finfo(float).max)
    y = np.asarray(y, dtype=float)

    # optional ridge: solve (X^T X + λ I) g = X^T y
    X = X[:len(y), :]
    XtX = X.T.dot(X)
    Xty = X.T.dot(y)
    if ridge > 0.0:
        XtX += ridge * np.eye(XtX.shape[0])

    try:
        G_hat = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # fallback to pinv
        G_hat = np.linalg.pinv(XtX).dot(Xty)

    # residuals and approximate stderr
    y_hat = X.dot(G_hat)
    resid = y - y_hat
    n, p = X.shape[0], X.shape[1]
    sigma2 = np.sum(resid ** 2) / max(n - p, 1)
    try:
        cov_g = sigma2 * np.linalg.inv(XtX)
        stderr = np.sqrt(np.maximum(np.diag(cov_g), 0.0))
    except Exception:
        stderr = np.full_like(G_hat, np.nan)

    lags = np.arange(1, max_lag + 1) * dt
    return lags, G_hat, stderr


# ------------------------
# Wiener--Hopf (Toeplitz) deconvolution
# ------------------------
def estimate_propagator_wiener_hopf(
        df: pd.DataFrame,
        price_col: str = "price",
        side_col: str = "side",
        size_col: str = "size",
        dt: float = 1.0,
        max_lag: int = 60,
        ridge: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wiener-Hopf style deconvolution via Toeplitz linear system:
      C_{εq,ΔP}(τ) = Σ_u G(u) C_{εq,εq}(τ-u)
    For discrete lags this yields Toeplitz system to solve for G.

    Returns lags (1..max_lag)*dt and G_hat.
    """
    # compute dp and eps_q aligned as in regression
    price = df[price_col].values
    dp = np.diff(price)
    eps_q = _signed_volume_series(df, side_col, size_col)

    # compute empirical cross-covariances up to max_lag
    n = len(dp)
    # ensure sufficient length
    if n <= max_lag:
        raise ValueError("series too short for chosen max_lag")

    # form series trimmed to same length
    y = dp[max_lag:]
    X_full = sliding_window_view(eps_q, window_shape=max_lag)
    X = X_full  # rows correspond to indices aligning with y (same len)

    # compute C_xx (Toeplitz first column) and C_xy (cross-cov vector)
    # Using sample covariances (uncentered)
    C_xx = np.zeros(max_lag)
    for k in range(max_lag):
        C_xx[k] = np.dot(X[:, 0], X[:, k]) / X.shape[0]  # covariance at lag k

    C_xy = np.zeros(max_lag)
    for k in range(max_lag):
        C_xy[k] = np.dot(y, X[:, k]) / X.shape[0]

    # Toeplitz system: T * G = C_xy, with T_{ij} = C_xx[|i-j|]
    toeplitz_col = C_xx.copy()
    if ridge > 0.0:
        toeplitz_col[0] += ridge

    # solve via Levinson if available (solve_toeplitz)
    try:
        G_hat = solve_toeplitz((toeplitz_col, toeplitz_col), C_xy)
    except Exception:
        # fallback: build full matrix (small max_lag typical)
        T = toeplitz(toeplitz_col)
        if ridge > 0.0:
            T = T + ridge * np.eye(T.shape[0])
        try:
            G_hat = np.linalg.solve(T, C_xy)
        except np.linalg.LinAlgError:
            G_hat = np.linalg.pinv(T).dot(C_xy)

    lags = np.arange(1, max_lag + 1) * dt
    return lags, G_hat


# ------------------------
# B. Multi-exponential parametric fit (robust)
# ------------------------
def _multi_exp_vector(t: np.ndarray, *params) -> np.ndarray:
    """
    Vectorized multi-exponential: params = [alpha1, ..., alpha_n, rho1, ..., rho_n]
    """
    n = len(params) // 2
    alphas = np.array(params[:n])
    rhos = np.array(params[n:])
    G = np.zeros_like(t, dtype=float)
    for i in range(n):
        G += alphas[i] * np.exp(-rhos[i] * t)
    return G


def fit_multi_exponential(
        lags: np.ndarray,
        G_nonpar: np.ndarray,
        n_exp: int = 2,
        init: Optional[Dict[str, np.ndarray]] = None,
        bounds_scale: float = 1e-8,
        maxfev: int = 10000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit multi-exponential kernel to non-parametric estimate G_nonpar.

    Returns alphas, rhos, info with pcov, rsq, G_fit.
    """
    # defensive clipping to avoid zeros / negative issues
    eps = 1e-16
    G_nonpar_clipped = np.clip(G_nonpar, eps, None)

    # reasonable initial guesses if not provided:
    if init is None:
        # scale alphas relative to typical magnitude
        scale = max(np.median(np.abs(G_nonpar_clipped)), 1e-12)
        alphas0 = (np.abs(G_nonpar_clipped[:n_exp]) + 1e-6) * (scale / (np.mean(np.abs(G_nonpar_clipped[:n_exp])) + 1e-12))
        # pick decay rates spread across lag support
        tmax = lags[-1] if len(lags) else 1.0
        rhos0 = np.linspace(1.0 / tmax, 5.0 / (tmax / max(1, n_exp)), n_exp)
    else:
        alphas0 = np.asarray(init['alphas'])
        rhos0 = np.asarray(init['rhos'])

    p0 = np.concatenate([alphas0, rhos0])

    # bounds: alphas >= 0, rhos >= small positive
    lower = np.concatenate([np.zeros(n_exp), np.full(n_exp, 1e-6)])
    upper = np.full(2 * n_exp, np.inf)

    try:
        popt, pcov = curve_fit(lambda t, *p: _multi_exp_vector(t, *p),
                               lags, G_nonpar_clipped, p0=p0, bounds=(lower, upper),
                               maxfev=maxfev)
        alphas = popt[:n_exp]
        rhos = popt[n_exp:]
        G_fit = _multi_exp_vector(lags, *popt)
        ss_res = np.sum((G_nonpar - G_fit) ** 2)
        ss_tot = np.sum((G_nonpar - np.mean(G_nonpar)) ** 2)
        rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        info = dict(pcov=pcov, rsq=rsq, G_fit=G_fit)
    except Exception as ex:
        # fallback: return nonparametric estimate with NaNs
        alphas = np.zeros(n_exp)
        rhos = np.zeros(n_exp)
        info = dict(pcov=None, rsq=np.nan, G_fit=np.full_like(G_nonpar, np.nan), error=str(ex))

    return alphas, rhos, info


# ------------------------
# Bootstrap CI for G(t)
# ------------------------
def bootstrap_propagator_ci(
        df: pd.DataFrame,
        estimator: str = "regression",
        B: int = 200,
        block_size: int = 100,
        seed: Optional[int] = None,
        **est_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Block bootstrap to get CI for estimated G(t).
    estimator: 'regression' or 'wiener'
    Returns lags, lower_bound (2.5%), upper_bound (97.5%)
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n <= block_size:
        raise ValueError("data too short for chosen block size")

    estimates = []
    for b in range(B):
        # sample starting indices for blocks
        starts = rng.integers(0, n - block_size + 1, size=(n // block_size) + 1)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        df_boot = df.iloc[idx].reset_index(drop=True)
        try:
            if estimator == "regression":
                lags, G_hat, _ = estimate_propagator_regression(df_boot, **est_kwargs)
            else:
                lags, G_hat = estimate_propagator_wiener_hopf(df_boot, **est_kwargs)
            estimates.append(G_hat)
        except Exception:
            continue

    if len(estimates) < max(10, B // 5):
        raise RuntimeError("too many bootstrap failures")

    arr = np.vstack(estimates)
    lower = np.percentile(arr, 2.5, axis=0)
    upper = np.percentile(arr, 97.5, axis=0)
    return lags, lower, upper


# ------------------------
# Plotting helper
# ------------------------
def plot_kernel_comparison(lags: np.ndarray,
                           G_nonpar: np.ndarray,
                           alphas: np.ndarray,
                           rhos: np.ndarray,
                           stderr: Optional[np.ndarray] = None,
                           ci_lower: Optional[np.ndarray] = None,
                           ci_upper: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None,
                           logy: bool = False):
    """
    Plot nonparametric estimate (points), fitted multi-exp (line), optional stderr shading.
    """
    if not _HAS_PLT:
        return
    t_fine = np.linspace(0, lags[-1], 300)
    G_fit_fine = np.zeros_like(t_fine)
    for a, r in zip(alphas, rhos):
        G_fit_fine += a * np.exp(-r * t_fine)

    plt.figure(figsize=(7, 4))
    plt.plot(lags, G_nonpar, 'o', label='non-parametric')
    plt.plot(t_fine, G_fit_fine, '-', label='multi-exp fit')
    if stderr is not None:
        plt.fill_between(lags, G_nonpar - 1.96 * stderr, G_nonpar + 1.96 * stderr,
                         color='gray', alpha=0.25, label='±1.96 stderr')
    if ci_lower is not None and ci_upper is not None:
        plt.fill_between(lags, ci_lower, ci_upper, color='orange', alpha=0.2, label='bootstrap 95% CI')

    plt.xlabel('lag (s)')
    plt.ylabel('G(τ)')
    plt.title('Propagator kernel estimate')
    if logy:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Kernel plot saved -> {save_path}")
    else:
        plt.show()


# ------------------------
# Demo / quick test
# ------------------------
def example():
    """Generate synthetic data and estimate propagator (demo)."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))
    from simulate_orderflow import simulate_orderbook_sequence

    T = 300.0  # 5 min
    hawkes_params = dict(
        mu_vec=np.array([0.5, 0.5]),
        A=np.array([[0.2, 0.15],
                    [0.15, 0.2]]),
        betas=np.array([[3.0, 3.0],
                        [3.0, 3.0]])
    )
    propagator_params = dict(
        permanent=0.0,
        transient=dict(
            alphas=np.array([5e-4, 2e-4]),
            betas=np.array([10.0, 50.0])
        ),
        rho=0.0
    )
    trade_size_dist = dict(name='fixed', size=10)

    df = simulate_orderbook_sequence(
        T, hawkes_params, propagator_params, trade_size_dist,
        initial_price=100.0, dt_ms=1.0, seed=42
    )
    print(f"Generated {len(df)} trades.")

    # non-param regression
    lags, G_nonpar, stderr = estimate_propagator_regression(df, dt=1.0, max_lag=60, ridge=1e-8)
    # Wiener-Hopf
    _, G_wh = estimate_propagator_wiener_hopf(df, dt=1.0, max_lag=60, ridge=1e-8)
    # fit multi-exp
    alphas, rhos, info = fit_multi_exponential(lags, G_nonpar, n_exp=2)
    print("Fitted multi-exp:")
    for i, (a, r) in enumerate(zip(alphas, rhos)):
        print(f"  α_{i+1} = {a:.2e},  ρ_{i+1} = {r:.2f}  (R²={info.get('rsq', np.nan):.3f})")

    # bootstrap CI (quick, small B for demo)
    try:
        lags, lower, upper = bootstrap_propagator_ci(df, estimator="regression",
                                                     B=50, block_size=64, seed=123,
                                                     price_col='price', side_col='side', size_col='size',
                                                     dt=1.0, max_lag=60, ridge=1e-8)
    except Exception as e:
        print("Bootstrap failed (demo):", e)
        lower = upper = None

    fig_path = os.path.join(os.path.dirname(__file__), "../../docs/figures/propagator_estimate.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plot_kernel_comparison(lags, G_nonpar, alphas, rhos, stderr=stderr,
                           ci_lower=lower, ci_upper=upper, save_path=fig_path, logy=False)


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    example()