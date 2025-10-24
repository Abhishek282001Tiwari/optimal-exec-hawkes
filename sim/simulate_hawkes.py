#!/usr/bin/env python3
"""
simulate_hawkes.py

Simulate univariate and bivariate Hawkes processes using
1. Ogata’s thinning algorithm (exact)
2. Cluster (branching) algorithm (fast when branching ratio < 1)

Author: <Abhishek Tiwari>
"""

from __future__ import annotations

import os
import time
from typing import Tuple, List, Optional

import numpy as np
from scipy import special

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    # dummy decorators
    def njit(*args, **kwargs):
        return lambda f: f
    prange = range

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@njit(cache=True)
def _exponential_intensity(t: float,
                           times: np.ndarray,
                           alphas: np.ndarray,
                           betas: np.ndarray) -> float:
    """
    Compute λ(t) for exponential kernels with possibly multiple exponentials.
    times: array of past event times <= t
    alphas, betas: 1-D arrays of same length (mixture components)
    """
    lam = 0.0
    for i in range(alphas.shape[0]):
        lam += alphas[i] * np.sum(np.exp(-betas[i] * (t - times)))
    return lam


# ------------------------------------------------------------------
# 1. Ogata’s thinning
# ------------------------------------------------------------------
def simulate_hawkes_ogata(T: float,
                          mu: float,
                          alphas: np.ndarray,
                          betas: np.ndarray,
                          seed: Optional[int] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate univariate Hawkes process on [0,T] using Ogata’s thinning.

    Parameters
    ----------
    T : float
        Horizon time.
    mu : float
        Baseline intensity.
    alphas : np.ndarray, shape (K,)
        Excitation amplitudes for K exponential kernels.
    betas : np.ndarray, shape (K,)
        Decay rates for K exponential kernels.
    seed : int, optional
        Random seed.

    Returns
    -------
    times : np.ndarray
        Sorted array of event times.
    marks : np.ndarray
        Array of zeros (place-holder for future mark support).
    """
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if alphas.shape != betas.shape:
        raise ValueError("alphas and betas must have same shape")
    K = alphas.shape[0]

    times: List[float] = []
    t = 0.0
    # Upper bound for thinning: μ + Σ α_i (since ∫β e^{-βs}ds = 1)
    M = mu + np.sum(alphas)

    while t < T:
        # next candidate with exponential gap with rate M
        tau = rng.exponential(scale=1.0 / M)
        t += tau
        if t > T:
            break
        # evaluate current intensity
        lam_t = mu + _exponential_intensity(t, np.array(times), alphas, betas)
        # accept with probability lam_t / M
        if rng.uniform(0, 1) < lam_t / M:
            times.append(t)

    times_arr = np.array(times)
    marks_arr = np.zeros_like(times_arr)
    return times_arr, marks_arr


# ------------------------------------------------------------------
# 2. Cluster (branching) algorithm
# ------------------------------------------------------------------
def simulate_hawkes_cluster(T: float,
                            mu: float,
                            alphas: np.ndarray,
                            betas: np.ndarray,
                            seed: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast simulation via branching (cluster) structure when branching ratio < 1.
    Generates exact sample on [0,T].

    Parameters
    ----------
    T, mu, alphas, betas, seed : same as simulate_hawkes_ogata

    Returns
    -------
    times, marks : same as above
    """
    rng = np.random.default_rng(seed)
    alphas = np.asarray(alphas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if alphas.shape != betas.shape:
        raise ValueError("alphas and betas must have same shape")
    n_tot = 0
    # immigrants via Poisson with rate mu
    immigrants = rng.poisson(lam=mu * T)
    immigrant_times = rng.uniform(0, T, size=immigrants)
    all_times: List[float] = list(immigrant_times)

    # branching ratio
    branching_ratio = np.sum(alphas)
    if branching_ratio >= 1.0:
        import warnings
        warnings.warn("Branching ratio >= 1; cluster algorithm may not terminate.")

    # offspring generation recursively
    def generate_offspring(parent_times: np.ndarray) -> np.ndarray:
        nonlocal n_tot
        if parent_times.size == 0:
            return np.array([], dtype=float)
        # number of offspring for each parent ~ Poisson(Σα)
        n_offspring = rng.poisson(branching_ratio, size=parent_times.size)
        offspring_times = []
        for i, t_p in enumerate(parent_times):
            if n_offspring[i] == 0:
                continue
            # draw which kernel triggered
            kernel_probs = alphas / branching_ratio
            kernel_choice = rng.choice(len(alphas), size=n_offspring[i], p=kernel_probs)
            # draw decay times
            for k in kernel_choice:
                dt = rng.exponential(scale=1.0 / betas[k])
                t_child = t_p + dt
                if t_child <= T:
                    offspring_times.append(t_child)
        return np.array(offspring_times, dtype=float)

    # iterate generations
    current = np.array(all_times, dtype=float)
    while current.size > 0:
        current = generate_offspring(current)
        all_times.extend(current.tolist())

    times_arr = np.sort(np.array(all_times, dtype=float))
    marks_arr = np.zeros_like(times_arr)
    return times_arr, marks_arr


# ------------------------------------------------------------------
# 3. Bivariate Hawkes
# ------------------------------------------------------------------
def simulate_bivariate_hawkes(T: float,
                              mu_vec: np.ndarray,
                              A: np.ndarray,
                              betas: np.ndarray,
                              seed: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate bivariate Hawkes using Ogata thinning (vectorised).

    Parameters
    ----------
    T : float
        Horizon.
    mu_vec : np.ndarray, shape (2,)
        Baselines for dim 0 and 1.
    A : np.ndarray, shape (2,2)
        Excitation matrix A[i,j] = alpha_{i<-j} (i excited by j).
    betas : np.ndarray, shape (2,) or (2,2)
        Decay rates. If 1-D same for all kernels, if 2-D separate.
    seed : int, optional

    Returns
    -------
    times_0, times_1 : np.ndarray
        Event times for each component.
    marks : np.ndarray
        0/1 indicating which dimension.
    """
    rng = np.random.default_rng(seed)
    mu_vec = np.asarray(mu_vec, dtype=float)
    A = np.asarray(A, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if mu_vec.shape != (2,):
        raise ValueError("mu_vec must be length-2")
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2 matrix")
    if betas.ndim == 1 and betas.shape[0] == 2:
        betas = np.array([[betas[0], betas[1]],
                          [betas[0], betas[1]]], dtype=float)
    elif betas.shape != (2, 2):
        raise ValueError("betas must be (2,) or (2,2)")

    times: List[Tuple[float, int]] = []  # (time, dimension)
    t = 0.0
    M = np.sum(mu_vec) + np.sum(A)  # crude upper bound

    while t < T:
        tau = rng.exponential(scale=1.0 / M)
        t += tau
        if t > T:
            break
        # evaluate both intensities
        lam = np.zeros(2)
        for d in 0, 1:
            lam[d] = mu_vec[d]
            # add influence from past events
            for (t_p, d_p) in times:
                lam[d] += A[d, d_p] * np.exp(-betas[d, d_p] * (t - t_p))
        lam_tot = lam.sum()
        if rng.uniform(0, 1) < lam_tot / M:
            # choose dimension proportional to intensity
            dim = rng.choice(2, p=lam / lam_tot)
            times.append((t, dim))

    times_arr = np.array([x[0] for x in times], dtype=float)
    dims = np.array([x[1] for x in times], dtype=int)
    # split
    times_0 = times_arr[dims == 0]
    times_1 = times_arr[dims == 1]
    return times_0, times_1, dims


# ------------------------------------------------------------------
# Demo / Plot
# ------------------------------------------------------------------
def example():
    import matplotlib.pyplot as plt

    T = 1000.0
    mu = 0.1
    alpha = 0.5
    beta = 1.0

    print("Running example simulation...")
    times, _ = simulate_hawkes_ogata(T, mu, np.array([alpha]), np.array([beta]), seed=42)
    print(f"Generated {len(times)} events.")

    # estimate empirical rate
    empirical_rate = len(times) / T
    theoretical_rate = mu / (1 - alpha / beta)
    print(f"Empirical rate: {empirical_rate:.3f}")
    print(f"Theoretical rate: {theoretical_rate:.3f}")

    # plot intensity on a grid
    t_grid = np.linspace(0, T, 10000)
    lam_grid = np.array([mu + _exponential_intensity(t, times,
                                                     np.array([alpha]),
                                                     np.array([beta]))
                         for t in t_grid])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_grid, lam_grid, lw=0.8, label="λ(t)")
    ax.scatter(times, np.zeros_like(times), marker="|", color="black", s=4, alpha=0.7, label="events")
    ax.set_xlabel("time")
    ax.set_ylabel("intensity")
    ax.set_title("Hawkes process sample path")
    ax.legend()
    fig.tight_layout()

    os.makedirs("../docs/figures", exist_ok=True)
    fig.savefig("../docs/figures/hawkes_example.png", dpi=150)
    print("Figure saved to ../docs/figures/hawkes_example.png")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    example()