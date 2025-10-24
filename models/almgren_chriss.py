#!/usr/bin/env python3
"""
almgren_chriss.py

Almgren–Chriss optimal execution baseline.

Features:
  - Closed-form continuous (hyperbolic) schedule
  - Discrete-time sampled version
  - Implementation shortfall cost evaluation
  - Optional plotting and runtime diagnostics

Author: Abhishek Tiwari
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


# ------------------------------------------------------------------
# 1. Closed-form continuous schedule
# ------------------------------------------------------------------
def compute_AC_schedule(
    x0: float,
    T: float,
    eta: float,
    gamma: float,
    sigma: float,
    dt: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Almgren–Chriss optimal (hyperbolic) liquidation schedule.

    Parameters
    ----------
    x0 : float
        Initial shares to sell (positive = sell, negative = buy).
    T : float
        Total trading horizon (seconds).
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    sigma : float
        Volatility of the underlying (σ > 0).
    dt : float, optional
        If provided, returns a discrete grid. Otherwise continuous 200-point grid.

    Returns
    -------
    t : np.ndarray
        Time grid from 0 to T.
    x : np.ndarray
        Remaining shares over time (x(0)=x0, x(T)=0).
    """
    # stability guard
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.sqrt(gamma * sigma ** 2 / eta)

    if not np.isfinite(k) or k <= 0:
        raise ValueError(f"Invalid hyperbolic parameter k={k:.4e}")

    t = np.linspace(0, T, int(T / dt) + 1) if dt else np.linspace(0, T, 200)
    x = x0 * np.sinh(k * (T - t)) / np.sinh(k * T)
    return t, x


# ------------------------------------------------------------------
# 2. Discrete-time schedule
# ------------------------------------------------------------------
def discrete_AC_schedule(
    x0: float,
    N: int,
    eta: float,
    gamma: float,
    sigma: float,
    T: float = 1.0
) -> np.ndarray:
    """
    Compute discrete trade list [u_0, u_1, ..., u_{N-1}] with Σu_n = x0.

    u_n > 0 → sell.
    """
    dt = T / N
    _, x = compute_AC_schedule(x0, T, eta, gamma, sigma, dt)
    u = -np.diff(x)  # trade increments
    return u


# ------------------------------------------------------------------
# 3. Cost evaluation
# ------------------------------------------------------------------
def evaluate_cost(
    schedule: np.ndarray,
    price_path: np.ndarray,
    eta: float,
    gamma: float,
    dt: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute implementation shortfall for a given execution schedule.

    Parameters
    ----------
    schedule : np.ndarray
        Trade list u_n (shares sold at step n, positive=sell).
    price_path : np.ndarray
        Mid-price path (length = len(schedule)+1).
    eta : float
        Temporary impact coefficient.
    gamma : float
        Permanent impact coefficient.
    dt : float, optional
        Time step (seconds).

    Returns
    -------
    shortfall : float
        Implementation shortfall in currency units.
    info : dict
        Components: temporary, permanent, variance.
    """
    if len(schedule) != len(price_path) - 1:
        raise ValueError("price_path must be one element longer than schedule")

    u = schedule
    P = price_path

    # Vectorized computation
    temp_cost = eta * np.sum(u ** 2)
    perm_cost = gamma * np.dot(u, P[:-1]) + 0.5 * gamma * np.sum(u ** 2)
    variance = np.sum(np.diff(P) ** 2)

    shortfall = temp_cost + perm_cost
    info = dict(temporary=temp_cost, permanent=perm_cost, variance=variance)
    return shortfall, info


# ------------------------------------------------------------------
# 4. Example / Sanity Test
# ------------------------------------------------------------------
def example(verbose: bool = True, plot: bool = True) -> None:
    """Run a demonstration of Almgren–Chriss schedule and cost."""
    x0, T = 100_000, 60.0
    eta, gamma, sigma = 1e-4, 5e-5, 0.02

    t, x = compute_AC_schedule(x0, T, eta, gamma, sigma)
    u_disc = discrete_AC_schedule(x0, N=60, eta=eta, gamma=gamma, sigma=sigma, T=T)

    if verbose:
        print(f"\n[INFO] Almgren–Chriss schedule")
        print(f"  Initial shares : {x0}")
        print(f"  Horizon (sec)  : {T}")
        print(f"  k parameter    : {np.sqrt(gamma * sigma ** 2 / eta):.5f}")

    # Generate mock price path
    rng = np.random.default_rng(42)
    dt = T / len(u_disc)
    returns = rng.normal(0, sigma * np.sqrt(dt), size=len(u_disc))
    price_path = 100.0 * np.exp(np.concatenate([[0], np.cumsum(returns)]))

    # Evaluate shortfall
    shortfall, info = evaluate_cost(u_disc, price_path, eta, gamma, dt)

    if verbose:
        print(f"\n[RESULT] Implementation shortfall:")
        print(f"  Total shortfall : {shortfall:.6f}")
        for k, v in info.items():
            print(f"  {k.capitalize():<10}: {v:.6f}")

    # Optional plot
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(t, x / x0, label='Remaining fraction', lw=2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fraction of shares")
            ax.set_title("Almgren–Chriss Optimal Execution Schedule")
            ax.grid(alpha=0.3)
            ax.legend()

            # Ensure output directory exists
            out_dir = "../../docs/figures"
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "AC_schedule.png")

            fig.tight_layout()
            fig.savefig(path, dpi=160)
            if verbose:
                print(f"\n[INFO] Figure saved → {path}")

        except ImportError:
            print("[WARN] matplotlib not available, skipping plot.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    example()