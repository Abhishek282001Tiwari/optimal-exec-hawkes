#!/usr/bin/env python3
"""
simulate_orderflow.py

Simulate a full order-flow trace:
  * event times from a bivariate Hawkes (buy / sell)
  * trade sizes drawn from a size distribution
  * mid-price evolves with propagator model (multi-exponential transient impact +
    permanent impact)

Author: <you>
"""

from __future__ import annotations

import os
import time
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# local import
try:
    from .simulate_hawkes import simulate_bivariate_hawkes
except ImportError:
    # allow direct run
    from simulate_hawkes import simulate_bivariate_hawkes

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_DT_MS = 1.0  # 1 ms discrete grid


# ------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------
def simulate_orderbook_sequence(
        T: float,
        hawkes_params: Dict[str, Any],
        propagator_params: Dict[str, Any],
        trade_size_dist: Dict[str, Any],
        initial_price: float = 100.0,
        dt_ms: float = DEFAULT_DT_MS,
        seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate order-flow with propagator price impact.

    Parameters
    ----------
    T : float
        Horizon in seconds.
    hawkes_params : dict
        Must contain keys
            mu_vec  : (2,) baseline intensities [buy, sell]
            A       : (2,2) excitation matrix
            betas   : (2,) or (2,2) decay rates
    propagator_params : dict
        Contains
            permanent : float, permanent impact coefficient Λ
            transient : dict, keys 'alphas' and 'betas' for
                        G(t)=Σ α_i exp(-β_i t)  (both np.ndarray same length)
            rho       : float, mean-reversion speed of transient impact
    trade_size_dist : dict
        Either
            {'name': 'lognormal', 'mean': μ, 'std': σ}
        or
            {'name': 'fixed', 'size': Q}
    initial_price : float
    dt_ms : float
        Discrete time step in milliseconds for integration.
    seed : int, optional

    Returns
    -------
    df : pd.DataFrame
        Columns: time, side (+1 buy / -1 sell), size, price
    """
    rng = np.random.default_rng(seed)

    # ---------- 1. Event times ----------
    times_0, times_1, dims = simulate_bivariate_hawkes(
        T,
        hawkes_params['mu_vec'],
        hawkes_params['A'],
        hawkes_params['betas'],
        seed=rng.integers(0, 2**31)
    )
    # stack & sort
    all_times = np.concatenate([times_0, times_1])
    all_sides = np.concatenate([np.ones_like(times_0), -np.ones_like(times_1)])
    idx = np.argsort(all_times)
    times = all_times[idx]
    sides = all_sides[idx]

    # ---------- 2. Trade sizes ----------
    dist = trade_size_dist['name']
    if dist == 'lognormal':
        mean = trade_size_dist['mean']
        std = trade_size_dist['std']
        # parameterise log-normal
        σ2 = np.log((std / mean) ** 2 + 1)
        μ = np.log(mean) - σ2 / 2
        sizes = rng.lognormal(mean=μ, sigma=np.sqrt(σ2), size=times.shape)
        sizes = np.round(sizes).astype(int)
        sizes = np.maximum(sizes, 1)
    elif dist == 'fixed':
        q = trade_size_dist['size']
        sizes = np.full_like(times, q, dtype=int)
    else:
        raise ValueError("Unsupported trade_size_dist")

    # ---------- 3. Price impact ----------
    Λ = propagator_params['permanent']
    G_alphas = np.asarray(propagator_params['transient']['alphas'], dtype=float)
    G_betas = np.asarray(propagator_params['transient']['betas'], dtype=float)
    rho = propagator_params['rho']

    # discrete grid
    dt = dt_ms / 1000.0  # seconds
    n_steps = int(np.ceil(T / dt))
    grid = np.arange(n_steps + 1) * dt
    price_path = np.zeros_like(grid)
    price_path[0] = initial_price

    # state variables
    permanent_impact = 0.0
    transient_impact = np.zeros_like(G_alphas)  # one per exponential

    # helper: which trades fall in which interval
    time_bins = np.searchsorted(grid, times, side='right') - 1  # index in [0, n_steps-1]

    k = 0  # trade index
    for i in range(1, n_steps + 1):
        # ---- transient decay ----
        transient_impact *= np.exp(-rho * dt)

        # ---- process trades in this interval ----
        while k < len(times) and time_bins[k] == i - 1:
            eps = sides[k]  # +1 or -1
            q = sizes[k]
            # permanent
            permanent_impact += Λ * eps * q
            # transient
            transient_impact += G_alphas * eps * q
            k += 1

        # ---- total impact ----
        impact = permanent_impact + transient_impact.sum()
        price_path[i] = initial_price + impact

    # build DataFrame
    df = pd.DataFrame({
        'time': times,
        'side': sides.astype(int),
        'size': sizes,
    })
    # assign prices by mapping event time to nearest grid
    event_idx = np.searchsorted(grid, times, side='right') - 1
    df['price'] = price_path[event_idx]

    return df


# ------------------------------------------------------------------
# Toy example
# ------------------------------------------------------------------
def toy_example():
    """Run a short simulation and write CSV."""
    T = 60.0  # 1 minute
    hawkes_params = dict(
        mu_vec=np.array([0.5, 0.5]),  # 0.5 Hz baseline each
        A=np.array([[0.3, 0.2],
                    [0.2, 0.3]]),
        betas=np.array([[2.0, 2.0],
                        [2.0, 2.0]])
    )
    propagator_params = dict(
        permanent=1e-4,  # permanent impact coeff
        transient=dict(
            alphas=np.array([5e-4, 2e-4]),
            betas=np.array([10.0, 50.0])
        ),
        rho=5.0
    )
    trade_size_dist = dict(name='lognormal', mean=10, std=5)

    print("Simulating order flow...")
    df = simulate_orderbook_sequence(
        T, hawkes_params, propagator_params, trade_size_dist,
        initial_price=100.0, dt_ms=1.0, seed=42
    )
    print(f"Generated {len(df)} trades.")

    outfile = os.path.join(os.path.dirname(__file__), "sample_orderflow.csv")
    df.to_csv(outfile, index=False, float_format="%.6f")
    print(f"Saved -> {outfile}")

    # quick sanity plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['time'], df['price'], lw=0.8, label='mid-price')
        buys = df[df['side'] == 1]
        sells = df[df['side'] == -1]
        ax.scatter(buys['time'], buys['price'], marker='^', color='g', s=10, alpha=0.7, label='buy')
        ax.scatter(sells['time'], sells['price'], marker='v', color='r', s=10, alpha=0.7, label='sell')
        ax.set_xlabel("time (s)")
        ax.set_ylabel("price")
        ax.legend()
        ax.set_title("Sample order-flow with propagator impact")
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), "sample_orderflow.png"), dpi=150)
        print("Plot saved -> sample_orderflow.png")
    except ImportError:
        pass


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    toy_example()