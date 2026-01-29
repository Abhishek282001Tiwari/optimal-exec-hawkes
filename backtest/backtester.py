#!/usr/bin/env python3
"""
backtester.py

Execution back-test engine.

  - Runs any strategy that implements a simple interface
  - Computes cost metrics: mean, std, VaR, max-drawdown
  - Compares strategies and persists results

Author: <you>
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Any, Callable, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------------
# Strategy interface
# ------------------------------------------------------------------
# A strategy function receives:
#   df : DataFrame with columns [time, side, size, price]  (full trace)
#   x0 : shares to sell (positive) or buy (negative)
#   T  : horizon in seconds
#   params : dict of strategy parameters
# and returns a DataFrame with columns
#   [time, shares_remaining, trade_rate, price]
# The trade_rate is the instantaneous speed (shares/sec) at that time.


# ------------------------------------------------------------------
# Core back-test for one strategy
# ------------------------------------------------------------------
def run_backtest(trace_df: pd.DataFrame,
                 strategy_fn: Callable,
                 params: Dict[str, Any],
                 x0: float,
                 T: float,
                 seed: int = 42
                 ) -> Dict[str, Any]:
    """
    Run a single strategy on the provided order-flow trace.

    Returns
    -------
    out : dict
        traj : DataFrame with strategy trajectory
        costs : dict with cost metrics
    """
    rng = np.random.default_rng(seed)

    # call strategy
    traj = strategy_fn(trace_df, x0, T, params)

    # ensure we have a trade_rate column
    if 'trade_rate' not in traj.columns:
        traj['trade_rate'] = -np.gradient(traj['shares_remaining'], traj['time'])

    # cost evaluation (implementation shortfall)
    # assume temporary impact η v^2 and permanent impact γ X v dt
    eta = params.get('eta', 1e-4)
    gamma = params.get('gamma', 5e-5)
    dt = np.gradient(traj['time'])
    dt = np.clip(dt, 1e-3, None)  # avoid zero
    v = traj['trade_rate'].values
    x_mid = (traj['shares_remaining'].values[:-1] + traj['shares_remaining'].values[1:]) / 2
    temp_cost = np.sum(eta * v[:-1] ** 2 * dt[:-1])
    perm_cost = np.sum(gamma * x_mid * v[:-1] * dt[:-1])

    # price slippage vs arrival
    arrival_price = trace_df['price'].iloc[0]
    exit_price = traj['price'].iloc[-1]
    shares_traded = x0 - traj['shares_remaining'].iloc[-1]
    slippage = (exit_price - arrival_price) * shares_traded

    total_cost = temp_cost + perm_cost + slippage

    # metrics
    costs = dict(
        temp_cost=temp_cost,
        perm_cost=perm_cost,
        slippage=slippage,
        total_cost=total_cost
    )

    # realised variance of P&L
    returns = np.diff(traj['price'].values) / traj['price'].values[:-1]
    costs['realised_var'] = np.var(returns)

    # max drawdown of remaining inventory
    x_frac = traj['shares_remaining'] / x0
    running_max = np.maximum.accumulate(x_frac)
    drawdown = (running_max - x_frac) / running_max
    costs['max_drawdown'] = np.max(drawdown)

    # VaR at 95% on cost distribution (bootstrap if few trades)
    n_boot = 500
    boot_costs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(traj), size=len(traj))
        boot_traj = traj.iloc[idx].sort_values('time')
        bv = boot_traj['trade_rate'].values
        bx = (boot_traj['shares_remaining'].values[:-1] +
              boot_traj['shares_remaining'].values[1:]) / 2
        bc = np.sum(eta * bv[:-1] ** 2 * dt[:-1]) + np.sum(gamma * bx * bv[:-1] * dt[:-1])
        boot_costs.append(bc)
    costs['VaR_95'] = np.percentile(boot_costs, 95)

    return dict(traj=traj, costs=costs, params=params)


# ------------------------------------------------------------------
# Multi-strategy comparison
# ------------------------------------------------------------------
def compare_strategies(trace_df: pd.DataFrame,
                       strategies: Dict[str, Tuple[Callable, Dict[str, Any]]],
                       x0: float,
                       T: float,
                       save_dir: str = "docs/figures"
                       ) -> pd.DataFrame:
    """
    Run all strategies and produce summary table + pickles.

    strategies : dict
        name -> (strategy_fn, params)
    """
    os.makedirs(save_dir, exist_ok=True)

    summary = []
    results = {}

    for name, (fn, params) in tqdm(strategies.items(), desc="Back-testing"):
        res = run_backtest(trace_df, fn, params, x0, T)
        results[name] = res
        summary.append({
            'strategy': name,
            'mean_cost': res['costs']['total_cost'],
            'std_cost': np.sqrt(res['costs'].get('realised_var', 0)),
            'VaR_95': res['costs']['VaR_95'],
            'max_drawdown': res['costs']['max_drawdown'],
            'temp_cost': res['costs']['temp_cost'],
            'perm_cost': res['costs']['perm_cost']
        })

    summary_df = pd.DataFrame(summary)
    csv_path = os.path.join(save_dir, "strategy_comparison.csv")
    summary_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Summary saved -> {csv_path}")

    pkl_path = os.path.join(save_dir, "backtest_results.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Raw results pickled -> {pkl_path}")

    return summary_df


# ------------------------------------------------------------------
# 5. Ready-made strategy wrappers
# ------------------------------------------------------------------
def twap_strategy(trace_df: pd.DataFrame,
                  x0: float,
                  T: float,
                  params: Dict[str, Any]
                  ) -> pd.DataFrame:
    """Uniform trading rate."""
    N = params.get('N', 60)
    t = np.linspace(0, T, N + 1)
    x = x0 * (1 - t / T)
    v = np.full_like(t, x0 / T)
    # price interpolated
    price = np.interp(t, trace_df['time'], trace_df['price'])
    return pd.DataFrame({'time': t, 'shares_remaining': x,
                         'trade_rate': v, 'price': price})


def ac_strategy(trace_df: pd.DataFrame,
                x0: float,
                T: float,
                params: Dict[str, Any]
                ) -> pd.DataFrame:
    """Almgren–Chriss hyperbolic schedule."""
    from models.almgren_chriss import compute_AC_schedule
    eta = params['eta']
    gamma = params['gamma']
    sigma = params['sigma']
    N = params.get('N', 60)
    dt = T / N
    t, x = compute_AC_schedule(x0, T, eta, gamma, sigma, dt)
    v = -np.gradient(x, t)
    price = np.interp(t, trace_df['time'], trace_df['price'])
    return pd.DataFrame({'time': t, 'shares_remaining': x,
                         'trade_rate': v, 'price': price})


def ow_strategy(trace_df: pd.DataFrame,
                x0: float,
                T: float,
                params: Dict[str, Any]
                ) -> pd.DataFrame:
    """Obizhaeva–Wang with exponential resilience."""
    from models.obizhaeva_wang import discrete_OW_simulation
    rho = params['rho']
    eta = params['eta']
    pi = params['pi']
    N = params.get('N', 60)
    t, x, info = discrete_OW_simulation(x0, N, rho, eta, pi, T)
    v = info['schedule']
    v = np.append(v, 0.0)  # append zero to match length
    price = np.interp(t, trace_df['time'], trace_df['price'])
    return pd.DataFrame({'time': t, 'shares_remaining': x,
                         'trade_rate': v, 'price': price})


def hawkes_lq_strategy(trace_df: pd.DataFrame,
                       x0: float,
                       T: float,
                       params: Dict[str, Any]
                       ) -> pd.DataFrame:
    """Hawkes LQ control (pre-computed gains)."""
    from models.hawkes_control import simulate_controlled_execution

    # unpack pre-computed gains
    gains = params['gains']
    prop_a = params['propagator_alphas']
    prop_b = params['propagator_betas']
    prop_rho = params['propagator_rho']
    hk_b = params['hawkes_betas']
    hk_A = params['hawkes_A']

    # initial state
    D0 = np.zeros(len(prop_a))
    lambda0 = np.array([0.5, 0.5])  # crude; better estimate from trace
    res = simulate_controlled_execution(
        x0, D0, lambda0, gains,
        prop_a, prop_b, prop_rho,
        hk_b, hk_A, sigma_noise=0.0
    )

    # build DataFrame
    df_out = pd.DataFrame({
        'time': res['t'],
        'shares_remaining': res['X'],
        'trade_rate': res['v'],
        'price': np.interp(res['t'], trace_df['time'], trace_df['price'])
    })
    return df_out


# ------------------------------------------------------------------
# 6. Quick demo
# ------------------------------------------------------------------
def example():
    """Generate a trace and compare strategies."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sim'))
    from simulate_orderflow import simulate_orderbook_sequence

    T = 60.0
    x0 = 100_000
    # use same parameters as earlier scripts
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
        rho=5.0
    )
    trade_size_dist = dict(name='fixed', size=10)

    trace_df = simulate_orderbook_sequence(
        T, hawkes_params, propagator_params, trade_size_dist,
        initial_price=100.0, dt_ms=1.0, seed=42
    )
    print(f"Generated trace with {len(trace_df)} events.")

    # pre-compute Hawkes-LQ gains
    from models.hawkes_control import compute_feedback_gains
    gains = compute_feedback_gains(
        propagator_alphas=np.array([5e-4, 2e-4]),
        propagator_betas=np.array([10.0, 50.0]),
        propagator_rho=5.0,
        hawkes_alphas=np.array([0.2, 0.15]),
        hawkes_betas=np.array([3.0, 3.0]),
        hawkes_A=np.array([[0.2, 0.15],
                           [0.15, 0.2]]),
        eta=1e-4,
        gamma=5e-5,
        theta=1e-4,
        T=T,
        N=60,
        agent_excitation=False
    )

    strategies = {
        'TWAP': (twap_strategy, dict(N=60, eta=1e-4, gamma=5e-5)),
        'AC': (ac_strategy, dict(eta=1e-4, gamma=5e-5, sigma=0.02, N=60)),
        'OW': (ow_strategy, dict(rho=1 / 30, eta=1e-4, pi=5e-5, N=60)),
        'Hawkes-LQ': (hawkes_lq_strategy, dict(
            gains=gains,
            propagator_alphas=np.array([5e-4, 2e-4]),
            propagator_betas=np.array([10.0, 50.0]),
            propagator_rho=5.0,
            hawkes_betas=np.array([3.0, 3.0]),
            hawkes_A=np.array([[0.2, 0.15],
                               [0.15, 0.2]]),
            eta=1e-4,
            gamma=5e-5))
    }

    summary = compare_strategies(trace_df, strategies, x0, T)
    print("\nSummary:")
    print(summary.to_string(index=False))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    example()
    # backtest/backtester.py  (add at the end if missing)
__all__ = [
    'run_backtest',
    'compare_strategies',
    'twap_strategy',
    'ac_strategy',
    'ow_strategy',
    'hawkes_lq_strategy'
]
