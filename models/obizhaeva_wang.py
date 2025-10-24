#!/usr/bin/env python3
"""
obizhaeva_wang.py

Obizhaeva–Wang optimal execution with exponential resilience.

  - Closed-form schedule: initial block, continuous trading, final block
  - Discrete-time simulation to verify cost

Author: Abhishek Tiwari
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional
import pandas as pd


# ------------------------------------------------------------------
# 1. Closed-form OW schedule
# ------------------------------------------------------------------
def compute_OW_schedule(x0: float,
                        T: float,
                        rho: float,
                        eta: float,
                        pi: float
                        ) -> Dict[str, Any]:
    """
    Obizhaeva–Wang optimal schedule.

    Parameters
    ----------
    x0 : float
        Shares to sell (positive) or buy (negative).
    T : float
        Horizon (seconds).
    rho : float
        Resilience speed (1/seconds).
    eta : float
        Temporary impact coefficient (same units as price).
    pi : float
        Permanent impact coefficient.

    Returns
    -------
    sched : dict
        Keys: 'x0_block', 'xT_block', 'u_cont', 'lambda_star'
    """
    # OW parameters
    λ = np.sqrt(rho * (pi + rho * eta))
    sinh = np.sinh(λ * T)
    cosh = np.cosh(λ * T)

    # initial & final blocks
    denom = sinh + (λ * cosh / rho)
    x0_block = x0 * sinh / denom
    xT_block = x0 * λ / (rho * sinh + λ * cosh)

    # continuous trading rate
    denom_cont = λ * cosh + rho * sinh
    A = x0 * λ * (λ ** 2 - rho ** 2) / denom_cont

    def u_cont(t: float) -> float:
        """Continuous trading rate function."""
        if λ != rho:
            return A * np.sinh(λ * (T - t)) / (λ ** 2 - rho ** 2)
        else:
            return A * (T - t)

    sched = dict(
        x0_block=x0_block,
        xT_block=xT_block,
        u_cont=u_cont,
        lambda_star=λ
    )
    return sched


# ------------------------------------------------------------------
# 2. Discrete-time simulation
# ------------------------------------------------------------------
def discrete_OW_simulation(x0: float,
                           N: int,
                           rho: float,
                           eta: float,
                           pi: float,
                           T: float = 1.0,
                           seed: Optional[int] = None
                           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Simulate the OW model on N equal steps and compute expected cost.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0

    # OW schedule
    sched = compute_OW_schedule(x0, T, rho, eta, pi)
    λ = sched['lambda_star']

    # pre-compute continuous trades (mid-point rule)
    u_cont = np.array([sched['u_cont'](t[i] + dt / 2) * dt for i in range(N)])

    # initial & final blocks
    u0 = sched['x0_block']
    uT = sched['xT_block']

    # trade list
    u = np.zeros(N)
    u[0] += u0
    u += u_cont
    u[-1] += uT
    # enforce exact total traded volume
    u *= x0 / u.sum()

    # remaining shares
    for i in range(1, N + 1):
        x[i] = x[i - 1] - u[i - 1]

    # cost evaluation (expected, ignoring noise)
    Q = np.cumsum(u) - u  # cumulative before step
    temp_cost = eta * np.sum(u ** 2)
    perm_cost = pi * np.sum(u * Q)
    total_cost = temp_cost + perm_cost

    info = dict(
        temp_cost=temp_cost,
        perm_cost=perm_cost,
        total_cost=total_cost,
        schedule=u,
        remaining=x
    )
    return t, x, info


# ------------------------------------------------------------------
# 3. Example
# ------------------------------------------------------------------
def example():
    x0 = 50_000  # sell
    T = 60.0  # 1 min
    rho = 1 / 30  # half-life ~ 20 s
    eta = 2e-4
    pi = 5e-5

    sched = compute_OW_schedule(x0, T, rho, eta, pi)
    print("Obizhaeva–Wang schedule:")
    print(f"  initial block: {sched['x0_block']:.0f} shares")
    print(f"  final block:   {sched['xT_block']:.0f} shares")
    print(f"  λ* = {sched['lambda_star']:.4f}")

    # discrete simulation
    t, x, info = discrete_OW_simulation(x0, N=60, rho=rho, eta=eta, pi=pi, T=T, seed=42)
    print("\nDiscrete simulation:")
    print(f"  expected cost: {info['total_cost']:.2f} "
          f"(temp={info['temp_cost']:.2f}, perm={info['perm_cost']:.2f})")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, x / x0, label='remaining fraction')
        ax.scatter(
            [0, T],
            [1 - sched['x0_block'] / x0, sched['xT_block'] / x0],
            color='C3', zorder=5, label='blocks'
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("fraction remaining")
        ax.set_title("Obizhaeva–Wang schedule")
        ax.legend()
        fig.tight_layout()
        fig.savefig("OW_schedule.png", dpi=150)
        print("Figure saved -> OW_schedule.png")
    except ImportError:
        pass


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    example()