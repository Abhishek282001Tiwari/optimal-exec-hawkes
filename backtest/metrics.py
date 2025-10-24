#!/usr/bin/env python3
"""
metrics.py – optional centralised cost / risk metrics
"""

import numpy as np
from typing import Dict, Any


def implementation_shortfall(trade_list: np.ndarray,
                             price_path: np.ndarray,
                             eta: float,
                             gamma: float) -> Dict[str, float]:
    """
    Temporary + permanent impact cost.
    trade_list : shares traded per interval (array)
    price_path : mid-price path (len = len(trade_list)+1)
    """
    v = trade_list
    dt = 1.0  # assumed 1-second steps; adjust if needed
    x_mid = (price_path[:-1] + price_path[1:]) / 2
    temp = eta * np.sum(v ** 2)
    perm = gamma * np.sum(x_mid * v)
    return dict(temp_cost=temp, perm_cost=perm, total=temp + perm)


def cvar(cost_samples: np.ndarray, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk (average of worst (1-α) tail)."""
    return np.mean(np.sort(cost_samples)[:int((1 - alpha) * len(cost_samples))])


def max_drawdown(x: np.ndarray) -> float:
    """Max drawdown of inventory fraction x(t)."""
    running_max = np.maximum.accumulate(x)
    drawdown = (running_max - x) / running_max
    return np.max(drawdown)