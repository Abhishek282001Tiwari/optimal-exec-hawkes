#!/usr/bin/env python3
"""
plot_execution.py
Inventory paths + cost distribution across strategies
"""
import os, pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'figure.dpi': 300})
os.makedirs("docs/figures", exist_ok=True)

def plot_inventory_paths(pkl_file="docs/figures/backtest_results.pkl", save_pdf=True):
    with open(pkl_file,'rb') as f: res = pickle.load(f)
    fig, ax = plt.subplots(figsize=(5,2.2))
    for name, data in res.items():
        traj = data['traj']
        ax.plot(traj['time'], traj['shares_remaining']/1e5, label=name, lw=1.4)
    ax.set_xlabel("time (s)"); ax.set_ylabel("remaining fraction")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig("docs/figures/inventory_paths.pdf" if save_pdf else "docs/figures/inventory_paths.png",
                bbox_inches='tight')
    print("saved inventory_paths")

def plot_cost_distribution(pkl_file="docs/figures/backtest_results.pkl", save_pdf=True):
    with open(pkl_file,'rb') as f: res = pickle.load(f)
    costs = {k: v['costs']['total_cost'] for k,v in res.items()}
    fig, ax = plt.subplots(figsize=(3,2.2))
    ax.bar(costs.keys(), costs.values(), color=['C0','C1','C2','C3'])
    ax.set_ylabel("total cost"); plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig("docs/figures/cost_bar.pdf" if save_pdf else "docs/figures/cost_bar.png",
                bbox_inches='tight')
    print("saved cost_bar")

if __name__=="__main__":
    plot_inventory_paths(); plot_cost_distribution()