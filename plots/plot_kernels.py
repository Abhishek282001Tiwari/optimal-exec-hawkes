#!/usr/bin/env python3
"""
plot_kernel.py
Propagator kernel: non-parametric vs multi-exp fit
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, os, pickle
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300})
os.makedirs("docs/figures", exist_ok=True)

def plot_kernel_fit(np_file="docs/figures/kernel_nonpar.csv",
                    pkl_fit="docs/figures/kernel_fit.pkl", save_pdf=True):
    # non-parametric
    df = pd.read_csv(np_file)
    lags, G_nonpar = df['lag'].values, df['G'].values
    # fit
    with open(pkl_fit,'rb') as f: fit = pickle.load(f)
    alphas, rhos = fit['alphas'], fit['rhos']
    t_fine = np.linspace(0, lags[-1], 200)
    G_fit = np.sum([a*np.exp(-r*t_fine) for a,r in zip(alphas,rhos)], axis=0)

    fig, ax = plt.subplots(figsize=(3.5,2.2))
    ax.plot(lags, G_nonpar*1e4, 'o', ms=3, label='non-parametric')
    ax.plot(t_fine, G_fit*1e4, '-', label='multi-exp fit')
    ax.set_xlabel("lag (s)"); ax.set_ylabel(r"$G(\tau)\,(10^{-4})$")
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig("docs/figures/kernel_fit.pdf" if save_pdf else "docs/figures/kernel_fit.png",
                bbox_inches='tight')
    print("saved kernel_fit")

if __name__=="__main__":
    plot_kernel_fit()