#!/usr/bin/env  python3
"""
plot_intensity.py
Intensity path and event times
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300})
os.makedirs("docs/figures", exist_ok=True)

def plot_intensity_trace(events_file="sim/sample_orderflow.csv",
                         save_pdf=True):
    df = pd.read_csv(events_file)
    buy  = df[df['side']==1]; sell = df[df['side']==-1]

    fig, ax = plt.subplots(figsize=(5,2))
    ax.scatter(buy['time'],  buy['price'],  marker='^', s=8, c='g', alpha=.7, label='buy')
    ax.scatter(sell['time'], sell['price'], marker='v', s=8, c='r', alpha=.7, label='sell')
    ax.set_xlabel("time (s)"); ax.set_ylabel("price"); ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/intensity_events.pdf" if save_pdf else "docs/figures/intensity_events.png",
                bbox_inches='tight')
    print("saved intensity_events")

if __name__=="__main__":
    plot_intensity_trace()