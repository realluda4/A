# -*- coding: utf-8 -*-
"""
Normal vs Fault 신호 비교 플롯
"""
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_channel(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    if not {"t","channel","value"}.issubset(df.columns):
        raise ValueError("CSV must have columns t, channel, value")
    g = df[df["channel"]==channel_hint]
    if g.empty:
        first = df["channel"].unique()[0]
        print(f"[warn] channel {channel_hint} not found, using {first}")
        g = df[df["channel"]==first]
    return g.sort_values("t").reset_index(drop=True)

def plot_compare(normal_df, fault_df, out_png):
    plt.figure(figsize=(12, 4))

    plt.plot(normal_df["t"], normal_df["value"], 
             label="Normal", linewidth=0.7, color='blue')
    plt.plot(fault_df["t"], fault_df["value"], 
             label="Fault", linewidth=0.7, color='red', alpha=0.7)

    plt.title("Normal vs Fault Bearing Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel (g)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✅ saved combined plot -> {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_csv", required=True)
    ap.add_argument("--fault_csv", required=True)
    ap.add_argument("--outdir", default="E:/data/reports/cwru")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    n = load_channel(args.normal_csv, "DE_time")
    f = load_channel(args.fault_csv, "DE_time")

    out_png = os.path.join(args.outdir, "compare_normal_fault.png")
    plot_compare(n, f, out_png)
