# -*- coding: utf-8 -*-
"""
Normal vs Fault 주파수 스펙트럼(FFT/PSD) 비교
- CSV 형식: t, channel, value
- 기본 채널: DE_time
"""

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def load_channel(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    need = {"t","channel","value"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns {need}")
    g = df[df["channel"]==channel_hint]
    if g.empty:
        first = df["channel"].unique()[0]
        print(f"[warn] channel {channel_hint} not found, using {first}")
        g = df[df["channel"]==first]
    g = g.sort_values("t").reset_index(drop=True)
    return g

def rfft_mag(x, fs):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0/fs)
    m = np.abs(X) / len(x)  # amplitude spectrum (normalized)
    return f, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_csv", required=True)
    ap.add_argument("--fault_csv",  required=True)
    ap.add_argument("--fs", type=float, default=12000.0, help="sampling rate (Hz)")
    ap.add_argument("--duration_sec", type=float, default=10.0, help="앞쪽 duration만 사용(초)")
    ap.add_argument("--fmax", type=float, default=5000.0, help="표시할 최대 주파수(Hz)")
    ap.add_argument("--method", choices=["rfft","welch","both"], default="both")
    ap.add_argument("--nperseg", type=int, default=4096, help="Welch nperseg")
    ap.add_argument("--outdir", default="E:/data/reports/cwru")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    n = load_channel(args.normal_csv, "DE_time")
    f = load_channel(args.fault_csv,  "DE_time")

    fs = args.fs
    # duration 맞춰 자르기
    n_cut = int(min(len(n), args.duration_sec*fs))
    f_cut = int(min(len(f), args.duration_sec*fs))
    x_n = n["value"].to_numpy()[:n_cut] - n["value"].to_numpy()[:n_cut].mean()
    x_f = f["value"].to_numpy()[:f_cut] - f["value"].to_numpy()[:f_cut].mean()

    if args.method in ("rfft","both"):
        fr_n, mag_n = rfft_mag(x_n, fs)
        fr_f, mag_f = rfft_mag(x_f, fs)

        fig = plt.figure(figsize=(12,5))
        plt.semilogy(fr_n, mag_n, label="Normal")
        plt.semilogy(fr_f, mag_f, label="Fault", alpha=0.9)
        plt.xlim(0, min(args.fmax, fs/2))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Amplitude Spectrum (RFFT)")
        plt.legend()
        plt.tight_layout()
        out1 = os.path.join(args.outdir, "compare_fft_rfft.png")
        fig.savefig(out1, dpi=150); plt.close(fig)
        print(f"✅ saved -> {out1}")

    if args.method in ("welch","both"):
        fw_n, Pxx_n = welch(x_n, fs=fs, nperseg=min(args.nperseg, len(x_n)))
        fw_f, Pxx_f = welch(x_f, fs=fs, nperseg=min(args.nperseg, len(x_f)))

        fig = plt.figure(figsize=(12,5))
        plt.semilogy(fw_n, Pxx_n, label="Normal")
        plt.semilogy(fw_f, Pxx_f, label="Fault", alpha=0.9)
        plt.xlim(0, min(args.fmax, fs/2))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (power/Hz)")
        plt.title("Power Spectral Density (Welch)")
        plt.legend()
        plt.tight_layout()
        out2 = os.path.join(args.outdir, "compare_fft_welch.png")
        fig.savefig(out2, dpi=150); plt.close(fig)
        print(f"✅ saved -> {out2}")

if __name__ == "__main__":
    main()
