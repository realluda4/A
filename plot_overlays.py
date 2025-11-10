# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_scores(scores_csv):
    s = pd.read_csv(scores_csv)["score"].to_numpy()
    return s

def load_signal(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    # 기대 컬럼: t, channel, value
    if not {"t","channel","value"}.issubset(df.columns):
        raise ValueError("CSV must have columns: t, channel, value")
    # 채널 선택
    g = df[df["channel"]==channel_hint]
    if g.empty:
        # 힌트가 없으면 첫 채널
        first = df["channel"].unique()[0]
        g = df[df["channel"]==first]
        print(f"[warn] channel {channel_hint} not found. use {first}")
    g = g.sort_values("t").reset_index(drop=True)
    return g

def spans_from_scores(scores, thr, win_sec=1.0, stride_sec=0.25, min_gap_sec=0.0):
    flags = (scores > thr).astype(int)
    idx = np.where(flags==1)[0]
    if len(idx)==0: return []
    runs=[]
    st=idx[0]; prev=idx[0]
    for i in idx[1:]:
        if i==prev+1:
            prev=i
        else:
            runs.append([st,prev])
            st=i; prev=i
    runs.append([st,prev])
    # 병합
    min_gap = int(round(min_gap_sec/stride_sec))
    merged=[]
    for s0,e0 in runs:
        if not merged or s0-merged[-1][1] > min_gap:
            merged.append([s0,e0])
        else:
            merged[-1][1]=e0
    # 시간 변환
    spans=[]
    for s0,e0 in merged:
        spans.append({
            "start_win": int(s0),
            "end_win": int(e0),
            "start_time": s0*stride_sec,
            "end_time": e0*stride_sec + win_sec
        })
    return spans

def plot_overlay(signal_df, spans, out_png):
    t = signal_df["t"].to_numpy()
    x = signal_df["value"].to_numpy()

    plt.figure(figsize=(12,4))
    plt.plot(t, x, linewidth=0.8, label="signal")
    for sp in spans:
        st, ed = sp["start_time"], sp["end_time"]
        plt.axvspan(st, ed, alpha=0.25, label="alarm" if "alarm_shown" not in locals() else None)
        alarm_shown = True  # show label once
    plt.xlabel("Time (s)")
    plt.ylabel("Accel (g)")
    plt.title("Signal with alarm spans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"✅ saved overlay -> {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault_signal_csv", required=True, help="결함 신호 CSV (DE_time)")
    ap.add_argument("--scores_fault_csv", required=True, help="결함 점수 CSV")
    ap.add_argument("--thr", type=float, required=True, help="임계값")
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--min_gap_sec", type=float, default=0.0)
    ap.add_argument("--outdir", default="E:/data/reports/cwru")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sig = load_signal(args.fault_signal_csv, "DE_time")
    scores = load_scores(args.scores_fault_csv)

    spans = spans_from_scores(scores, thr=args.thr, win_sec=args.win_sec,
                              stride_sec=args.stride_sec, min_gap_sec=args.min_gap_sec)

    # 저장: spans CSV
    spans_df = pd.DataFrame(spans)
    spans_csv = os.path.join(args.outdir, "alarms_fault_overlay.csv")
    spans_df.to_csv(spans_csv, index=False)
    print(f"✅ saved spans -> {spans_csv}")

    # 플롯
    plot_overlay(sig, spans, os.path.join(args.outdir, "signal_alarm_overlay.png"))
