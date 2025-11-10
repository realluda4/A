# -*- coding: utf-8 -*-
import argparse, os, pandas as pd, numpy as np

def group_alarms(flags, min_gap=8):
    idx = np.where(flags==1)[0]
    if len(idx)==0: return []
    runs=[]; st=idx[0]; prev=idx[0]
    for i in idx[1:]:
        if i==prev+1: prev=i
        else: runs.append([st,prev]); st=i; prev=i
    runs.append([st,prev])
    merged=[]
    for s0,e0 in runs:
        if not merged or s0-merged[-1][1] > min_gap: merged.append([s0,e0])
        else: merged[-1][1]=e0
    return merged

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    s = pd.read_csv(args.scores_csv)["score"].to_numpy()
    flags = (s > args.thr).astype(int)
    spans = group_alarms(flags, min_gap=int(2/args.stride_sec))  # 2            

    rows=[]
    for s0,e0 in spans:
        rows.append({
            "start_win": int(s0),
            "end_win": int(e0),
            "start_time": s0*args.stride_sec,
            "end_time": e0*args.stride_sec + args.win_sec
        })
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"saved -> {args.out_csv}")
