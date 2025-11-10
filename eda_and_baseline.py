# -*- coding: utf-8 -*-
"""
CWRU 변환된 CSV로 EDA + IsolationForest 베이스라인
예:
python eda_and_baseline.py ^
  --normal_csv "E:/data/processed/cwru/Normal_0/csv/DE_time.csv" ^
  --fault_csv  "E:/data/processed/cwru/OR007@6_0/csv/DE_time.csv" ^
  --fs 12000 --win_sec 1.0 --stride_sec 0.25 ^
  --outdir "E:/data/reports/cwru"
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

def read_csv(path):
    df = pd.read_csv(path)
    assert {"t","channel","value"}.issubset(df.columns)
    return df

def take_channel(df, channel="DE_time"):
    g = df[df["channel"]==channel]
    if g.empty:
        first = df["channel"].unique()[0]
        print(f"[warn] channel {channel} not found, using {first}")
        g = df[df["channel"]==first]
    return g.reset_index(drop=True)

def segment_windows(x, win, stride):
    starts = np.arange(0, len(x)-win+1, stride, dtype=int)
    return np.stack([x[s:s+win] for s in starts]), starts

def crest_factor(x): return np.max(np.abs(x)) / (np.sqrt(np.mean(x**2))+1e-9)

def extract_features(Xw, fs):
    feats=[]
    for w in Xw:
        rms = np.sqrt(np.mean(w**2))
        cf  = crest_factor(w)
        pp  = np.ptp(w)
        ku  = kurtosis(w, fisher=False)
        sk  = skew(w)
        f,P = welch(w, fs=fs, nperseg=min(1024,len(w)))
        be1 = np.trapz(P[(f<fs*0.1)],f[(f<fs*0.1)])
        be2 = np.trapz(P[(f>=fs*0.1)&(f<fs*0.3)],f[(f>=fs*0.1)&(f<fs*0.3)])
        be3 = np.trapz(P[(f>=fs*0.3)],f[(f>=fs*0.3)])
        feats.append([rms,cf,pp,ku,sk,be1,be2,be3])
    cols=["rms","crest","p2p","kurt","skew","be1","be2","be3"]
    return np.array(feats),cols

def plot_fft(norm,fault,fs,outdir):
    os.makedirs(outdir,exist_ok=True)
    n=12000
    f1=abs(np.fft.rfft(norm["value"][:n])); f2=abs(np.fft.rfft(fault["value"][:n]))
    fr=np.fft.rfftfreq(n,1/fs)
    plt.figure()
    plt.semilogy(fr,f1,label="normal"); plt.semilogy(fr,f2,label="fault")
    plt.xlabel("Freq(Hz)"); plt.legend(); plt.title("FFT snippet")
    plt.savefig(os.path.join(outdir,"fft_snippet.png"),dpi=150)
    plt.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--normal_csv",required=True)
    p.add_argument("--fault_csv",required=True)
    p.add_argument("--fs",type=float,default=12000)
    p.add_argument("--win_sec",type=float,default=1.0)
    p.add_argument("--stride_sec",type=float,default=0.25)
    p.add_argument("--outdir",default="reports/cwru")
    args=p.parse_args()

    fs=args.fs; os.makedirs(args.outdir,exist_ok=True)
    dfn=take_channel(read_csv(args.normal_csv))
    dff=take_channel(read_csv(args.fault_csv))
    plot_fft(dfn,dff,fs,args.outdir)

    win=int(fs*args.win_sec); st=int(fs*args.stride_sec)
    Xn,_=segment_windows(dfn["value"].values,win,st)
    Xf,_=segment_windows(dff["value"].values,win,st)
    Fn,cols=extract_features(Xn,fs); Ff,_=extract_features(Xf,fs)

    med=np.median(Fn,axis=0); mad=np.median(np.abs(Fn-med),axis=0)+1e-9
    Fnz=(Fn-med)/mad; Ffz=(Ff-med)/mad

    clf=IsolationForest(n_estimators=300,contamination=0.05,random_state=0)
    clf.fit(Fnz)
    s_n=-clf.score_samples(Fnz); s_f=-clf.score_samples(Ffz)
    # 점수 CSV 저장 추가
    pd.DataFrame({"score": s_n}).to_csv(os.path.join(args.outdir, "scores_normal.csv"), index=False)
    pd.DataFrame({"score": s_f}).to_csv(os.path.join(args.outdir, "scores_fault.csv"), index=False)
    # 간단 평가: 정상=0, 결함=1
    y=np.concatenate([np.zeros_like(s_n),np.ones_like(s_f)])
    s=np.concatenate([s_n,s_f])
    print(f"ROC-AUC={roc_auc_score(y,s):.3f}, PR-AUC={average_precision_score(y,s):.3f}")

    plt.figure()
    plt.hist(s_n,bins=50,alpha=0.6,label="normal")
    plt.hist(s_f,bins=50,alpha=0.6,label="fault")
    plt.xlabel("anomaly score"); plt.legend(); plt.title("Score distribution")
    plt.savefig(os.path.join(args.outdir,"score_hist.png"),dpi=150)
    plt.close()
    print(f"Saved plots in {args.outdir}")

if __name__=="__main__":
    main()
    
