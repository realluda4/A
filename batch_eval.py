# -*- coding: utf-8 -*-
"""
멀티 조건 배치 평가:
- 입력: 각 폴더 안의 CSV(*DE_time*.csv; columns=t,channel,value)
- 조합: 모든 Normal × 모든 Fault
- 방법: (A) 1D-CNN Autoencoder (정상만 학습) (B) Isolation Forest(피처 기반)
- 출력: summary.csv (각 조합의 ROC/PR, 임계값 초과비율 등), per-pair 로그

사용 예:
python batch_eval.py --norm_dirs "E:/data/processed/cwru/Normal_*" --fault_dirs "E:/data/processed/cwru/*fault*" --fs 12000 --win 1.0 --stride 0.25 --epochs 8 --outdir "E:/data/reports/cwru_batch"
"""
import os, glob, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

import torch, torch.nn as nn

# ---------- 공통 유틸 ----------
def find_det_csvs(patterns):
    paths=[]
    for p in patterns:
        # 사용자가 와일드카드 문자열로 넣은 걸 glob 처리
        paths += glob.glob(p)
    # 폴더가 들어오면 폴더 내 *DE_time*.csv 찾기
    csvs=[]
    for path in paths:
        if os.path.isdir(path):
            cand=glob.glob(os.path.join(path, "csv", "*DE_time*.csv"))
            csvs += cand
        elif path.lower().endswith(".csv"):
            if "de_time" in os.path.basename(path).lower():
                csvs.append(path)
    return sorted(set(csvs))

def load_channel(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    need = {"t","channel","value"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns {need}")
    g = df[df["channel"]==channel_hint]
    if g.empty:
        first = df["channel"].unique()[0]
        g = df[df["channel"]==first]
        print(f"[warn] {csv_path}: use channel {first}")
    return g.sort_values("t").reset_index(drop=True)

def segment_windows(x, win, stride):
    starts = np.arange(0, len(x)-win+1, stride, dtype=int)
    if len(starts)==0:
        return np.empty((0,win), dtype=np.float32), starts
    X = np.stack([x[s:s+win] for s in starts]).astype(np.float32)
    return X, starts

# ---------- Isolation Forest(피처) ----------
def crest_factor(x): return float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2))+1e-12))
def feat_windows(Xw, fs):
    feats=[]
    for w in Xw:
        rms = np.sqrt(np.mean(w**2))
        cf  = crest_factor(w)
        p2p = np.ptp(w)
        ku  = kurtosis(w, fisher=False)
        sk  = skew(w)
        f,P = welch(w, fs=fs, nperseg=min(len(w),1024))
        def band(lo,hi):
            m=(f>=lo)&(f<hi)
            return float(np.trapz(P[m], f[m])) if m.any() else 0.0
        be1=band(0,fs*0.1); be2=band(fs*0.1,fs*0.3); be3=band(fs*0.3,fs*0.5)
        feats.append([rms,cf,p2p,ku,sk,be1,be2,be3])
    cols=["rms","crest","p2p","kurt","skew","be1","be2","be3"]
    F=np.array(feats, dtype=float)
    med=np.median(F,axis=0); mad=np.median(np.abs(F-med),axis=0)+1e-9
    return (F-med)/mad, cols

def eval_isoforest(Xn_feat, Xf_feat):
    clf = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    clf.fit(Xn_feat)
    s_n = -clf.score_samples(Xn_feat)
    s_f = -clf.score_samples(Xf_feat)
    y = np.concatenate([np.zeros_like(s_n), np.ones_like(s_f)])
    s = np.concatenate([s_n, s_f])
    return s_n, s_f, roc_auc_score(y,s), average_precision_score(y,s)

# ---------- Autoencoder ----------
class Conv1dAE(nn.Module):
    def __init__(self, in_len):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(8,16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16,32,7, stride=2, padding=3), nn.ReLU(),
        )
        L=in_len
        for _ in range(3): L=(L+2*3-(7-1)-1)//2+1
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32,16,7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,8, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, stride=2, padding=3, output_padding=1),
        )
    def forward(self, x):
        return self.dec(self.enc(x))

@torch.no_grad()
def ae_scores(model, X, device):
    if len(X)==0: return np.array([])
    ds=torch.utils.data.TensorDataset(torch.from_numpy(X[:,None,:]))
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False)
    crit=nn.MSELoss(reduction="none")
    out=[]
    model.eval()
    for (xb,) in dl:
        xb=xb.to(device)
        yb=model(xb)
        mse=crit(yb,xb).mean(dim=(1,2)).cpu().numpy()
        out.append(mse)
    return np.concatenate(out,axis=0)

def train_ae_on_normal(Xn, epochs=8, batch=128, device="cpu"):
    N,L=Xn.shape
    model=Conv1dAE(L).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    crit=nn.MSELoss()
    ds=torch.utils.data.TensorDataset(torch.from_numpy(Xn[:,None,:]))
    dl=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=True)
    model.train()
    for ep in range(1,epochs+1):
        tot=0.0
        for (xb,) in dl:
            xb=xb.to(device); yb=model(xb); loss=crit(yb,xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(xb)
        print(f"[AE] ep{ep}/{epochs} loss={tot/N:.6f}")
    return model

# ---------- 배치 루프 ----------
def run_pair(norm_csv, fault_csv, fs, win_sec, stride_sec, epochs, outdir):
    # 로드 & 윈도
    ndf=load_channel(norm_csv,"DE_time"); fdf=load_channel(fault_csv,"DE_time")
    x_n=ndf["value"].to_numpy().astype(np.float32) - ndf["value"].mean()
    x_f=fdf["value"].to_numpy().astype(np.float32) - fdf["value"].mean()
    win=int(fs*win_sec); st=int(fs*stride_sec)
    Xn,_=segment_windows(x_n,win,st); Xf,_=segment_windows(x_f,win,st)
    if len(Xn)==0 or len(Xf)==0:
        return None

    # IF
    Fn,_ = feat_windows(Xn,fs); Ff,_=feat_windows(Xf,fs)
    sN_if, sF_if, roc_if, pr_if = eval_isoforest(Fn,Ff)

    # AE (정상 표준화 기준)
    mu = Xn.mean(); sd = Xn.std()+1e-8
    Xn_z=(Xn-mu)/sd; Xf_z=(Xf-mu)/sd
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=train_ae_on_normal(Xn_z, epochs=epochs, device=device)
    sN_ae = ae_scores(model, Xn_z, device)
    sF_ae = ae_scores(model, Xf_z, device)
    y = np.concatenate([np.zeros_like(sN_ae), np.ones_like(sF_ae)])
    s = np.concatenate([sN_ae, sF_ae])
    roc_ae = roc_auc_score(y,s); pr_ae = average_precision_score(y,s)

    # 임계값/비율(95%)
    thr_if = float(np.quantile(sN_if,0.95))
    thr_ae = float(np.quantile(sN_ae,0.95))
    ratio_if = float((sF_if>thr_if).mean())
    ratio_ae = float((sF_ae>thr_ae).mean())

    # 결과 반환
    return {
        "normal": os.path.basename(os.path.dirname(os.path.dirname(norm_csv))),  # ex: Normal_0
        "fault":  os.path.basename(os.path.dirname(os.path.dirname(fault_csv))), # ex: fault_data_1
        "n_win": int(len(Xn)), "f_win": int(len(Xf)),
        "IF_ROC": roc_if, "IF_PR": pr_if, "IF_thr95": thr_if, "IF_fault_ratio": ratio_if,
        "AE_ROC": roc_ae, "AE_PR": pr_ae, "AE_thr95": thr_ae, "AE_fault_ratio": ratio_ae,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--norm_dirs", nargs="+", required=True, help="예: E:/data/processed/cwru/Normal_*")
    ap.add_argument("--fault_dirs", nargs="+", required=True, help="예: E:/data/processed/cwru/*fault*  또는  결함 폴더들")
    ap.add_argument("--fs", type=float, default=12000.0)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--outdir", default="E:/data/reports/cwru_batch")
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    norm_csvs = find_det_csvs(args.norm_dirs)
    fault_csvs = find_det_csvs(args.fault_dirs)
    if not norm_csvs: raise SystemExit("No normal CSVs found")
    if not fault_csvs: raise SystemExit("No fault CSVs found")

    rows=[]
    for n_csv in norm_csvs:
        for f_csv in fault_csvs:
            print(f"\n=== Pair ===\nN: {n_csv}\nF: {f_csv}")
            try:
                res = run_pair(n_csv, f_csv, args.fs, args.win, args.stride, args.epochs, args.outdir)
                if res: rows.append(res)
                else:   print("  [skip] too few samples for windows")
            except Exception as e:
                print("  [error]", e)

    if rows:
        df=pd.DataFrame(rows)
        out_csv=os.path.join(args.outdir,"summary.csv")
        df.to_csv(out_csv, index=False)
        print("\n✅ saved summary ->", out_csv)
        # 간단 정렬된 미니 리포트
        try:
            top=df.sort_values("AE_ROC", ascending=False).head(10)
            print("\nTop (by AE_ROC):")
            print(top[["normal","fault","AE_ROC","AE_PR","AE_fault_ratio"]].to_string(index=False))
        except Exception:
            pass
    else:
        print("No results.")

if __name__=="__main__":
    main()
