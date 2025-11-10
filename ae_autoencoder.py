# -*- coding: utf-8 -*-
import os, argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.signal import welch

def load_channel(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    assert {"t","channel","value"}.issubset(df.columns), "CSV must have t,channel,value"
    g = df[df["channel"]==channel_hint]
    if g.empty:
        first = df["channel"].unique()[0]
        print(f"[warn] channel {channel_hint} not found, using {first}")
        g = df[df["channel"]==first]
    return g.sort_values("t").reset_index(drop=True)

def segment_windows(x, win, stride):
    starts = np.arange(0, len(x)-win+1, stride, dtype=int)
    if len(starts)==0:
        return np.empty((0,win), dtype=np.float32), starts
    X = np.stack([x[s:s+win] for s in starts]).astype(np.float32)
    return X, starts

class Conv1dAE(nn.Module):
    def __init__(self, in_len):
        super().__init__()
        # 채널=1, 길이=in_len
        self.enc = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(8,16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16,32,7, stride=2, padding=3), nn.ReLU(),
        )
        # 길이 축 계산
        with torch.no_grad():
            L = in_len
            for _ in range(3): L = (L + 2*3 - 1*(7-1) -1)//2 + 1  # conv stride=2 패딩=3
            self._latent_len = int(L)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(32,16,7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16,8, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 7, stride=2, padding=3, output_padding=1),
        )
    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        return y

def make_windows(csv_path, fs, win_sec, stride_sec, channel_hint="DE_time"):
    g = load_channel(csv_path, channel_hint)
    x = g["value"].to_numpy().astype(np.float32)
    # DC 제거(권장)
    x = x - x.mean()
    win = int(fs*win_sec); st = int(fs*stride_sec)
    X, starts = segment_windows(x, win, st)
    return X, starts

def standardize_from_normal(Xn):
    mu = Xn.mean(axis=1, keepdims=True)  # per-window 제거?
    # 보다 안정적으로 전체 평균/표준편차 사용
    mu_g = Xn.mean()
    sd_g = Xn.std() + 1e-8
    return (Xn - mu_g)/sd_g, mu_g, sd_g

def apply_standardize(X, mu, sd):
    return (X - mu)/sd

def train_ae(Xn, epochs=10, batch=128, lr=1e-3, device="cpu"):
    N, L = Xn.shape
    model = Conv1dAE(L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xn[:,None,:]))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    for ep in range(1, epochs+1):
        loss_tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            yb = model(xb)
            loss = crit(yb, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_tot += loss.item()*len(xb)
        print(f"[AE] epoch {ep}/{epochs} loss {loss_tot/N:.6f}")
    return model

@torch.no_grad()
def window_mse(model, X, device="cpu"):
    if len(X)==0: return np.array([])
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[:,None,:]))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    crit = nn.MSELoss(reduction="none")
    scores = []
    for (xb,) in dl:
        xb = xb.to(device)
        yb = model(xb)
        # per-window MSE
        mse = crit(yb, xb).mean(dim=(1,2)).detach().cpu().numpy()
        scores.append(mse)
    return np.concatenate(scores, axis=0)

def save_hist(scores_n, scores_f, out_png):
    plt.figure()
    if len(scores_n): plt.hist(scores_n, bins=50, alpha=0.6, label="normal")
    if len(scores_f): plt.hist(scores_f, bins=50, alpha=0.6, label="fault")
    plt.xlabel("AE reconstruction error"); plt.ylabel("count"); plt.legend(); plt.title("AE score distribution")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal_csv", required=True)
    ap.add_argument("--fault_csv",  required=True)
    ap.add_argument("--fs", type=float, default=12000.0)
    ap.add_argument("--win_sec", type=float, default=1.0)
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--outdir", default="E:/data/reports/cwru")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    # 윈도 생성
    Xn_raw, _ = make_windows(args.normal_csv, args.fs, args.win_sec, args.stride_sec)
    Xf_raw, _ = make_windows(args.fault_csv,  args.fs, args.win_sec, args.stride_sec)
    if len(Xn_raw)==0 or len(Xf_raw)==0:
        raise RuntimeError("Not enough samples to window; check fs/win/stride and CSV paths.")

    # 표준화(정상 기준)
    Xn, mu, sd = standardize_from_normal(Xn_raw)
    Xf = apply_standardize(Xf_raw, mu, sd)

    # 모델 학습(정상만)
    model = train_ae(Xn, epochs=args.epochs, batch=args.batch, lr=1e-3, device=device)

    # 점수 계산
    s_n = window_mse(model, Xn, device=device)
    s_f = window_mse(model, Xf, device=device)
    pd.DataFrame({"score": s_n}).to_csv(os.path.join(args.outdir, "ae_loss_normal.csv"), index=False)
    pd.DataFrame({"score": s_f}).to_csv(os.path.join(args.outdir, "ae_loss_fault.csv"), index=False)
    print("saved scores ->", args.outdir)

    # 지표 (정상=0, 결함=1)
    y = np.concatenate([np.zeros_like(s_n), np.ones_like(s_f)])
    s = np.concatenate([s_n, s_f])
    roc = roc_auc_score(y, s)
    ap  = average_precision_score(y, s)
    open(os.path.join(args.outdir, "ae_metrics.txt"), "w", encoding="utf-8").write(f"ROC-AUC={roc:.4f}\nPR-AUC={ap:.4f}\n")
    print(f"ROC-AUC={roc:.4f} PR-AUC={ap:.4f}")

    # 히스토그램
    save_hist(s_n, s_f, os.path.join(args.outdir, "ae_score_hist.png"))

    # 임계값(정상 95%) 저장
    thr = float(np.quantile(s_n, 0.95))
    open(os.path.join(args.outdir, "ae_threshold.txt"), "w").write(str(thr))
    print("threshold ->", thr)

    # 재구성 예시 그림(정상/결함 1개씩)
    with torch.no_grad():
        def recon_fig(x, title, out_png):
            xb = torch.from_numpy(x[None,None,:]).to(device)
            yb = model(xb).cpu().numpy().squeeze()
            plt.figure(figsize=(10,3))
            t = np.arange(len(yb))/args.fs
            plt.plot(t, x.squeeze(), label="orig", linewidth=0.8)
            plt.plot(t, yb.squeeze(), label="recon", linewidth=0.8)
            plt.title(title); plt.xlabel("Time (s)"); plt.legend(); plt.tight_layout()
            plt.savefig(out_png, dpi=150); plt.close()
        recon_fig(Xn_raw[0], "Normal (orig vs recon)", os.path.join(args.outdir,"ae_recon_normal.png"))
        recon_fig(Xf_raw[0], "Fault (orig vs recon)",  os.path.join(args.outdir,"ae_recon_fault.png"))

if __name__ == "__main__":
    main()
