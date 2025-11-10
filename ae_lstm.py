# -*- coding: utf-8 -*-
"""
LSTM Autoencoder 기반 베어링 이상 탐지
- 정상 CSV로만 학습
- 정상/결함 윈도별 재구성 오차 계산
- ROC-AUC, PR-AUC, 히스토그램, 임계값까지 저장

입력 CSV 형식:
  t, channel, value
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------- 공통 유틸 ----------
def load_channel(csv_path, channel_hint="DE_time"):
    df = pd.read_csv(csv_path)
    need = {"t", "channel", "value"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns {need}")
    g = df[df["channel"] == channel_hint]
    if g.empty:
        first = df["channel"].unique()[0]
        print(f"[warn] {csv_path}: channel {channel_hint} not found, use {first}")
        g = df[df["channel"] == first]
    return g.sort_values("t").reset_index(drop=True)

def segment_windows(x, win, stride):
    starts = np.arange(0, len(x) - win + 1, stride, dtype=int)
    if len(starts) == 0:
        return np.empty((0, win), dtype=np.float32), starts
    X = np.stack([x[s:s+win] for s in starts]).astype(np.float32)
    return X, starts


# ---------- LSTM Autoencoder ----------
class LSTMAE(nn.Module):
    """
    간단한 LSTM Autoencoder
    - Encoder: LSTM → 마지막 hidden
    - Decoder: 그 hidden을 초기 상태로 두고, 0 입력 시퀀스를 넣어 원 신호를 복원
    입력/출력 shape: (batch, seq_len, 1)
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 인코더: 입력(1) -> hidden_dim
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 디코더: 입력(1) -> hidden_dim
        # (0 벡터를 넣고 hidden에서 정보 꺼내 씀)
        self.decoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # hidden_dim -> 1 (최종 복원 출력)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, 1)
        B, T, _ = x.size()

        # Encoder
        _, (h_n, c_n) = self.encoder(x)
        # h_n, c_n: (num_layers, B, hidden_dim)

        # Decoder 초기 상태 = encoder의 마지막 상태
        h0 = h_n
        c0 = c_n

        # Decoder 입력: 전부 0인 시퀀스
        dec_in = torch.zeros(B, T, self.input_dim, device=x.device)
        dec_out, _ = self.decoder(dec_in, (h0, c0))  # (B, T, hidden_dim)

        # 각 타임스텝 hidden → 1차원 출력
        y = self.out(dec_out)  # (B, T, 1)
        return y



def make_windows(csv_path, fs, win_sec, stride_sec, channel_hint="DE_time"):
    g = load_channel(csv_path, channel_hint)
    x = g["value"].to_numpy().astype(np.float32)
    x = x - x.mean()
    win = int(fs * win_sec)
    st = int(fs * stride_sec)
    X, starts = segment_windows(x, win, st)
    return X, starts


def standardize_from_normal(Xn):
    mu = float(Xn.mean())
    sd = float(Xn.std() + 1e-8)
    return (Xn - mu) / sd, mu, sd

def apply_standardize(X, mu, sd):
    return (X - mu) / sd


def train_lstm_ae(Xn, epochs=15, batch_size=128, lr=1e-3, device="cpu"):
    N, T = Xn.shape
    model = LSTMAE(input_dim=1, hidden_dim=64, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xn[:, :, None]))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            yb = model(xb)
            loss = crit(yb, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"[LSTM-AE] epoch {ep}/{epochs} loss={total / N:.6f}")
    return model



@torch.no_grad()
def window_mse(model, X, device="cpu"):
    if len(X) == 0:
        return np.array([])
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[:, :, None]))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    crit = nn.MSELoss(reduction="none")
    scores = []
    model.eval()
    for (xb,) in dl:
        xb = xb.to(device)
        yb = model(xb)
        mse = crit(yb, xb).mean(dim=(1, 2)).cpu().numpy()
        scores.append(mse)
    return np.concatenate(scores, axis=0)


def save_hist(scores_n, scores_f, out_png):
    plt.figure()
    if len(scores_n):
        plt.hist(scores_n, bins=50, alpha=0.6, label="normal")
    if len(scores_f):
        plt.hist(scores_f, bins=50, alpha=0.6, label="fault")
    plt.xlabel("LSTM-AE reconstruction error")
    plt.ylabel("count")
    plt.legend()
    plt.title("LSTM-AE score distribution")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal_csv", required=True)
    p.add_argument("--fault_csv", required=True)
    p.add_argument("--fs", type=float, default=12000.0)
    p.add_argument("--win_sec", type=float, default=1.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--outdir", default="E:/data/reports/cwru_lstm")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    # 윈도 생성
    Xn_raw, _ = make_windows(args.normal_csv, args.fs, args.win_sec, args.stride_sec)
    Xf_raw, _ = make_windows(args.fault_csv,  args.fs, args.win_sec, args.stride_sec)

    if len(Xn_raw) == 0 or len(Xf_raw) == 0:
        raise RuntimeError("Not enough data for windows. Check fs/win/stride/csv paths.")

    # 표준화 (정상 기준)
    Xn, mu, sd = standardize_from_normal(Xn_raw)
    Xf = apply_standardize(Xf_raw, mu, sd)

    # 학습 (정상만)
    model = train_lstm_ae(Xn, epochs=args.epochs, batch_size=args.batch, lr=1e-3, device=device)

    # 점수 계산
    s_n = window_mse(model, Xn, device=device)
    s_f = window_mse(model, Xf, device=device)

    # 저장
    pd.DataFrame({"score": s_n}).to_csv(os.path.join(args.outdir, "lstm_loss_normal.csv"), index=False)
    pd.DataFrame({"score": s_f}).to_csv(os.path.join(args.outdir, "lstm_loss_fault.csv"), index=False)
    print("[save] scores ->", args.outdir)

    # 지표 (정상=0, 결함=1)
    y = np.concatenate([np.zeros_like(s_n), np.ones_like(s_f)])
    s = np.concatenate([s_n, s_f])
    roc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    with open(os.path.join(args.outdir, "lstm_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"ROC-AUC={roc:.4f}\nPR-AUC={ap:.4f}\n")
    print(f"[metric] ROC-AUC={roc:.4f}, PR-AUC={ap:.4f}")

    # 히스토그램
    save_hist(s_n, s_f, os.path.join(args.outdir, "lstm_score_hist.png"))

    # 임계값 (정상 95%)
    thr = float(np.quantile(s_n, 0.95))
    with open(os.path.join(args.outdir, "lstm_threshold.txt"), "w") as f:
        f.write(str(thr))
    print(f"[threshold] 95% -> {thr}")

    # 재구성 예시 (정상/결함 한 윈도씩)
    with torch.no_grad():
        def recon_plot(x_raw, title, path_png):
            x_in = ((x_raw - mu) / sd)[None, :, None]
            xb = torch.from_numpy(x_in).to(device)
            yb = model(xb).cpu().numpy().squeeze()
            t = np.arange(len(x_raw)) / args.fs

            plt.figure(figsize=(10,3))
            plt.plot(t, x_raw, label="orig", linewidth=0.8)
            plt.plot(t, (yb * sd + mu), label="recon", linewidth=0.8)
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path_png, dpi=150)
            plt.close()

        recon_plot(Xn_raw[0], "Normal (orig vs recon)", os.path.join(args.outdir, "lstm_recon_normal.png"))
        recon_plot(Xf_raw[0], "Fault (orig vs recon)",  os.path.join(args.outdir, "lstm_recon_fault.png"))

if __name__ == "__main__":
    main()
