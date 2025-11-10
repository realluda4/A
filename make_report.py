# -*- coding: utf-8 -*-
"""
IsolationForest / Conv-AE / LSTM-AE 성능 비교 리포트 생성

입력:
  - 각 모델의 정상/결함 score CSV
    (컬럼 이름은 모두 'score'라고 가정)

기능:
  - ROC-AUC, PR-AUC 계산
  - 모델별 성능 비교 CSV 저장
  - 점수 분포 히스토그램
  - 성능 막대그래프
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


def load_scores(path_normal, path_fault):
    sn = pd.read_csv(path_normal)["score"].to_numpy()
    sf = pd.read_csv(path_fault)["score"].to_numpy()
    return sn, sf


def eval_model(sn, sf):
    y = np.concatenate([np.zeros_like(sn), np.ones_like(sf)])
    s = np.concatenate([sn, sf])
    roc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    return roc, ap


def hist_compare(models, out_png):
    plt.figure(figsize=(10, 6))
    for name, sn, sf in models:
        # 정상은 실선, 결함은 점선 느낌으로 구분
        plt.hist(sn, bins=40, alpha=0.35, label=f"{name} normal")
        plt.hist(sf, bins=40, alpha=0.35, label=f"{name} fault")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.title("Score distribution comparison")
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 10)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[save] {out_png}")


def bar_compare(rows, out_png):
    names = [r["model"] for r in rows]
    rocs = [r["ROC_AUC"] for r in rows]
    prs  = [r["PR_AUC"] for r in rows]

    x = np.arange(len(names))
    w = 0.35

    plt.figure(figsize=(7,5))
    # 색상 팔레트 (톤 다운된 파랑/주황)
    colors = {"ROC-AUC": "#4C72B0", "PR-AUC": "#DD8452"}

    # 막대 그래프
    plt.bar(x - w/2, rocs, width=w, label="ROC-AUC", color=colors["ROC-AUC"], alpha=0.9)
    plt.bar(x + w/2, prs,  width=w, label="PR-AUC",  color=colors["PR-AUC"], alpha=0.9)

    # 라벨 스타일링
    plt.xticks(x, names, fontsize=11)
    plt.yticks(fontsize=10)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance (ROC-AUC / PR-AUC)", fontsize=13, weight="bold")

    # 격자 추가 (눈에 거슬리지 않게)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.ylim(0, 1.05)

    # 값 라벨 표시
    for i, (r, p) in enumerate(zip(rocs, prs)):
        plt.text(i - w/2, r + 0.02, f"{r:.3f}", ha="center", va="bottom", fontsize=9)
        plt.text(i + w/2, p + 0.02, f"{p:.3f}", ha="center", va="bottom", fontsize=9)

    plt.legend(frameon=False, fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_png}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--if_normal",   required=True)
    ap.add_argument("--if_fault",    required=True)
    ap.add_argument("--ae_normal",   required=True)
    ap.add_argument("--ae_fault",    required=True)
    ap.add_argument("--lstm_normal", required=True)
    ap.add_argument("--lstm_fault",  required=True)
    ap.add_argument("--outdir", default="E:/data/reports/cwru_report")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    models_for_hist = []

    # Isolation Forest
    sn_if, sf_if = load_scores(args.if_normal, args.if_fault)
    roc_if, pr_if = eval_model(sn_if, sf_if)
    rows.append({"model": "IForest", "ROC_AUC": roc_if, "PR_AUC": pr_if})
    models_for_hist.append(("IForest", sn_if, sf_if))

    # Conv-AE
    sn_ae, sf_ae = load_scores(args.ae_normal, args.ae_fault)
    roc_ae, pr_ae = eval_model(sn_ae, sf_ae)
    rows.append({"model": "Conv-AE", "ROC_AUC": roc_ae, "PR_AUC": pr_ae})
    models_for_hist.append(("Conv-AE", sn_ae, sf_ae))

    # LSTM-AE
    sn_lstm, sf_lstm = load_scores(args.lstm_normal, args.lstm_fault)
    roc_lstm, pr_lstm = eval_model(sn_lstm, sf_lstm)
    rows.append({"model": "LSTM-AE", "ROC_AUC": roc_lstm, "PR_AUC": pr_lstm})
    models_for_hist.append(("LSTM-AE", sn_lstm, sf_lstm))

    # CSV 저장
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "model_comparison_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    # 히스토그램 & 바차트
    hist_compare(models_for_hist, os.path.join(args.outdir, "model_scores_hist_all.png"))
    bar_compare(rows, os.path.join(args.outdir, "model_metrics_bar.png"))


if __name__ == "__main__":
    main()
