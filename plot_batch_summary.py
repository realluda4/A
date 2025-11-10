# -*- coding: utf-8 -*-
"""
batch_eval.py 결과(summary.csv)를 시각화:
- Normal_x vs fault_data_y 조합별 ROC-AUC 히트맵
- IsolationForest / Conv-AE 두 모델 비교
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_heatmap(df, value_col, title, out_path, vmin=0.9, vmax=1.0):
    pivot = df.pivot(index="normal", columns="fault", values=value_col)
    plt.figure(figsize=(6,4))
    im = plt.imshow(pivot.values, vmin=vmin, vmax=vmax, aspect="auto")
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(pivot.index)),   labels=pivot.index)
    plt.colorbar(im, label=value_col)
    plt.title(title)
    # 값 숫자 찍기
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i,j]
            if np.isnan(val): continue
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[save] {out_path}")

if __name__ == "__main__":
    base = r"E:/data/reports/cwru_batch"
    csv_path = os.path.join(base, "summary.csv")
    df = pd.read_csv(csv_path)

    # IF, AE ROC-AUC 히트맵
    plot_heatmap(df, "IF_ROC",
                 "IsolationForest ROC-AUC by pair",
                 os.path.join(base, "if_roc_heatmap.png"),
                 vmin=0.9, vmax=1.0)

    plot_heatmap(df, "AE_ROC",
                 "Conv-AE ROC-AUC by pair",
                 os.path.join(base, "ae_roc_heatmap.png"),
                 vmin=0.9, vmax=1.0)

    # 필요하면 PR-AUC도 추가
    plot_heatmap(df, "IF_PR",
                 "IsolationForest PR-AUC by pair",
                 os.path.join(base, "if_pr_heatmap.png"),
                 vmin=0.9, vmax=1.0)

    plot_heatmap(df, "AE_PR",
                 "Conv-AE PR-AUC by pair",
                 os.path.join(base, "ae_pr_heatmap.png"),
                 vmin=0.9, vmax=1.0)
