# -*- coding: utf-8 -*-
"""
batch_eval.py가 생성한 summary.csv를 분석해서:
- IsolationForest vs Conv-AE 성능 비교
- 결함 유형(Ball/Inner/Outer)별, 결함 크기(007/014/021)별 성능 요약
- 간단한 시각화(막대/박스/산점도) 생성

입력: E:/data/reports/cwru_batch/summary.csv
출력:
  - E:/data/reports/cwru_batch/analysis_overview.txt
  - E:/data/reports/cwru_batch/model_auc_bar.png
  - E:/data/reports/cwru_batch/model_auc_box.png
  - E:/data/reports/cwru_batch/ae_vs_if_scatter.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"E:/data/reports/cwru_batch"
CSV_PATH = os.path.join(BASE, "summary.csv")
OUT_TXT = os.path.join(BASE, "analysis_overview.txt")

os.makedirs(BASE, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---- 유틸: fault 이름에서 유형/크기 추출 ----
def parse_fault_type(name: str) -> str:
    n = name.lower()
    if "ball" in n or re.search(r"\bb0?0?7\b", n):
        return "Ball"
    if "inner" in n or "ir" in n:
        return "Inner"
    if "outer" in n or "or" in n:
        return "Outer"
    return "Other"

def parse_severity(name: str) -> str:
    n = name
    # 007,014,021 패턴 찾기
    m = re.search(r"0(07|14|21)", n)
    if m:
        return m.group(1)  # "07","14","21"
    # 없으면 fallback
    return "UNK"

df["fault_type"] = df["fault"].astype(str).apply(parse_fault_type)
df["severity"] = df["fault"].astype(str).apply(parse_severity)

# ---- 전체 모델별 성능 요약 ----
rows = []

def add_row(model, aucs, prs):
    if len(aucs) == 0:
        return
    rows.append({
        "model": model,
        "n_pairs": len(aucs),
        "ROC_AUC_mean": float(np.mean(aucs)),
        "ROC_AUC_min": float(np.min(aucs)),
        "ROC_AUC_max": float(np.max(aucs)),
        "PR_AUC_mean": float(np.mean(prs)),
        "PR_AUC_min": float(np.min(prs)),
        "PR_AUC_max": float(np.max(prs)),
    })

add_row("IForest", df["IF_ROC"], df["IF_PR"])
add_row("Conv-AE", df["AE_ROC"], df["AE_PR"])

summary_model = pd.DataFrame(rows)

# ---- 유형/크기별 AE / IF 성능 ----
grp_type = df.groupby("fault_type")[["IF_ROC","AE_ROC","IF_PR","AE_PR"]].mean().reset_index()
grp_sev  = df.groupby("severity")[["IF_ROC","AE_ROC","IF_PR","AE_PR"]].mean().reset_index()

# ---- 텍스트 리포트 저장 ----
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== Model-level summary ===\n")
    f.write(summary_model.to_string(index=False))
    f.write("\n\n=== By fault type (mean AUC) ===\n")
    f.write(grp_type.to_string(index=False))
    f.write("\n\n=== By severity code (mean AUC) ===\n")
    f.write(grp_sev.to_string(index=False))
    f.write("\n")

print("[save]", OUT_TXT)

# ---- 시각화 1: 모델별 AUC 평균 막대그래프 ----
plt.figure(figsize=(6,4))
x = np.arange(len(summary_model))
w = 0.35

plt.bar(x - w/2, summary_model["ROC_AUC_mean"], width=w, label="ROC-AUC")
plt.bar(x + w/2, summary_model["PR_AUC_mean"],  width=w, label="PR-AUC")

plt.xticks(x, summary_model["model"])
plt.ylim(0, 1.05)
plt.ylabel("AUC")
plt.title("Model Performance (mean over all pairs)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
for i, r in summary_model.iterrows():
    plt.text(i - w/2, r["ROC_AUC_mean"]+0.01, f"{r['ROC_AUC_mean']:.3f}", ha="center", va="bottom", fontsize=8)
    plt.text(i + w/2, r["PR_AUC_mean"]+0.01, f"{r['PR_AUC_mean']:.3f}", ha="center", va="bottom", fontsize=8)
plt.legend(frameon=False)
plt.tight_layout()
out_bar = os.path.join(BASE, "model_auc_bar.png")
plt.savefig(out_bar, dpi=180)
plt.close()
print("[save]", out_bar)

# ---- 시각화 2: AUC 분포 박스플롯 ----
plt.figure(figsize=(6,4))
data = [
    df["IF_ROC"].values,
    df["AE_ROC"].values,
]
plt.boxplot(data, labels=["IForest ROC", "Conv-AE ROC"])
plt.ylim(0, 1.05)
plt.ylabel("ROC-AUC")
plt.title("AUC distribution by model")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
out_box = os.path.join(BASE, "model_auc_box.png")
plt.savefig(out_box, dpi=180)
plt.close()
print("[save]", out_box)

# ---- 시각화 3: AE vs IF 산점도 ----
plt.figure(figsize=(5,5))
plt.scatter(df["IF_ROC"], df["AE_ROC"], alpha=0.7)
plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
plt.xlabel("IForest ROC-AUC")
plt.ylabel("Conv-AE ROC-AUC")
plt.title("AE vs IF per (Normal, Fault) pair")
plt.grid(True, linestyle="--", alpha=0.4)
plt.xlim(0,1.02)
plt.ylim(0,1.02)
out_scatter = os.path.join(BASE, "ae_vs_if_scatter.png")
plt.tight_layout()
plt.savefig(out_scatter, dpi=180)
plt.close()
print("[save]", out_scatter)
