# -*- coding: utf-8 -*-
"""
CWRU/IMS 등 MATLAB(.mat) 진동 데이터를 파이썬에서 쓰기 좋게 변환
- v7 이하는 scipy.io.loadmat
- v7.3(HDF5)는 h5py 또는 mat73
- CWRU: DE_time/FE_time/BA_time 키 자동 탐지
사용 예:
  python matlab_data_loader.py --mat "E:\\data\\bearing data\\normal bearing\\Normal_0.mat" --dataset cwru --out "E:\\data\\processed\\cwru"
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

def _is_hdf5(path: str) -> bool:
    try:
        import h5py
        return h5py.is_hdf5(path)
    except Exception:
        return False

def _load_v7(path: str):
    from scipy.io import loadmat
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}

def _load_v73(path: str):
    try:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = obj[()]
                    if isinstance(data, (bytes, np.bytes_)):
                        data = data.decode("utf-8", errors="ignore")
                    out[name.split("/")[-1]] = np.array(data)
            f.visititems(visit)
        if out:
            return out
    except Exception:
        pass
    try:
        import mat73
        return mat73.loadmat(path)
    except Exception as e:
        raise RuntimeError("v7.3 파일은 h5py 또는 mat73가 필요합니다.") from e

def load_mat_any(path: str):
    return _load_v73(path) if _is_hdf5(path) else _load_v7(path)

def _normalize_keys(d):
    return {k.lower(): k for k in d.keys()}

def parse_cwru(mat: dict, default_fs: float | None = 12000.0):
    nk = _normalize_keys(mat)
    def get(name):
        return mat.get(nk.get(name.lower(), ""), None)
    fs = get("fs") or get("rate") or get("sampling_rate")
    if isinstance(fs, (list, np.ndarray)) and np.size(fs) == 1:
        fs = float(np.squeeze(fs))
    elif isinstance(fs, (int, float)):
        fs = float(fs)
    else:
        fs = default_fs  # 없으면 12k 가정

    series = []
    for ch in ["DE_time", "FE_time", "BA_time"]:
        v = get(ch)
        if v is None:
            continue
        arr = np.squeeze(np.array(v, dtype=float))
        t = np.arange(arr.shape[0]) / fs if fs else np.arange(arr.shape[0])
        series.append(pd.DataFrame({"t": t, "channel": ch, "value": arr}))
    if not series:
        # 채널명이 달라도 1D 수치 벡터를 모두 채널로 수집
        for k, v in mat.items():
            a = np.squeeze(np.array(v))
            if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
                t = np.arange(a.shape[0]) / fs if fs else np.arange(a.shape[0])
                series.append(pd.DataFrame({"t": t, "channel": k, "value": a}))
    df = pd.concat(series, ignore_index=True) if series else pd.DataFrame(columns=["t","channel","value"])
    meta = {"fs": fs, "keys": list(mat.keys())}
    return df, meta

def to_parquet_by_channel(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ch, grp in df.groupby("channel"):
        grp.sort_values("t").to_parquet(os.path.join(out_dir, f"{ch}.parquet"), index=False)

def to_csv_by_channel(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ch, grp in df.groupby("channel"):
        grp.sort_values("t").to_csv(os.path.join(out_dir, f"{ch}.csv"), index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", required=True, help="입력 .mat 파일 경로")
    p.add_argument("--dataset", choices=["cwru","auto"], default="cwru")
    p.add_argument("--out", default="data/processed/cwru")
    p.add_argument("--fmt", choices=["csv","parquet","both"], default="both")
    p.add_argument("--default_fs", type=float, default=12000.0, help="fs 미존재시 기본 샘플레이트(Hz)")
    args = p.parse_args()

    mat = load_mat_any(args.mat)
    df, meta = parse_cwru(mat, default_fs=args.default_fs)
    print("meta:", meta)

    if args.fmt in ("parquet","both"):
        to_parquet_by_channel(df, os.path.join(args.out, "parquet"))
    if args.fmt in ("csv","both"):
        to_csv_by_channel(df, os.path.join(args.out, "csv"))
    print(f"Saved to {args.out}/(csv|parquet)")

if __name__ == "__main__":
    main()
