"""提交前格式自检：把《虚拟细胞方向材料提交说明》第 6 节的每一条变成断言。

    python scripts/validate_submission.py --prediction prediction.csv

检查项（任一不过即非零退出）：
  1. 4,454 行；sample_ID 唯一，集合与顺序与官方测试 metadata 一致
  2. sample_ID 之后为 4,422 个官方蛋白列，名称与顺序严格匹配 feature contract
  3. 全部预测值有限：无 NA / Inf / 重复列 / 重复 sample_ID / 未声明的额外列
  4. 尺度落在 log2 强度的合理范围（不是 z-score 或 PCA latent）
  5. manifest 里声明了 prediction_scale=log2，且 SHA256 与文件实际一致

Jiao Xinyuan 2026-09-02
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

FAIL = []


def check(cond, ok_msg, bad_msg):
    print(("  ✅ " if cond else "  ❌ ") + (ok_msg if cond else bad_msg))
    if not cond:
        FAIL.append(bad_msg)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default=os.path.join(ROOT, "prediction.csv"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "final.yaml"))
    args = ap.parse_args()

    from vcell import pipeline as pl
    from vcell.harness import protein_keep_mask
    from vcell.io import load_metadata, load_proteome
    cfg = pl.load_config(args.config)

    print(f"检查 {args.prediction}")
    with open(args.prediction, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    df = pd.read_csv(args.prediction, index_col=0)

    print("\n[1] 样本")
    te = load_metadata("test")
    ids_official = te["sample_ID"].astype(str).tolist()
    ids = [str(i) for i in df.index]
    check(len(df) == cfg["data"]["expect_test_rows"],
          f"行数 {len(df)}", f"行数 {len(df)} != {cfg['data']['expect_test_rows']}")
    check(len(set(ids)) == len(ids), "sample_ID 唯一", "sample_ID 有重复")
    check(ids == ids_official, "sample_ID 集合与顺序与官方测试 metadata 一致",
          "sample_ID 顺序或集合与官方测试 metadata 不一致")
    check(header[0] == "sample_ID", "首列名为 sample_ID", f"首列名是 {header[0]}")

    print("\n[2] 蛋白列（feature contract）")
    P = load_proteome("train_val")
    keep = protein_keep_mask(P.meta, P.X, cfg["data"]["protein_missing_max"])
    official = [str(x) for x in P.proteins[keep]]
    cols = [str(c) for c in df.columns]
    check(len(cols) == cfg["data"]["expect_n_proteins"],
          f"蛋白列数 {len(cols)}", f"蛋白列数 {len(cols)} != {cfg['data']['expect_n_proteins']}")
    check(cols == official, "蛋白列名称与顺序严格匹配 train-only 缺失率<0.80 的官方蛋白集",
          "蛋白列名称或顺序与 feature contract 不一致")
    check(len(set(cols)) == len(cols), "无重复列名", "存在重复列名")
    check(header[1:] == official, "CSV 表头逐字段与官方一致（含带逗号的蛋白名）",
          "CSV 表头与官方不一致")

    print("\n[3] 数值有效性")
    V = df.to_numpy(dtype=np.float64)
    check(np.isfinite(V).all(),
          f"全部 {V.size:,} 个值有限，无 NA/Inf",
          f"存在 {int((~np.isfinite(V)).sum())} 个 NA/Inf")

    print("\n[4] 尺度")
    lo, hi, mean = V.min(), V.max(), V.mean()
    check(0.0 < lo and hi < 60.0 and 8.0 < mean < 35.0,
          f"落在 log2 强度合理区间（min {lo:.2f} / mean {mean:.2f} / max {hi:.2f}）",
          f"疑似非 log2 尺度：min {lo:.2f} / mean {mean:.2f} / max {hi:.2f}")

    print("\n[5] manifest")
    mp = os.path.splitext(args.prediction)[0] + "_manifest.json"
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8"))
        check(m.get("prediction_scale") == "log2",
              "manifest 声明 prediction_scale = log2",
              f"manifest 的 prediction_scale = {m.get('prediction_scale')}")
        check(m.get("test_labels_used") is False,
              "manifest 声明未使用测试标签", "manifest 未声明 test_labels_used=False")
        sha = pl.sha256_of(args.prediction)
        check(m.get("sha256") == sha, f"SHA256 与文件一致：{sha}",
              f"SHA256 不一致：manifest {m.get('sha256')} vs 实际 {sha}")
    else:
        check(False, "", f"缺少 {os.path.basename(mp)}")

    print()
    if FAIL:
        print(f"❌ {len(FAIL)} 项未通过：")
        for f in FAIL:
            print("   -", f)
        sys.exit(1)
    print("✅ 全部检查通过，可以提交。")


if __name__ == "__main__":
    main()
