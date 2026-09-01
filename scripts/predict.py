"""官方要求的第 3 条主命令：冻结模型推理并生成 prediction.csv。

    python scripts/predict.py --metadata <test_metadata> --run-dir runs/final \
        --output prediction.csv

只读取：测试 metadata、scripts/train.py 产出的冻结产物。**不读取**任何蛋白真值——
测试蛋白质组文件在 vcell/io.py 层被硬性拒绝，训练标签也不在本脚本的任何路径上。

输出 4,454 行 ×（sample_ID + 4,422 个官方蛋白列），log2 尺度，行序与官方测试 metadata
一致，并在写盘后立即执行 scripts/validate_submission.py 的全部检查。

Jiao Xinyuan 2026-09-02
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from vcell import pipeline as pl                                  # noqa: E402



def _code_version():
    """git describe（若在 git 仓库内），否则回退到 REPRODUCIBILITY_MANIFEST 里的 commit。"""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    try:
        import json as _json
        m = _json.load(open(os.path.join(ROOT, "REPRODUCIBILITY_MANIFEST.json"), encoding="utf-8"))
        return (m.get("code_version") or {}).get("commit", "unknown")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="测试 metadata（核对 sample_ID 用）")
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "runs", "final"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "final.yaml"))
    ap.add_argument("--output", default=os.path.join(ROOT, "prediction.csv"))
    ap.add_argument("--no-validate", action="store_true", help="跳过写盘后的格式自检")
    args = ap.parse_args()

    cfg = pl.load_config(args.config)
    run = json.load(open(os.path.join(args.run_dir, "run.json"), encoding="utf-8"))
    names = [m["name"] for m in run["members"]]
    want = [m["name"] for m in cfg["ensemble"]["members"]]
    missing = [n for n in want if n not in names]
    assert not missing, f"run 目录缺少成员：{missing}"
    print(f"集成成员 {len(want)} 个：{want}", flush=True)

    t0 = time.time()
    ids = run["test_sample_ids"]
    proteins = run["proteins"]
    if args.metadata:
        te = pd.read_csv(args.metadata)
        assert te["sample_ID"].astype(str).tolist() == ids, \
            "测试 metadata 的 sample_ID 顺序与训练时记录的不一致"
        print("已核对 --metadata 的 sample_ID 顺序一致", flush=True)

    is_test = np.zeros(run["n_rows"], bool)
    is_test[-len(ids):] = True        # load_combined 把 test 行接在 train_val 之后
    preds = []
    for n in want:
        art = pl.load_member(args.run_dir, n)
        preds.append(pl.predict_member(art, is_test))
        print(f"  {n}: 均值 {preds[-1].mean():.4f}", flush=True)
    w = cfg["ensemble"].get("weights")
    Y = pl.compose(preds, w)

    df = pd.DataFrame(Y, index=pd.Index(ids, name="sample_ID"), columns=proteins)
    assert df.shape == (cfg["data"]["expect_test_rows"], cfg["data"]["expect_n_proteins"]), df.shape
    assert np.isfinite(df.to_numpy()).all(), "存在 NA/Inf"
    tmp = args.output + ".tmp"
    df.to_csv(tmp, float_format=cfg["output"]["float_format"])
    os.replace(tmp, args.output)
    sha = pl.sha256_of(args.output)
    json.dump({
        "prediction_scale": cfg["output"]["prediction_scale"],
        "n_test_rows": df.shape[0], "n_proteins": df.shape[1],
        "ensemble_members": want, "weights": w or "equal",
        "sha256": sha, "mb": round(os.path.getsize(args.output) / 1e6, 1),
        "test_labels_used": False,
        "config": os.path.relpath(args.config, ROOT),
        "config_sha256": pl.sha256_of(args.config),
        "code_version": _code_version(),
    }, open(os.path.splitext(args.output)[0] + "_manifest.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n写出 {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB)")
    print(f"sha256 = {sha}")
    print(f"用时 {time.time()-t0:.0f}s")

    if not args.no_validate:
        import subprocess
        print("\n== 自动执行提交格式自检 ==", flush=True)
        rc = subprocess.call([sys.executable,
                              os.path.join(ROOT, "scripts", "validate_submission.py"),
                              "--prediction", args.output, "--config", args.config])
        if rc != 0:
            print("⚠ 格式自检未通过（见上）；prediction.csv 已写出，请修复后重跑自检。")
            sys.exit(rc)


if __name__ == "__main__":
    main()
