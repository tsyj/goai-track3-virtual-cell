"""官方要求的第 2 条主命令：从头训练最终模型。

    python scripts/train.py --metadata <train_metadata> --proteome <train_proteome> \
        --config configs/final.yaml --output-dir runs/final

最终模型是 12 个近优配置的等权集成（两种因子拟合顺序 × 六个收缩强度点）。本脚本按
configs/final.yaml 冻结的配置，依次训练全部成员，并把每个成员的冻结产物写入
--output-dir；推理由 scripts/predict.py 单独完成，两者不共享进程。

标签边界：只使用官方训练文件中 split_final == 'train' 的 5,920 行标签。验证集只在
scripts/evaluate_val_mirror.py 里用于模型选择，不参与本脚本；测试蛋白真值在 io 层被拒绝
读取，任何路径都拿不到。

Jiao Xinyuan 2026-09-02
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from vcell import pipeline as pl                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="训练 metadata（核对用；实际路径见 configs）")
    ap.add_argument("--proteome", default=None, help="训练蛋白质组（核对用）")
    ap.add_argument("--test-metadata", default=None, help="测试 metadata（转导式设计矩阵需要）")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "final.yaml"))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "runs", "final"))
    ap.add_argument("--members", default=None, help="只训练这些成员（逗号分隔），默认全部")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("VCELL_LGB_THREADS", 16)))
    ap.add_argument("--finalize", action="store_true",
                    help="不训练，只扫描 --output-dir/members 汇总出 run.json（成员并行训练后用）")
    args = ap.parse_args()

    cfg = pl.load_config(args.config)
    t0 = time.time()
    # 转导式设计说明：本模型在 train+test 行的并集上拟合设计矩阵（测试 metadata 是官方
    # 公开发放的、不含任何蛋白真值），只有 train 行的标签可见。因此训练需要同时挂载
    # 训练与测试 metadata 两个 CSV；缺测试 metadata 会在下面直接报 FileNotFoundError。
    P, visible, is_test = pl.build_design(cfg)
    print(f"设计矩阵：{len(P.meta)} 行（train+test 并集） 可见标签 {visible.sum()} 行 "
          f"测试 {is_test.sum()} 行 蛋白 {P.X.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)

    members = cfg["ensemble"]["members"]
    if args.members:
        want = [m.strip() for m in args.members.split(",")]
        members = [m for m in members if m["name"] in want]
    os.makedirs(args.output_dir, exist_ok=True)

    done = []
    if args.finalize:
        for m in cfg["ensemble"]["members"]:
            d = os.path.join(args.output_dir, "members", m["name"])
            assert os.path.exists(os.path.join(d, "member.json")), f"成员 {m['name']} 未训练完成"
            done.append({"name": m["name"], "dir": os.path.relpath(d, args.output_dir), "secs": None})
        print(f"汇总 {len(done)} 个已训练成员", flush=True)
        members = []
    for i, m in enumerate(members, 1):
        t1 = time.time()
        print(f"[{i}/{len(members)}] 训练成员 {m['name']}", flush=True)
        art = pl.fit_member(cfg, m, P, visible, is_test, n_jobs=args.threads)
        d = pl.save_member(art, args.output_dir)
        done.append({"name": m["name"], "dir": os.path.relpath(d, args.output_dir),
                     "secs": round(time.time() - t1)})
        print(f"    完成，用时 {time.time()-t1:.0f}s → {d}", flush=True)

    if args.members and not args.finalize:
        print(f"\n本次只训练了子集 {[d['name'] for d in done]}，未写 run.json。")
        print("全部成员训练完成后执行：python scripts/train.py --finalize "
              f"--output-dir {args.output_dir}")
        return

    json.dump({
        "config": os.path.relpath(args.config, ROOT),
        "members": done,
        "n_rows": int(len(P.meta)), "n_train_labels": int(visible.sum()),
        "n_test_rows": int(is_test.sum()), "n_proteins": int(P.X.shape[1]),
        "proteins": [str(x) for x in P.proteins],
        "test_sample_ids": P.meta.loc[is_test, "sample_ID"].astype(str).tolist(),
        "total_secs": round(time.time() - t0),
        "note": "标签仅来自 split_final == 'train'；测试蛋白真值未被读取。",
    }, open(os.path.join(args.output_dir, "run.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n全部完成，共 {len(done)} 个成员，总用时 {time.time()-t0:.0f}s")
    print(f"run 目录：{args.output_dir}")


if __name__ == "__main__":
    main()
