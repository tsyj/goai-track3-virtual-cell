"""官方要求的第 1 条主命令：构建外部特征 / embedding。

本作品的最终模型 **不使用任何外部数据**——输入只有官方 metadata 字段（菌株、化合物、
培养基、温度、时间、数据来源、仪器、板号、孔位）与官方训练集蛋白质组标签。因此按官方
说明「未使用外部数据时明确写明"无需执行"」，本脚本不产生任何模型输入。

保留它有两个作用：
  1. 给出可执行、可核验的"确实没有外部输入"的证据：脚本扫描 src/ 与 scripts/ 里
     最终建模路径上的所有文件，确认没有对 data/genomes、data/chem 等外部目录的引用。
  2. 记录我们**探索过但未采用**的外部资源（1,011 株酵母基因组、PubChem、SGD），
     写入 external_data/source_manifest.json，含来源 URL、版本、下载日期、许可与 SHA256，
     供评审核对"外部数据申报"一致性。这些资源只出现在 scripts/experiments/ 的诊断脚本里，
     不进入 train.py / predict.py 的任何路径。

    python scripts/build_embeddings.py --check
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 最终建模路径（train.py / predict.py 会 import 的东西）
MODEL_PATH_FILES = [
    "src/vcell/io.py", "src/vcell/models.py", "src/vcell/pipeline.py", "src/vcell/harness.py",
    "src/vcell/design.py", "src/vcell/metrics.py",
    "scripts/train.py", "scripts/predict.py",
    "scripts/_member_defs.py", "scripts/_predict_member_impl.py", "scripts/_compose_impl.py",
]
# 外部数据来源：最终建模路径上一处都不该出现
FORBIDDEN = [
    r"data/genomes", r"data/chem", r"genesMatrix", r"SGD_features", r"pubchem",
    r"esm2", r"huggingface", r"uniprot", r"1011",
]
# 测试真值文件：只允许出现在 io.py 的隔离守卫里（那里是"拒绝读取"的实现）
TEST_TRUTH = [r"proteome_raw_test", r"quarantine"]
GUARD_FILE = "src/vcell/io.py"


def scan():
    hits, missing = [], []
    pats = FORBIDDEN
    for rel in MODEL_PATH_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        for pat in (pats if rel != GUARD_FILE else pats):
            for m in re.finditer(pat, src, re.I):
                line = src[:m.start()].count("\n") + 1
                ctx = src.splitlines()[line - 1].strip()
                if ctx.lstrip().startswith("#") or ctx.lstrip().startswith("*"):
                    continue          # 注释里的说明不算引用
                hits.append({"file": rel, "line": line, "pattern": pat, "text": ctx[:160]})
    # 测试真值字样：GUARD_FILE 之外一律禁止
    for rel in MODEL_PATH_FILES:
        if rel == GUARD_FILE:
            continue
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()
        for pat in TEST_TRUTH:
            for m in re.finditer(pat, src, re.I):
                line = src[:m.start()].count("\n") + 1
                ctx = src.splitlines()[line - 1].strip()
                if ctx.lstrip().startswith(("#", "*", '"', "'")):
                    continue          # 注释/文档字符串里的说明不算引用
                hits.append({"file": rel, "line": line, "pattern": pat, "text": ctx[:160]})
    return hits, missing


def guard_works():
    """正向验证：io 层确实拒绝交出测试蛋白真值。"""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from vcell.io import load_proteome
    try:
        load_proteome("test")
    except RuntimeError as e:
        return True, str(e).split("\n")[0][:120]
    except Exception as e:
        return False, f"抛出了非预期异常：{type(e).__name__}: {e}"
    return False, "load_proteome('test') 竟然成功返回了 —— 隔离失效"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="接受但不使用；保持与官方命令签名一致")
    ap.add_argument("--output", default=os.path.join(ROOT, "artifacts", "embeddings"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print("最终模型不使用外部数据/embedding —— 本步骤【无需执行】。")
    print("（保留本脚本用于自证与外部资源申报，详见 external_data/source_manifest.json）\n")

    hits, missing = scan()
    print("== 静态核验：最终建模路径是否引用了外部数据或测试真值 ==")
    for rel in MODEL_PATH_FILES:
        mark = "缺失" if rel in missing else "OK"
        print(f"  [{mark}] {rel}")
    if hits:
        print("\n⚠ 发现引用：")
        for h in hits:
            print(f"  {h['file']}:{h['line']}  /{h['pattern']}/  {h['text']}")
        sys.exit(1)
    print("\n✅ 未发现对外部数据目录的引用；测试真值字样只出现在 io.py 的隔离守卫内。")

    ok, msg = guard_works()
    print(f"\n== 正向核验：load_proteome('test') 是否被拒绝 ==\n  {'✅ 已拒绝' if ok else '❌ 未拒绝'}：{msg}")
    if not ok:
        sys.exit(1)

    man = os.path.join(ROOT, "external_data", "source_manifest.json")
    if os.path.exists(man):
        d = json.load(open(man, encoding="utf-8"))
        print(f"\n== 已申报的外部资源（{len(d.get('resources', []))} 项，均只用于诊断实验，未进入最终模型）==")
        for r in d.get("resources", []):
            print(f"  · {r['name']}  [{r['used_in_final_model']}]  {r['url']}")
    os.makedirs(args.output, exist_ok=True)
    open(os.path.join(args.output, "NOT_REQUIRED.txt"), "w").write(
        "最终模型不使用外部特征/embedding；本目录故意为空。\n"
        "The final model uses no external features/embeddings; this directory is intentionally empty.\n")
    print(f"\n写出 {args.output}/NOT_REQUIRED.txt")


if __name__ == "__main__":
    main()
