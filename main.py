# -*- coding: utf-8 -*-
"""EFDiscover 主程序：Expression Functional Dependency（EFD）挖掘

论文：*EFDiscover: Prior-Free Discovery of Inter-Attribute Expression Functional
Dependencies in Noisy Data*（ICDE 2024）

整体流程（论文 §3.1–3.3）：

1. 以每个属性 ``Y`` 为 RHS，按与 ``Y`` 的相关性（``1-|corr|`` 升序）排列其余属性，
   得到候选 LHS 的插入顺序；
2. 用一个扩展二叉树 ``EBTree`` 维护 nonEFD：节点表示"包含/不包含某属性"，
   某个节点右子树为空即意味着该属性集合是 EFD；
3. 每个节点调用一次符号回归（``SymRegression``）判断损失是否低于 ``phi``：
   低于则整棵右子树剪掉（全是 EFD），高于则整棵子树剪掉（全是 nonEFD）；
4. 收集所有 EFD 及其拟合表达式。

用法::

    python main.py                                              # glass.data 前 60 行、8 个属性
    python main.py --rows 100 --cols -1                         # 全部属性
    python main.py --variant all                                # 三个变体一起跑（消融对比）
    python main.py --grid                                       # 论文的规模灵敏度扫描
"""
import argparse
import os
import random
import sys
import time

import numpy as np

from EBTree import EBTree
from SymRegression import SymRegression
from SymRegression1 import SymRegression1
from SymRegression2 import SymRegression2

VARIANTS = {
    "0": SymRegression,      # 完整 EFDiscover
    "1": SymRegression1,     # EFDiscover-Sym1：无增量发现（nonSymPop）
    "2": SymRegression2,     # EFDiscover-Sym2：无损失变化率早停（nonSymLoss）
}


# --------------------------------------------------------------------- 核心
def mine(filename, phi, row=-1, col=-1, cls=SymRegression, seed=0, **kwargs):
    """对一个数据集跑一遍 EFD 挖掘

    返回 ``(rules, sr_count, avg_loss, data)``：

    * ``rules``：``[(LHS 下标列表, RHS 下标, 表达式字符串, 损失), ...]``
    * ``sr_count``：实际执行的符号回归次数
    * ``avg_loss``：所有 EFD 拟合损失的平均值
    * ``data``：``SymRegression`` 实例，供输出列名等使用
    """
    data = cls(filename, row=row, col=col, seed=seed, **kwargs)
    corr = np.array(data.df.corr())
    rules, total_loss = [], 0.0

    for rhs in range(len(corr)):
        ids, weights = [], []
        for j in range(len(corr)):
            if j == rhs:
                continue
            ids.append(j)
            weights.append(1 - abs(corr[rhs][j]))
        if not weights or np.isnan(weights[0]):
            continue                      # 该属性与谁都不相关（或全常数），跳过

        order = [j for _, j in sorted(zip(weights, ids))]
        tree = EBTree(rhs, data.Calc_Regression, phi)
        for k in range(len(order)):
            tree.insert(order[k], order[k + 1:])

        for lhs in tree.GetEFD():
            expr, loss = data.Get_Regression_with_loss(lhs, rhs)
            rules.append((sorted(lhs), rhs, expr, loss))
            total_loss += loss

    avg_loss = total_loss / len(rules) if rules else 1.0
    return rules, data.cnt, avg_loss, data


def EFDiscover(filename, phi, row=-1, col=-1, **kwargs):
    """完整 EFDiscover"""
    rules, cnt, loss, _ = mine(filename, phi, row, col, SymRegression, **kwargs)
    return rules, cnt, loss


def EFDiscover1(filename, phi, row=-1, col=-1, **kwargs):
    """消融：关闭增量发现（nonSymPop）"""
    rules, cnt, loss, _ = mine(filename, phi, row, col, SymRegression1, **kwargs)
    return rules, cnt, loss


def EFDiscover2(filename, phi, row=-1, col=-1, **kwargs):
    """消融：关闭损失变化率早停（nonSymLoss）"""
    rules, cnt, loss, _ = mine(filename, phi, row, col, SymRegression2, **kwargs)
    return rules, cnt, loss


# --------------------------------------------------------------------- 输出
def format_rules(data, rules, top=20):
    """把规则格式化成可读文本"""
    names = data.column_names()
    lines = []
    for lhs, rhs, expr, loss in rules[:top]:
        body = ", ".join(f"x{a}" for a in lhs)
        label = names[rhs] if rhs < len(names) and names[rhs] != f"x{rhs}" else ""
        head = f"{body} -> x{rhs}" + (f"   [{label}]" if label else "")
        lines.append(f"{head}   loss={loss:.4g}")
        lines.append(f"    f = {expr}")
    if len(rules) > top:
        lines.append(f"...（另有 {len(rules) - top} 条，用 --top 调整）")
    return "\n".join(lines)


def report(name, rules, sr_count, loss, elapsed):
    print(f"\n变体：{name}")
    print(f"EFD 数量：{len(rules)}    符号回归次数：{sr_count}    "
          f"平均损失：{loss:.4g}    耗时：{elapsed:.2f} s")


# --------------------------------------------------------------- 论文扫描图
def grid(filename, phi, variants=("0", "1", "2"), seed=0, **kwargs):
    """复现论文的规模灵敏度实验：分别扫行数与列数"""
    row_list = [100, 200, 300, 500, 700, 1000, 1200, 1500, 1700, 2000]
    col_list = list(range(5, 12))
    rows_n = {k: {"cnt": [], "time": [], "sr": [], "loss": []} for k in variants}
    cols_n = {k: {"cnt": [], "time": [], "sr": [], "loss": []} for k in variants}

    for key in variants:
        cls = VARIANTS[key]
        for n_row in row_list:
            start = time.time()
            rules, sr, loss, _ = mine(filename, phi, row=n_row, col=-1,
                                      cls=cls, seed=seed, **kwargs)
            rows_n[key]["cnt"].append(len(rules))
            rows_n[key]["time"].append(time.time() - start)
            rows_n[key]["sr"].append(sr)
            rows_n[key]["loss"].append(loss)
        for n_col in col_list:
            start = time.time()
            rules, sr, loss, _ = mine(filename, phi, row=200, col=n_col,
                                      cls=cls, seed=seed, **kwargs)
            cols_n[key]["cnt"].append(len(rules))
            cols_n[key]["time"].append(time.time() - start)
            cols_n[key]["sr"].append(sr)
            cols_n[key]["loss"].append(loss)

    print("\n=== 行数扫描（列数固定）===")
    for key in variants:
        print(f"[{VARIANTS[key].VARIANT_NAME}] EFD={rows_n[key]['cnt']} "
              f"Time={[round(t, 2) for t in rows_n[key]['time']]}")

    print("\n=== 列数扫描（行数固定为 200）===")
    for key in variants:
        print(f"[{VARIANTS[key].VARIANT_NAME}] EFD={cols_n[key]['cnt']} "
              f"Time={[round(t, 2) for t in cols_n[key]['time']]}")

    import pandas as pd
    pd.DataFrame(rows_n).to_csv("row_sweep.csv")
    pd.DataFrame(cols_n).to_csv("col_sweep.csv")
    print("\n已写出 row_sweep.csv / col_sweep.csv")


# ----------------------------------------------------------------------- CLI
def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="EFDiscover: EFD 挖掘")
    ap.add_argument("--csv", default=os.path.join(here, "glass.data"), help="数据集路径")
    ap.add_argument("--rows", type=int, default=60, help="只取前多少行（-1 表示全部）")
    ap.add_argument("--cols", type=int, default=8, help="随机取多少列属性（-1 表示全部）")
    ap.add_argument("--phi", type=float, default=0.02, help="EFD 损失阈值 ε")
    ap.add_argument("--variant", default="0", choices=sorted(VARIANTS) + ["all"],
                    help="0=完整 EFDiscover，1=无增量发现，2=无损失变化率早停，all=三个都跑")
    ap.add_argument("--drop-id", action="store_true", help="无表头数据去掉第一列（样本编号）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=3, help="表达式树最大深度")
    ap.add_argument("--population", type=int, default=100, help="种群大小")
    ap.add_argument("--generations", type=int, default=20, help="遗传代数")
    ap.add_argument("--gp-loss", type=float, default=None,
                    help="符号回归内部早停阈值（适应度低于它就不再进化）；默认取 --phi/10")
    ap.add_argument("--top", type=int, default=20, help="最多打印多少条规则")
    ap.add_argument("--grid", action="store_true", help="跑论文的规模灵敏度扫描")
    ap.add_argument("--quiet", action="store_true", help="只打印统计，不打印表达式")
    args = ap.parse_args(argv)

    if not os.path.exists(args.csv):
        sys.exit(f"[错误] 找不到数据文件：{args.csv}")

    kwargs = dict(drop_id=args.drop_id,
                  max_depth=args.max_depth,
                  population_size=args.population,
                  generations=args.generations,
                  loss_threshold=args.gp_loss if args.gp_loss is not None else args.phi / 10)

    # 固定随机数，保证同一组参数可复现（遗传编程本身是随机的）
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.grid:
        grid(args.csv, args.phi, seed=args.seed, **kwargs)
        return 0

    keys = sorted(VARIANTS) if args.variant == "all" else [args.variant]
    for key in keys:
        cls = VARIANTS[key]
        start = time.time()
        rules, sr, loss, data = mine(args.csv, args.phi, row=args.rows, col=args.cols,
                                     cls=cls, seed=args.seed, **kwargs)
        elapsed = time.time() - start
        report(f"{key} / {cls.VARIANT_NAME}", rules, sr, loss, elapsed)
        if rules and not args.quiet:
            print(format_rules(data, rules, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
