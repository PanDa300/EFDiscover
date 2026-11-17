# -*- coding: utf-8 -*-
"""变体 EFDiscover-Sym1（nonSymPop）

与完整版唯一的差别：**关闭增量发现**。每次符号回归都用全新随机种群，
不沿 EBTree 复用上一节点的种群，因此 ``pop`` 参数被忽略。

对应 ``main.py`` 里的 ``EFDiscover1``，结果表里记作 ``nonSymPop``。
"""
from SymRegression import SymRegression


class SymRegression1(SymRegression):
    VARIANT_NAME = "EFDiscover-Sym1 (nonSymPop)"
    REUSE_POP = False     # 不做增量发现
    EARLY_STOP = True
