# -*- coding: utf-8 -*-
"""变体 EFDiscover-Sym2（nonSymLoss）

与完整版唯一的差别：**关闭损失变化率早停**。回归一律跑满 ``generations`` 代，
不看相邻两代的适应度变化是否已经收敛（种群仍然沿 EBTree 增量复用）。

对应 ``main.py`` 里的 ``EFDiscover2``，结果表里记作 ``nonSymLoss``。
"""
from SymRegression import SymRegression


class SymRegression2(SymRegression):
    VARIANT_NAME = "EFDiscover-Sym2 (nonSymLoss)"
    REUSE_POP = True
    EARLY_STOP = False    # 不做损失变化率早停
