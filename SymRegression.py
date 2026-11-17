# -*- coding: utf-8 -*-
"""符号回归封装：把数据集交给 ``SymbolicRNN``，给出某个属性集合上的最优表达式与损失

约定（与 ``EBTree`` 一致）：

* ``Calc_Regression(LHS, RHS, pop) -> (pop_now, loss)``：EBTree 每次判断节点是否为
  nonEFD 都会调用它，``pop`` 是父节点传下来的状态（RNN 后端即已训练的模型）；
* ``Get_Regression(LHS, RHS, pop) -> expression``：规则输出时取一次最优表达式。

三个变体：

===============  =================================  ==========================
类               增量发现（模型复用）               损失变化率早停
===============  =================================  ==========================
``SymRegression``   开启（``REUSE_POP = True``）       开启（``EARLY_STOP = True``）   完整 EFDiscover
``SymRegression1``  关闭                                开启                           EFDiscover-Sym1 / nonSymPop
``SymRegression2``  开启                                关闭                           EFDiscover-Sym2 / nonSymLoss
===============  =================================  ==========================
"""
import pandas as pd


def _looks_numeric(cells):
    try:
        for v in cells:
            if v == "":
                continue
            float(v)
        return True
    except ValueError:
        return False


class SymRegression:
    """完整版：增量发现 + 损失变化率早停"""

    VARIANT_NAME = "EFDiscover"
    REUSE_POP = True      # 增量发现：沿用上一个节点训练过的模型
    EARLY_STOP = True     # 损失变化率早停

    def __init__(self, file_name, row=-1, col=-1, seed=0, drop_id=False,
                 max_depth=3, samples=50, iterations=10,
                 loss_threshold=0.01, pretrain=True, pretrain_tasks=20):
        self.file_name = file_name
        self.seed = seed
        self.max_depth = max_depth
        self.samples = samples            # 每轮采样的表达式个数
        self.iterations = iterations      # 每个节点的训练轮数
        self.loss_threshold = loss_threshold
        self.rnn_pretrain = pretrain
        self.rnn_pretrain_tasks = pretrain_tasks
        self._rnn = None                # 惰性创建的 RNN 引擎（数据集级预训练）
        self.cnt = 0                    # 实际执行的符号回归次数（SR 计数）
        self.cache = {}                 # (RHS, LHS 元组) -> (损失, 表达式)

        self.df = self._read(file_name, drop_id)
        if row != -1:
            self.df = self.df.head(row)
        if col != -1 and col < self.df.shape[1]:
            self.df = self.df.sample(n=col, axis="columns", random_state=seed)
        # 抽样后列标签会变得不连续，这里统一改成位置下标 0..d-1，
        # 原始列名另存到 self.names，供输出使用（EBTree 内部一律用位置下标）
        self.names = [str(c) for c in self.df.columns]
        self.df = self.df.reset_index(drop=True)
        self.df.columns = range(self.df.shape[1])
        self._data = None

    # ------------------------------------------------------------------ 数据
    @staticmethod
    def _read(file_name, drop_id=False):
        """读 CSV

        * 首行能整体解析为数字时视为**无表头**（如 ``glass.data``），否则按表头读；
        * 只保留数值列；
        * ``drop_id=True`` 时去掉无表头数据的第一列（通常是样本编号）；
        * 无表头数据的列名记为位置下标 ``0..d-1``。
        """
        with open(file_name, encoding="utf-8", errors="ignore") as fh:
            first = fh.readline().strip()
        has_header = not _looks_numeric(first.split(","))
        df = pd.read_csv(file_name, header=0 if has_header else None)
        df = df.select_dtypes(include=["float64", "int64"])
        if drop_id and not has_header and df.shape[1] > 1:
            df = df.iloc[:, 1:]
        if not has_header:
            df.columns = range(df.shape[1])
        return df

    @property
    def data(self):
        if self._data is None:
            self._data = self.df.values
        return self._data

    def column_names(self):
        return list(self.names)

    # ------------------------------------------------------------- 回归调用
    def _fit(self, LHS, RHS, pop):
        """跑一次符号回归，并把结果写入缓存

        缓存很关键：同一个 (LHS, RHS) 在 EBTree 判定时已经回归过一次，
        报告表达式时直接用那次的结果，否则两次独立的搜索会给出
        不一致的损失（曾出现"判定通过、打印出来的损失却大于 φ"的情况）。
        """
        samples = max(8, self.samples)
        expr, loss = self._rnn_engine().fit(
            list(LHS), RHS,
            max_depth=self.max_depth,
            iterations=max(3, self.iterations),
            samples=samples,
            elite=max(4, samples // 6),
            loss_threshold=self.loss_threshold,
            early_stop=self.EARLY_STOP,
            reuse_pop=self.REUSE_POP,
        )

        self.cnt += 1
        self.cache[(RHS, tuple(LHS))] = (float(loss), expr)
        return None, float(loss)

    def _rnn_engine(self):
        """惰性创建 RNN 引擎，并在整个数据集上预训练一次"""
        from SymbolicRNN import RNNSymbolicRegressor
        if self._rnn is None:
            self._rnn = RNNSymbolicRegressor(self.data, seed=self.seed,
                                             max_len=max(8, self.max_depth * 4 + 3))
            if self.rnn_pretrain:
                self._rnn.pretrain(tasks=self.rnn_pretrain_tasks, max_depth=self.max_depth)
        return self._rnn

    def Calc_Regression(self, LHS, RHS, pop, loss_threshold=None):
        """EBTree 节点判定：返回 ``(新种群, 最小损失)``；LHS 为空时损失记为 1"""
        if not LHS:
            return pop, 1.0
        if loss_threshold is not None:
            self.loss_threshold = loss_threshold
        return self._fit(LHS, RHS, pop)

    def explain(self, LHS, RHS):
        """返回 ``(表达式, 损失)``：优先用缓存里那次回归的结果"""
        key = (RHS, tuple(LHS))
        if key not in self.cache:
            self._fit(LHS, RHS, [])
        loss, expr = self.cache[key]
        return expr, loss

    def Get_Regression(self, LHS, RHS, pop=None):
        """取该属性集合上的最优表达式（字符串）"""
        if not LHS:
            return "1"
        return self.explain(LHS, RHS)[0]

    def Get_Regression_with_loss(self, LHS, RHS, pop=None):
        """同时返回表达式与损失，便于报告"""
        if not LHS:
            return "1", 1.0
        return self.explain(LHS, RHS)
