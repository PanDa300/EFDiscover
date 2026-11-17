# -*- coding: utf-8 -*-
"""RNN 驱动的符号回归（``--backend rnn``）

与 ``mygenetic``（遗传编程）互为替代：同样输入 ``Y`` 与 LHS 列，输出最优表达式与 MSE，
但搜索方式换成"LSTM 逐个符号生成表达式 + 精英序列的交叉熵训练"。

表达式的表示与生成
------------------
表达式按**前缀序列**表示，例如 ``c = a + b`` 为 ``['+', 'x0', 'x1']``。生成时用
一个"待填槽位"栈保证语法永远合法：

* 选到二元运算符 → 弹掉当前槽位、压入两个子槽位；
* 选到一元运算符（``sin``/``cos``）→ 弹掉当前槽位、压入一个子槽位；
* 选到叶子（属性或常数）→ 弹掉当前槽位。

再加上两条约束：槽位深度达到 ``max_depth`` 时只能选叶子；本次生成的 token 数不能
超过 ``max_len``（否则表达式无限长）。这样 LSTM 只需在每一步对"合法 token 集合"
输出概率分布，不会产生非法表达式。

训练方式
--------
每一轮采样若干个表达式、算 MSE，取误差最小的若干个作为"精英序列"，对它们做
教师强制的交叉熵训练（相当于 cross-entropy method，CEM）。精英里始终保留历史最优，
因此不会退化。这比 REINFORCE 稳定，也不需要额外的基线函数。

与遗传编程的对应关系
--------------------
``population_size`` → 每轮采样数；``generations`` → 训练轮数；
``reuse_pop``（增量发现）→ 是否跨节点复用已训练的 LSTM（词表不同的 LHS 会重建模型）；
``early_stop``（损失变化率早停）→ 连续若干轮无改进就停。
"""
import math
import random

import numpy as np

try:                      # torch 是可选的：没有装也不影响 --backend gp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:       # pragma: no cover
    HAS_TORCH = False

BINARY = ("+", "-", "*", "/")
UNARY = ("sin", "cos")

BOS = 0                   # 序列起始符（词表下标 0）


# --------------------------------------------------------------- 表达式工具
def _parse(tokens):
    """前缀 token 序列 → 语法树（嵌套元组）"""
    pos = [0]

    def build():
        tok = tokens[pos[0]]
        pos[0] += 1
        if isinstance(tok, tuple):          # ('const', 数值)
            return ("const", tok[1])
        if tok in UNARY:
            return ("unary", tok, build())
        if tok in BINARY:
            return ("binary", tok, build(), build())
        return ("var", int(tok[1:]))

    return build()


def _evaluate(node, X):
    """用 numpy 向量化求值某棵语法树"""
    kind = node[0]
    if kind == "const":
        return node[1]
    if kind == "var":
        col = node[1]
        return X[:, col] if col < X.shape[1] else 0.0
    if kind == "unary":
        v = _evaluate(node[2], X)
        return np.sin(v) if node[1] == "sin" else np.cos(v)

    a = _evaluate(node[2], X)
    b = _evaluate(node[3], X)
    with np.errstate(all="ignore"):
        if node[1] == "+":
            return a + b
        if node[1] == "-":
            return a - b
        if node[1] == "*":
            return a * b
        safe = np.where(np.abs(b) < 1e-9, 1.0, b)      # 除零保护
        return a / safe


def _to_string(node):
    """语法树 → 可读表达式串"""
    kind = node[0]
    if kind == "const":
        return f"{node[1]:.4g}"
    if kind == "var":
        return f"x{node[1]}"
    if kind == "unary":
        return f"{node[1]}({_to_string(node[2])})"
    return f"({_to_string(node[2])} {node[1]} {_to_string(node[3])})"


def _mse(tokens, X, y):
    """表达式在数据上的均方误差；非法表达式返回 inf"""
    try:
        pred = _evaluate(_parse(tokens), X)
        pred = np.broadcast_to(np.asarray(pred, dtype=float), y.shape)
        if not np.all(np.isfinite(pred)):
            return float("inf")
        if np.abs(pred).max() > 1e12:
            return float("inf")
        return float(np.mean((y - pred) ** 2))
    except Exception:
        return float("inf")


# ------------------------------------------------------------------- 模型
class _Policy(nn.Module):
    """LSTM 策略网络：输入已生成的 token，输出下一个 token 的分布"""

    def __init__(self, vocab_size, hidden=48, emb=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, lengths):
        e = self.emb(idx)
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out)


class RNNRegressor:
    """一次符号回归任务：在给定 LHS 上训练 LSTM 并生成最优表达式"""

    def __init__(self, num_features, max_len=15, hidden=48, lr=0.02, seed=0):
        if not HAS_TORCH:
            raise RuntimeError("RNN 后端需要 PyTorch：pip install torch")
        self.num_features = num_features
        self.max_len = max_len
        self.hidden = hidden
        self.lr = lr
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = [f"x{i}" for i in range(num_features)] + ["c"] + list(BINARY) + list(UNARY)
        self.index = {tok: i + 1 for i, tok in enumerate(self.vocab)}   # 0 留给 BOS
        torch.manual_seed(seed)
        self.model = _Policy(len(self.vocab) + 1, hidden=hidden).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------ 生成
    def _mask_for(self, depth, n_tokens, n_slots):
        """当前状态下允许选择的 token 集合（bool 数组，长度 = 词表 + 1）"""
        mask = np.zeros(len(self.vocab) + 1, dtype=bool)
        left = self.max_len - n_tokens - 1          # 本步之后还剩多少 token 预算
        leaf_ok = left >= n_slots - 1               # 选叶子：槽位数 -1
        unary_ok = left >= n_slots                  # 选一元：槽位数不变
        binary_ok = left >= n_slots + 1             # 选二元：槽位数 +1
        if depth >= self.max_len:                   # 理论上不会发生，保险
            leaf_ok, unary_ok, binary_ok = True, False, False
        if leaf_ok:
            for i in range(self.num_features + 1):  # 属性 + 常数 c
                mask[i + 1] = True
        if unary_ok:
            for op in UNARY:
                mask[self.index[op]] = True
        if binary_ok:
            for op in BINARY:
                mask[self.index[op]] = True
        if not mask.any():                          # 兜底
            mask[1:self.num_features + 2] = True
        return mask

    def sample_batch(self, n, max_depth, const_scale):
        """并行采样 n 个合法表达式（每一步一次批量前向）"""
        states = [{"tokens": [], "slots": [0], "seq": [BOS]} for _ in range(n)]
        active = list(range(n))

        while active:
            seqs = [states[i]["seq"] for i in active]
            lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)   # pack 要求 CPU 张量
            width = int(max(len(s) for s in seqs))
            batch = torch.zeros((len(active), width), dtype=torch.long, device=self.device)
            for r, seq in enumerate(seqs):
                batch[r, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model(batch, lens)[torch.arange(len(active)), lens - 1]
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            done = []
            for r, i in enumerate(active):
                st = states[i]
                depth = st["slots"][-1]
                mask = self._mask_for(depth, len(st["tokens"]), len(st["slots"]))
                p = np.where(mask, probs[r], 0.0)
                if p.sum() <= 0:
                    p = mask.astype(float)
                p = p / p.sum()
                choice = int(np.random.choice(len(p), p=p))
                tok = "c" if choice == self.index.get("c") else \
                      (self.vocab[choice - 1] if choice > 0 else "c")

                st["slots"].pop()
                if tok in BINARY:
                    st["slots"] += [depth + 1, depth + 1]
                    st["tokens"].append(tok)
                elif tok in UNARY:
                    st["slots"].append(depth + 1)
                    st["tokens"].append(tok)
                elif tok == "c":
                    value = float(np.random.normal(0.0, const_scale))
                    st["tokens"].append(("const", value))
                else:
                    st["tokens"].append(tok)

                st["seq"].append(choice)
                if not st["slots"]:
                    done.append(i)
            active = [i for i in active if i not in done]

        return [states[i]["tokens"] for i in range(n)]

    # ------------------------------------------------------------ 训练
    def _train_step(self, sequences):
        """对精英序列做一次教师强制交叉熵训练"""
        if not sequences:
            return 0.0
        lens = torch.tensor([len(s) - 1 for s in sequences], dtype=torch.long)
        width = int(max(len(s) - 1 for s in sequences))
        inputs = torch.zeros((len(sequences), width), dtype=torch.long, device=self.device)
        targets = torch.full((len(sequences), width), -100, dtype=torch.long, device=self.device)
        for r, seq in enumerate(sequences):
            n = len(seq) - 1
            inputs[r, :n] = torch.tensor(seq[:-1], dtype=torch.long, device=self.device)
            targets[r, :n] = torch.tensor(seq[1:], dtype=torch.long, device=self.device)

        logits = self.model(inputs, lens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()
        return float(loss)

    # ------------------------------------------------------------ 主流程
    def fit(self, X, y, max_depth=3, loss_threshold=0.01,
            iterations=10, samples=24, elite=6, early_stop=True, verbose=False):
        """训练并返回 ``(最优表达式串, 最优 MSE)``"""
        scale = float(np.std(y)) or 1.0
        best_tokens, best_loss = None, float("inf")
        best_seq, stale = None, 0

        for it in range(iterations):
            const_scale = scale * (0.5 ** it)        # 常数范围逐轮收紧
            batch = self.sample_batch(samples, max_depth, const_scale)
            losses = [_mse(toks, X, y) for toks in batch]

            order = np.argsort(losses)
            if losses[order[0]] < best_loss:
                best_loss = float(losses[order[0]])
                best_tokens = batch[int(order[0])]
                best_seq = [BOS] + self._encode(best_tokens)
                stale = 0
            else:
                stale += 1

            if best_loss < loss_threshold:           # 已经足够好
                break

            elites = [batch[int(i)] for i in order[:elite] if np.isfinite(losses[int(i)])]
            seqs = [[BOS] + self._encode(t) for t in elites]
            if best_tokens is not None:              # 精英保留：历史最优进入训练集
                seqs.append(best_seq)
            self._train_step(seqs)

            if early_stop and stale >= 3:            # 连续无改进 → 收敛
                break
            if verbose:
                print(f"    rnn iter {it + 1}/{iterations}: best_MSE={best_loss:.6g}")

        if best_tokens is None:
            return "1", float("inf")
        return _to_string(_parse(best_tokens)), best_loss

    def _encode(self, tokens):
        out = []
        for tok in tokens:
            if isinstance(tok, tuple):
                out.append(self.index["c"])
            else:
                out.append(self.index[tok])
        return out


# --------------------------------------------------------------- 对外接口
def fit_model(data, target_column, input_columns, init_pop=None, loss_threshold=0.01,
              max_depth=3, reuse_pop=True, early_stop=True,
              population_size=100, generations=20, mutation_rate=0.1, verbose=False):
    """与 ``mygenetic.fit_model`` 同签名，供 ``SymRegression`` 直接替换

    返回 ``(state, expression, loss)``；``state`` 是训练好的 ``RNNRegressor``，
    开启 ``reuse_pop`` 时可在下一个节点继续用（词表不同则自动重建）。
    """
    input_columns = list(input_columns)
    X = np.asarray(data)[:, input_columns]
    y = np.asarray(data)[:, target_column]

    regressor = init_pop if (reuse_pop and isinstance(init_pop, RNNRegressor)
                             and init_pop.num_features == len(input_columns)) else None
    if regressor is None:
        regressor = RNNRegressor(len(input_columns))

    iterations = max(3, generations // 2)
    samples = max(8, population_size // 4)
    expr, loss = regressor.fit(X, y, max_depth=max_depth, loss_threshold=loss_threshold,
                               iterations=iterations, samples=samples,
                               elite=max(3, samples // 4), early_stop=early_stop,
                               verbose=verbose)
    return (regressor if reuse_pop else None), expr, loss
