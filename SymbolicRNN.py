# -*- coding: utf-8 -*-
"""RNN 驱动的符号回归

输入 ``Y`` 与候选 LHS 列，输出最优表达式与 MSE。用**编码器-解码器**式的自回归生成，
并在数据集级做一次预训练：

1. **编码器**：读入"属性上下文"序列——对数据集里每个属性给出
   ``[是否为候选 LHS, |corr(x_j, y)|, std(x_j)/std(y), (mean(x_j)-mean(y))/std(y)]``，
   经 LSTM 编码成解码器的初始隐状态。不同 RHS / 不同候选集因此共享同一个网络。
2. **解码器（自回归）**：逐符号生成前缀表达式，每步把已生成 token 喂回 LSTM。
   用一个"待填槽位"栈 + 深度/长度约束保证语法永远合法，只允许选候选 LHS 里的属性，
   并且要求表达式至少出现一个属性（否则会退化成常数函数，等于说"Y 是常数"）。
3. **预训练 + 微调**：先在数据集上随机采样若干 (RHS, LHS) 任务做一遍交叉熵精英训练，
   得到一个任务无关的先验；每个 EBTree 节点再从这份先验出发微调几轮。
4. **常数最小二乘精修**：把表达式里的常数当作自由参数，用最小二乘把它们重解一遍
   （表达式对常数是仿射时是精确解），误差变小才采纳。

训练仍用 CEM（cross-entropy method）：每轮采样一批表达式、按 MSE 取精英、
对精英序列做教师强制交叉熵；历史最优始终留在精英集里。
"""
import copy

import numpy as np

try:                      # 需要 PyTorch
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
    """前缀 token 序列 → ``(语法树, 常数列表)``

    常数叶子记为 ``('cparam', i)``，``i`` 是在常数列表里的下标，供最小二乘精修使用。
    """
    consts, pos = [], [0]

    def build():
        tok = tokens[pos[0]]
        pos[0] += 1
        if isinstance(tok, tuple):                 # ('const', 数值)
            consts.append(float(tok[1]))
            return ("cparam", len(consts) - 1)
        if tok in UNARY:
            return ("unary", tok, build())
        if tok in BINARY:
            return ("binary", tok, build(), build())
        return ("var", int(tok[1:]))

    tree = build()
    return tree, consts


def _evaluate(node, X, consts=None):
    """向量化求值；常数叶子从 ``consts`` 取值（缺省为 0）"""
    kind = node[0]
    if kind == "cparam":
        return 0.0 if consts is None else float(consts[node[1]])
    if kind == "var":
        col = node[1]
        return X[:, col] if col < X.shape[1] else 0.0
    if kind == "unary":
        v = _evaluate(node[2], X, consts)
        return np.sin(v) if node[1] == "sin" else np.cos(v)

    a = _evaluate(node[2], X, consts)
    b = _evaluate(node[3], X, consts)
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
    kind = node[0]
    if kind == "cparam":
        return None                                     # 由 _tokens_to_string 替换
    if kind == "var":
        return f"x{node[1]}"
    if kind == "unary":
        return f"{node[1]}({_to_string(node[2])})"
    return f"({_to_string(node[2])} {node[1]} {_to_string(node[3])})"


def _predict(tokens, X):
    """token 序列 → 预测向量（含常数）"""
    tree, consts = _parse(tokens)
    return _evaluate(tree, X, consts)


def _mse(tokens, X, y):
    """表达式在数据上的均方误差；非法表达式返回 inf"""
    try:
        pred = np.broadcast_to(np.asarray(_predict(tokens, X), dtype=float), y.shape)
        if not np.all(np.isfinite(pred)) or np.abs(pred).max() > 1e12:
            return float("inf")
        return float(np.mean((y - pred) ** 2))
    except Exception:
        return float("inf")


def _replace_consts(tokens, consts):
    """按顺序把 token 里的常数替换成新值"""
    out, i = [], 0
    for tok in tokens:
        if isinstance(tok, tuple):
            out.append(("const", float(consts[i])))
            i += 1
        else:
            out.append(tok)
    return out


def uses_attribute(tokens):
    """化简后表达式里是否还真的用到了属性（避免退化成常数函数）"""
    tree, consts = _parse(tokens)
    tree = _simplify(_substitute(tree, consts))

    def walk(node):
        kind = node[0]
        if kind == "var":
            return True
        if kind == "unary":
            return walk(node[2])
        if kind == "binary":
            return walk(node[2]) or walk(node[3])
        return False

    return walk(tree)


def refine_constants(tokens, X, y, rounds=2):
    """常数最小二乘精修

    把表达式写成"常数向量 c 的仿射函数"：``f(x) = v0 + Σ c_i·(v_i − v0)``，
    其中 ``v0`` 是全部常数取 0 时的输出、``v_i`` 只有第 i 个常数取 1。
    对该线性组合做最小二乘即可一次解出全部常数。表达式对常数非线性时结果可能更差，
    因此每一轮都重新算 MSE，只有变好才采纳。

    额外一条约束：精修后不能把属性系数压成 0（否则 "x6 -> Fe" 会退化成常数函数，
    语义上不再是被属性决定）。返回 ``(tokens, mse)``。
    """
    best_tokens = tokens
    best_loss = _mse(tokens, X, y)

    for _ in range(rounds):
        tree, consts = _parse(best_tokens)
        k = len(consts)
        if k == 0:
            break
        v0 = np.broadcast_to(np.asarray(_evaluate(tree, X, np.zeros(k)), dtype=float), y.shape)
        cols = [v0]
        for i in range(k):
            e = np.zeros(k)
            e[i] = 1.0
            vi = np.broadcast_to(np.asarray(_evaluate(tree, X, e), dtype=float), y.shape)
            cols.append(vi - v0)
        A = np.column_stack(cols)
        if not np.all(np.isfinite(A)):
            break
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            break
        new_tokens = _replace_consts(best_tokens, coef[1:])
        new_loss = _mse(new_tokens, X, y)
        if new_loss < best_loss and (uses_attribute(new_tokens) or not uses_attribute(best_tokens)):
            best_tokens, best_loss = new_tokens, new_loss
        else:
            break
    return best_tokens, best_loss


def expression_string(tokens):
    """token 序列 → 可读表达式串（常数按 4 位有效数字输出，并做常数折叠/去冗余）"""
    tree, consts = _parse(tokens)
    tree = _substitute(tree, consts)
    tree = _simplify(tree)
    return _tree_string(tree)


def _substitute(node, consts):
    """('cparam', i) → ('const', 数值)"""
    kind = node[0]
    if kind == "cparam":
        return ("const", float(consts[node[1]]))
    if kind == "unary":
        return ("unary", node[1], _substitute(node[2], consts))
    if kind == "binary":
        return ("binary", node[1], _substitute(node[2], consts), _substitute(node[3], consts))
    return node


def _simplify(node):
    """常数折叠与平凡结构化简，例如 ``1 * X → X``、``X + 0 → X``"""
    kind = node[0]
    if kind in ("const", "var"):
        return node
    if kind == "unary":
        child = _simplify(node[2])
        if child[0] == "const":
            v = float(child[1])
            try:
                return ("const", float(np.sin(v) if node[1] == "sin" else np.cos(v)))
            except Exception:
                return ("unary", node[1], child)
        return ("unary", node[1], child)

    op, left, right = node[1], _simplify(node[2]), _simplify(node[3])
    if left[0] == "const" and right[0] == "const":
        a, b = float(left[1]), float(right[1])
        try:
            if op == "+":
                return ("const", a + b)
            if op == "-":
                return ("const", a - b)
            if op == "*":
                return ("const", a * b)
            if abs(b) > 1e-9:
                return ("const", a / b)
        except Exception:
            pass
    if op == "*" and right[0] == "const" and abs(float(right[1]) - 1.0) < 1e-9:
        return left
    if op == "*" and left[0] == "const" and abs(float(left[1]) - 1.0) < 1e-9:
        return right
    if op == "*" and right[0] == "const" and abs(float(right[1])) < 1e-9:
        return ("const", 0.0)
    if op in ("+", "-") and right[0] == "const" and abs(float(right[1])) < 1e-9:
        return left
    if op == "+" and left[0] == "const" and abs(float(left[1])) < 1e-9:
        return right
    if op == "/" and right[0] == "const" and abs(float(right[1]) - 1.0) < 1e-9:
        return left
    return ("binary", op, left, right)


def _tree_string(node):
    kind = node[0]
    if kind == "const":
        return f"{node[1]:.4g}"
    if kind == "var":
        return f"x{node[1]}"
    if kind == "unary":
        return f"{node[1]}({_tree_string(node[2])})"
    return f"({_tree_string(node[2])} {node[1]} {_tree_string(node[3])})"


# ------------------------------------------------------------------- 网络
class _EncDec(nn.Module):
    """属性上下文编码器 + 表达式解码器"""

    def __init__(self, vocab_size, ctx_dim=4, hidden=48, emb=32):
        super().__init__()
        self.encoder = nn.LSTM(ctx_dim, hidden, batch_first=True)
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=BOS)
        self.decoder = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def encode(self, ctx):
        """ctx: (B, 属性数, 4) → 解码器初始 (h, c)"""
        _, (h, c) = self.encoder(ctx)
        return h, c

    def logits(self, ctx, idx):
        """对已生成序列计算每步下一个 token 的 logits（自回归、无 pack，末尾补位不影响前部）"""
        h, c = self.encode(ctx)
        out, _ = self.decoder(self.emb(idx), (h, c))
        return self.head(out)


# ------------------------------------------------------------------- 引擎
def _flatten(model):
    """让 LSTM 权重回到连续内存（deepcopy / load_state_dict 之后会变散）"""
    for module in (model.encoder, model.decoder):
        try:
            module.flatten_parameters()
        except Exception:                              # pragma: no cover
            pass


class RNNSymbolicRegressor:
    """整个数据集共用一个编码器-解码器；每类任务由上下文区分"""

    def __init__(self, data, seed=0, max_len=15, hidden=48, emb=32, lr=0.02):
        if not HAS_TORCH:
            raise RuntimeError("RNN 后端需要 PyTorch：pip install torch")
        self.data = np.asarray(data, dtype=float)
        self.n_rows, self.n_attr = self.data.shape
        self.max_len = max_len
        self.seed = seed
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab = [f"x{i}" for i in range(self.n_attr)] + ["c"] + list(BINARY) + list(UNARY)
        self.index = {tok: i + 1 for i, tok in enumerate(self.vocab)}
        self.leaf_ids = [self.index[f"x{i}"] for i in range(self.n_attr)] + [self.index["c"]]

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = _EncDec(len(self.vocab) + 1, hidden=hidden, emb=emb).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.pretrained = None          # 预训练后的权重（state_dict）
        self.work_model = None          # 微调用的副本

    # ------------------------------------------------------------ 上下文
    def context(self, lhs_cols, rhs):
        """把 (候选 LHS, RHS) 编码成 (属性数, 4) 的上下文序列"""
        y = self.data[:, rhs]
        mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
        lhs = set(lhs_cols)
        ctx = np.zeros((self.n_attr, 4), dtype=float)
        for j in range(self.n_attr):
            col = self.data[:, j]
            cand = 1.0 if j in lhs else 0.0
            if j in lhs:
                with np.errstate(all="ignore"):
                    corr = np.corrcoef(col, y)[0, 1]
                corr = 0.0 if not np.isfinite(corr) else abs(float(corr))
            else:
                corr = 0.0
            ctx[j] = [cand, corr,
                      float(np.std(col)) / sd,
                      (float(np.mean(col)) - mu) / sd]
        return ctx

    def _ctx_tensor(self, contexts):
        return torch.tensor(np.asarray(contexts, dtype=np.float32), device=self.device)

    # ------------------------------------------------------------ 生成
    def _mask(self, lhs_cols, depth, n_tokens, n_slots, has_attr, max_depth):
        mask = np.zeros(len(self.vocab) + 1, dtype=bool)
        left = self.max_len - n_tokens - 1
        leaf_ok = left >= n_slots - 1
        unary_ok = left >= n_slots
        binary_ok = left >= n_slots + 1
        if depth >= max_depth:               # 到达深度上限，只能收叶子
            leaf_ok, unary_ok, binary_ok = True, False, False
        if leaf_ok:
            for j in lhs_cols:
                mask[self.index[f"x{j}"]] = True
            if has_attr:                     # 还没出现过属性时不允许只用常数
                mask[self.index["c"]] = True
        if unary_ok:
            for op in UNARY:
                mask[self.index[op]] = True
        if binary_ok:
            for op in BINARY:
                mask[self.index[op]] = True
        if not mask.any():                   # 兜底：必须有属性
            for j in lhs_cols:
                mask[self.index[f"x{j}"]] = True
        return mask

    def sample_batch(self, lhs_cols, rhs, n, max_depth, const_scale):
        """并行采样 n 个合法表达式（每步一次批量前向）"""
        ctx = self._ctx_tensor([self.context(lhs_cols, rhs)] * n)
        states = [{"tokens": [], "slots": [0], "seq": [BOS], "has_attr": False}
                  for _ in range(n)]
        active = list(range(n))

        while active:
            seqs = [states[i]["seq"] for i in active]
            width = max(len(s) for s in seqs)
            batch = torch.zeros((len(active), width), dtype=torch.long, device=self.device)
            for r, seq in enumerate(seqs):
                batch[r, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.model.logits(ctx[active], batch)
                rows = torch.arange(len(active), device=self.device)
                last = torch.tensor([len(s) - 1 for s in seqs], device=self.device)
                probs = F.softmax(logits[rows, last], dim=-1).cpu().numpy()

            finished = []
            for r, i in enumerate(active):
                st = states[i]
                depth = st["slots"][-1]
                mask = self._mask(lhs_cols, depth, len(st["tokens"]),
                                  len(st["slots"]), st["has_attr"], max_depth)
                p = np.where(mask, probs[r], 0.0)
                p = mask.astype(float) if p.sum() <= 0 else p / p.sum()
                choice = int(np.random.choice(len(p), p=p))

                st["slots"].pop()
                if choice == 0 or choice > len(self.vocab):        # 兜底
                    tok = ("const", float(np.random.normal(0.0, const_scale)))
                else:
                    tok = self.vocab[choice - 1]
                    if tok == "c":
                        tok = ("const", float(np.random.normal(0.0, const_scale)))

                if isinstance(tok, tuple):
                    pass
                elif tok in BINARY:
                    st["slots"] += [depth + 1, depth + 1]
                elif tok in UNARY:
                    st["slots"].append(depth + 1)
                elif tok.startswith("x"):
                    st["has_attr"] = True
                st["tokens"].append(tok)
                st["seq"].append(choice)
                if not st["slots"]:
                    finished.append(i)
            active = [i for i in active if i not in finished]

        return [states[i]["tokens"] for i in range(n)]

    # ------------------------------------------------------------ 训练
    def _train_step(self, contexts, sequences):
        """对精英序列做教师强制交叉熵训练"""
        if not sequences:
            return 0.0
        width = max(len(s) - 1 for s in sequences)
        inputs = torch.zeros((len(sequences), width), dtype=torch.long, device=self.device)
        targets = torch.full((len(sequences), width), -100, dtype=torch.long, device=self.device)
        for r, seq in enumerate(sequences):
            n = len(seq) - 1
            inputs[r, :n] = torch.tensor(seq[:-1], dtype=torch.long, device=self.device)
            targets[r, :n] = torch.tensor(seq[1:], dtype=torch.long, device=self.device)

        logits = self.model.logits(self._ctx_tensor(contexts), inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()
        return float(loss)

    def _encode(self, tokens):
        return [self.index["c"] if isinstance(t, tuple) else self.index[t] for t in tokens]

    # ------------------------------------------------------------ 预训练
    def pretrain(self, tasks=20, samples=24, iterations=2, max_depth=3, verbose=False):
        """在数据集上随机采样 (RHS, LHS) 任务，先把网络训成一个通用先验"""
        if self.n_attr < 3:
            return
        for t in range(tasks):
            rhs = int(np.random.randint(self.n_attr))
            others = [j for j in range(self.n_attr) if j != rhs]
            size = int(np.random.randint(1, min(3, len(others)) + 1))
            lhs = [int(j) for j in np.random.choice(others, size=size, replace=False)]
            ctx = self.context(lhs, rhs)
            y = self.data[:, rhs]
            scale = float(np.std(y)) or 1.0

            batch = []
            for it in range(iterations):
                batch = self.sample_batch(lhs, rhs, samples, max_depth, scale * (0.5 ** it))
                losses = [_mse(toks, self.data, y) for toks in batch]
                order = np.argsort(losses)
                elites = [batch[int(i)] for i in order[: max(3, samples // 4)]
                          if np.isfinite(losses[int(i)])]
                self._train_step([ctx] * len(elites), [[BOS] + self._encode(t) for t in elites])

            if verbose:
                best = min(_mse(t, self.data, y) for t in batch)
                print(f"  pretrain {t + 1}/{tasks}: LHS={lhs} RHS={rhs} best={best:.4g}")

        self.pretrained = copy.deepcopy(self.model.state_dict())
        _flatten(self.model)

    # ------------------------------------------------------------ 单节点微调
    def fit(self, lhs_cols, rhs, max_depth=3, iterations=8, samples=24, elite=6,
            loss_threshold=0.01, early_stop=True, reuse_pop=True, verbose=False):
        """从预训练权重出发微调，返回 ``(表达式串, MSE)``"""
        if self.pretrained is not None and (self.work_model is None or not reuse_pop):
            self.work_model = copy.deepcopy(self.model)
            self.work_model.load_state_dict(self.pretrained)
            _flatten(self.work_model)                     # deepcopy 后权重不连续，压平一次
            self.opt = torch.optim.Adam(self.work_model.parameters(), lr=self.lr)
        if self.work_model is not None:
            self.model = self.work_model

        y = self.data[:, rhs]
        scale = float(np.std(y)) or 1.0
        ctx = self.context(lhs_cols, rhs)

        best_tokens, best_loss, stale = None, float("inf"), 0
        for it in range(iterations):
            batch = self.sample_batch(lhs_cols, rhs, samples, max_depth, scale)
            if best_tokens is not None:                  # 上一轮最优直接回到候选池
                batch = batch + [best_tokens]

            # 先对排名靠前的一批做常数最小二乘精修，再按精修后的误差排序
            order = np.argsort([_mse(toks, self.data, y) for toks in batch])
            refined = [refine_constants(batch[int(i)], self.data, y)
                       for i in order[:max(elite * 2, 8)]]
            refined.sort(key=lambda pair: pair[1])

            top_tokens, top_loss = refined[0]
            if top_loss < best_loss:
                best_tokens, best_loss, stale = top_tokens, top_loss, 0
            else:
                stale += 1

            if best_loss < loss_threshold or (early_stop and stale >= 3):
                break

            elites = [toks for toks, _ in refined[:elite]]
            seqs = [[BOS] + self._encode(t) for t in elites]
            if best_tokens is not None:
                seqs.append([BOS] + self._encode(best_tokens))       # 精英保留
            self._train_step([ctx] * len(seqs), seqs)
            if verbose:
                print(f"    rnn iter {it + 1}/{iterations}: best={best_loss:.6g}")

        if best_tokens is None:
            return "1", float("inf")
        return expression_string(best_tokens), best_loss
