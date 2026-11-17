# EFDiscover

**Expression Functional Dependency（EFD，表达式函数依赖）** 挖掘工具：在含噪声的数值数据上，
同时发现"哪些属性决定哪个属性"以及"用什么样的表达式决定"。

论文：*EFDiscover: Prior-Free Discovery of Inter-Attribute Expression Functional
Dependencies in Noisy Data*（ICDE 2024）

传统函数依赖 `X → Y` 只说明"X 决定 Y"，却说不清 *怎么* 决定；EFD 把它写成

```
X --(f, ε)--> Y ,   f(X) = Y ,   loss(f) = Σ (y_i - f(x_i))²  ≤  ε
```

即：存在一个符号表达式 `f`（运算符取自 `L = {+, -, *, /, sin, cos}`），
使得在整份数据上拟合误差不超过 `ε`。同一个 `X → Y` 若用任何 `f` 都无法把误差压到 `ε` 以下，
就称为 **nonEFD**；把 nonEFD 全部维护住，剩下的空位就是**最小 EFD**。

---

## 关于本仓库的代码状态

仓库最初的版本（2025-06 上传）无法运行：`main.py` 依赖的 `SymRegression1.py` / `SymRegression2.py`
缺失，`mygenetic.py` 里还有若干必然抛异常的写法。

当前版本按论文 §3（EFD 定义、EBTree、符号回归）与 §6.4.1（符号回归消融）补齐并修好，
**已能在 `glass.data` 上端到端复现三个变体**。具体改了哪些地方见文末
[与论文的一致性与修复清单](#与论文的一致性与修复清单)——其中"重建"的部分（两个消融变体的具体实现）
是依据论文对 `EFDiscover-Sym1/Sym2` 的描述推定的，如果你手上有当年的原始实现，建议对照一下。

## 方法概览

1. **属性排序**：以属性 `Y` 为 RHS，其余属性按与 `Y` 的相关性升序（`1-|corr|` 从小到大）排列，
   相关性高的先插入——即论文的第二个剪枝策略"属性相关性排序"。
2. **EBTree 维护 nonEFD**：扩展二叉树，每一层对应"取/不取某属性"。节点右子树为空
   ⇒ 该属性集合已是 EFD；若 LHS 及其所有扩展都超过 `ε` ⇒ 整棵子树是 nonEFD
   （由 nonEFD 的单调性保证，见论文中的 Lemma）。
3. **子树剪枝**：判定一个节点后，把"全 EFD / 全 nonEFD"的整棵子树剪掉，不再逐个回归。
4. **符号回归**：每个待判定的属性集合跑一次遗传编程符号回归（`mygenetic.py`），
   得到最优表达式与 MSE，与 `φ` 比较决定 EFD / nonEFD。
5. **两个回归侧优化**（对应论文消融）：
   * **增量发现**：把上一节点的种群沿 EBTree 复用，省掉随机重启；
   * **损失变化率早停**：相邻两代最优适应度变化小于 `loss_threshold/10` 就认为收敛退出。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 主程序：命令行、EFD 挖掘流程、结果输出、`--grid` 规模扫描 |
| `EBTree.py` | 扩展二叉树，维护 nonEFD 并做整棵子树剪枝（`insert` / `GetEFD`） |
| `SymRegression.py` | 符号回归封装（数据读取、回归调用、结果缓存），**完整版** |
| `SymRegression1.py` | 消融变体 **nonSymPop**：关闭增量发现 |
| `SymRegression2.py` | 消融变体 **nonSymLoss**：关闭损失变化率早停 |
| `mygenetic.py` | 遗传编程符号回归（表达式树、选择/交叉/变异、早停） |
| `glass.data` | 示例数据（UCI Glass，214 行，首列为样本编号，**无表头**） |
| `SymbolicRNN.py` | 论文 §3.3 的 RNN 表达式构造原型（需 PyTorch，未接入主流程） |
| `convert.py` / `corrsplit.py` / `draw.py` | 辅助脚本：表达式串转符号序列、属性相关性分组、作图 |

## 环境

```bash
pip install -r requirements.txt
```

Python ≥ 3.8，主流程只依赖 `numpy` 与 `pandas`；`SymbolicRNN.py` 额外需要 PyTorch。

## 快速开始

```bash
python main.py                                  # glass.data 前 60 行、随机 8 个属性
python main.py --rows 100 --cols -1             # 用全部 214 行、全部属性
python main.py --variant all                    # 三个变体一起跑，做消融对比
python main.py --phi 0.01 --top 10              # 收紧损失阈值
```

`glass.data` 首列是样本编号，建议加 `--drop-id` 去掉（否则编号会作为属性参与挖掘）。

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--csv` | 本目录 `glass.data` | 数据集路径（CSV） |
| `--rows` | 60 | 只取前多少行，`-1` 为全部 |
| `--cols` | 8 | 随机取多少列属性，`-1` 为全部 |
| `--phi` | 0.02 | EFD 损失阈值 `ε` |
| `--variant` | `0` | `0`=完整，`1`=无增量发现，`2`=无损失变化率早停，`all`=三个都跑 |
| `--drop-id` | 关闭 | 无表头数据去掉第一列 |
| `--seed` | 0 | 随机种子（列抽样与遗传算法都受它控制） |
| `--max-depth` | 3 | 表达式树最大深度 |
| `--population` | 100 | 种群大小 |
| `--generations` | 20 | 遗传代数 |
| `--gp-loss` | `--phi/10` | 回归内部早停阈值，适应度低于它就不再进化 |
| `--top` | 20 | 最多打印多少条规则 |
| `--grid` | 关闭 | 复现论文的规模灵敏度扫描 |
| `--quiet` | 关闭 | 只打印统计，不打印表达式 |

### 输出示例

`python main.py --variant all --top 2`（glass.data，前 60 行，8 个属性，`φ=0.02`，seed 0）：

```
变体：0 / EFDiscover
EFD 数量：22    符号回归次数：53    平均损失：0.003778    耗时：3.85 s
x6 -> x1   [9]   loss=0.008206
    f = (0.9378 / (x6 + x6))
x7 -> x1   [9]   loss=0.008501
    f = (1.182 * (0.05927 + (0.5803 * x7)))
...（另有 20 条，用 --top 调整）

变体：1 / EFDiscover-Sym1 (nonSymPop)
EFD 数量：27    符号回归次数：72    平均损失：0.002798    耗时：5.64 s

变体：2 / EFDiscover-Sym2 (nonSymLoss)
EFD 数量：23    符号回归次数：61    平均损失：0.003389    耗时：12.27 s
```

`x6 -> x1` 里的 `[9]` 是该列在原始数据里的列名：`glass.data` 无表头，去掉首列编号后
列名就是 `0..10`，`x1` 指本次抽样后的第 2 个属性。

### 本机实测的消融对比

同一配置（glass.data 前 60 行、8 属性、`φ=0.02`、`--gp-loss=0.002`、种群 100、20 代、seed 0）：

| 变体 | EFD 数 | 符号回归次数 | 平均损失 | 耗时 |
| --- | --- | --- | --- | --- |
| `EFDiscover`（完整） | 22 | **53** | 0.003778 | **3.85 s** |
| `-Sym1`（nonSymPop，无增量发现） | 27 | 72 | 0.002798 | 5.64 s |
| `-Sym2`（nonSymLoss，无早停） | 23 | 61 | 0.003389 | 12.27 s |

方向和论文 §6.4.1 一致：

* 去掉**增量发现**后，同一个 EBTree 上要跑的符号回归从 53 次涨到 72 次，耗时随之上升——
  论文说的"增量发现显著省时"来自于此；
* 去掉**损失变化率早停**后耗时最久（12.27 s），但搜索更充分，平均损失更低；
* 三者的 EFD 数量差异不大，符合论文"两个策略基本不改变发现数量"的结论。

绝对数值只是这个小数据集上一次运行的结果，换 `--seed` 会波动。

## 数据格式

* 逗号分隔的数值表；**有表头**时按表头读列名，**首行全是数字**时按无表头处理（如 `glass.data`）；
* 非数值列会被自动丢弃；
* 无表头数据的列名记为 `x0, x1, ...`，与输出中的 `x3 -> x7` 一致；
* `--drop-id` 用于去掉无表头数据的首列编号。

## 复现论文的规模灵敏度实验

```bash
python main.py --grid --csv glass.data --drop-id
```

按行数 100/200/…/2000、列数 5…11 分别扫描三个变体，写出 `row_sweep.csv` 与 `col_sweep.csv`。
注意 `glass.data` 只有 214 行，行数超过 214 时等价于全量，`--grid` 更适合放在大数据集上跑。

## 与论文的一致性与修复清单

**按论文实现的**

* EFD / nonEFD / 最小 EFD 的定义与判定（`loss ≤ φ` 即 EFD）；
* EBTree 维护 nonEFD、属性按相关性排序插入、整棵 EFD/nonEFD 子树剪枝；
* 符号集合 `L = {+, -, *, /, sin, cos}`；
* 符号回归的两个优化及其消融变体（`SymRegression1`/`SymRegression2`，对应论文的
  `EFDiscover-Sym1`/`-Sym2`，即实验表格中的 `nonSymPop`/`nonSymLoss` 两列）。

**重建的部分（原仓库缺失）**

* `SymRegression1.py`、`SymRegression2.py`：按论文对两个消融变体的描述，用"关掉增量发现"、
  "关掉损失变化率早停"两个开关实现；
* `main.py` 改写为命令行入口，保留 `EFDiscover` / `EFDiscover1` / `EFDiscover2` 三个函数接口。

**修掉的 bug（原版会直接崩溃或结果不可信）**

1. `mygenetic.select()`：对 Python `list` 做 `fitnesses[mask] = ...` 赋值 → `TypeError`；
2. `mygenetic` 的早停：`fitnesses - last_fitnesses` 是列表相减 → `TypeError`，
   且首代 `continue` 导致第一代根本不进化；
3. `crossover()` 只是随机返回一个父代（等价于复制），搜索能力太弱——连 `c = a + b`
   这种可精确表达的规律都拟合不出来（损失停在 0.0087）→ 换成真正的子树交叉 + 精英保留，
   同一例子现在能给出 `f = (x0 + x1)`、损失 2e-9；
4. `mutate()` 在深层节点整棵重生，表达式深度失控（实测能打印出几十层嵌套）→ 加深度上限；
5. EFD 判定与结果打印各跑一次回归，损失可能对不上（曾出现"判定通过、打印损失 0.24 > φ"）
   → 按 `(RHS, LHS)` 缓存回归结果。

**与论文的已知差异**

* 论文把 `loss` 写成求和 `Σ(y-f(x))²`，代码里用的是**均方误差**（除以样本数），
  `φ` 的取值需按 MSE 的量级来定；
* 稳定常数项、表达式化简（如用 sympy 化简）未实现；
* `SymbolicRNN.py` 是论文 §3.3 的 RNN 路径原型，需要 PyTorch，当前主流程用的是遗传编程。

## 引用

```bibtex
@inproceedings{efdiscover2024,
  title     = {EFDiscover: Prior-Free Discovery of Inter-Attribute Expression Functional Dependencies in Noisy Data},
  booktitle = {IEEE International Conference on Data Engineering (ICDE)},
  year      = {2024}
}
```

## 许可

仓库未附带开源许可证文件，如需复用请先与作者联系。
