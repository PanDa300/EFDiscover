# EFDiscover

在含噪声的数值数据上挖掘**表达式函数依赖**（Expression Functional Dependency, EFD）：
同时给出"哪些属性决定哪个属性"和"用什么样的表达式决定"，无需预先指定函数形式。

EFD 的形式是

```
X --(f, ε)--> Y ,   f(X) = Y ,   拟合误差 ≤ ε
```

其中 `f` 由运算符集合 `L = {+, -, *, /, sin, cos}` 与常数组合而成。同一组
`X → Y` 若用任何 `f` 都无法把误差压到 `ε` 以下，就称为 **nonEFD**；把 nonEFD 都维护住，
剩下没被覆盖的属性集合就是最小的 EFD。

---

## 方法概览

1. **属性排序**：以属性 `Y` 为 RHS，其余属性按与 `Y` 的相关性升序（`1-|corr|` 从小到大）
   排列，相关性高的先参与搜索。
2. **EBTree 维护 nonEFD**：扩展二叉树，每一层对应"取/不取某属性"。节点右子树为空
   ⇒ 该属性集合已是 EFD；若 LHS 及其所有扩展都超过 `ε` ⇒ 整棵子树是 nonEFD
   （nonEFD 对属性扩展单调，因此可以整棵剪掉）。
3. **子树剪枝**：判定一个节点后，把"全 EFD / 全 nonEFD"的整棵子树剪掉，不再逐个回归。
4. **符号回归**：每个待判定的属性集合跑一次符号回归，得到最优表达式与 MSE，
   与 `φ` 比较决定是 EFD 还是 nonEFD。两个后端可选：
   * `--backend gp`：遗传编程（`mygenetic.py`）；
   * `--backend rnn`：编码器-解码器（`SymbolicRNN.py`，需 PyTorch）。编码器把"每个属性与
     RHS 的关系"编码成解码器初始隐状态，解码器按前缀序列逐步生成表达式；网络先在整份
     数据集上预训练一次，再对每个节点微调，最后用最小二乘把表达式里的常数重解一遍。
   * 两个后端生成的表达式都至少包含一个 LHS 属性——纯常数拟合不算"被属性决定"，不判为 EFD。
5. **两个可选优化**：
   * **增量发现**：把上一节点的种群沿 EBTree 复用，省掉随机重启（RNN 后端是复用已训练的 LSTM）；
   * **损失变化率早停**：相邻两代最优适应度变化小于 `loss_threshold/10` 就认为收敛退出。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `main.py` | 主程序：命令行、EFD 挖掘流程、结果输出、按行数/列数扫描 |
| `EBTree.py` | 扩展二叉树，维护 nonEFD 并做整棵子树剪枝（`insert` / `GetEFD`） |
| `SymRegression.py` | 符号回归封装（数据读取、回归调用、结果缓存），完整版 |
| `SymRegression1.py` | 变体 nonSymPop：关闭增量发现 |
| `SymRegression2.py` | 变体 nonSymLoss：关闭损失变化率早停 |
| `mygenetic.py` | 遗传编程符号回归（表达式树、选择、交叉、变异、早停），`--backend gp` 使用 |
| `glass.data` | 示例数据（UCI Glass，214 行，首列为样本编号，无表头） |
| `SymbolicRNN.py` | RNN 符号回归：编码器-解码器自回归生成 + 数据集级预训练 + 常数最小二乘精修，`--backend rnn` 使用 |
| `convert.py` / `corrsplit.py` / `draw.py` | 辅助脚本：表达式串转符号序列、属性相关性分组、作图 |

## 环境

```bash
pip install -r requirements.txt
```

Python ≥ 3.8，默认（`--backend gp`）只依赖 `numpy` 与 `pandas`；
`--backend rnn` 额外需要 PyTorch。

## 数据

* 逗号分隔的数值表；首行能整体解析为数字时按**无表头**读（如 `glass.data`），否则按表头读；
* 非数值列会被自动丢弃；无表头数据的列名记为 `x0, x1, ...`，与输出里的 `x3 -> x7` 对应；
* `glass.data` 首列是样本编号，用 `--drop-id` 去掉，否则编号会作为属性参与挖掘。

## 运行

```bash
python main.py                                  # glass.data 前 60 行、随机 8 个属性
python main.py --drop-id                        # 去掉样本编号列
python main.py --rows 100 --cols -1 --drop-id   # 用全部 214 行、全部属性
python main.py --variant all                    # 三个变体一起跑
python main.py --backend rnn                    # 符号回归改用 LSTM 生成
python main.py --phi 0.01 --top 10              # 收紧损失阈值
```

### 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--csv` | 本目录 `glass.data` | 数据集路径（CSV） |
| `--rows` | 60 | 只取前多少行，`-1` 为全部 |
| `--cols` | 8 | 随机取多少列属性，`-1` 为全部 |
| `--phi` | 0.02 | EFD 损失阈值 `ε` |
| `--variant` | `0` | `0`=完整流程，`1`=不做增量发现，`2`=不做损失变化率早停，`all`=三个都跑 |
| `--backend` | `gp` | 符号回归后端：`gp`=遗传编程，`rnn`=编码器-解码器自回归生成（需 PyTorch） |
| `--no-pretrain` | 关闭 | rnn 后端跳过数据集级预训练，直接逐节点训练 |
| `--pretrain-tasks` | 20 | rnn 后端预训练采样的任务数 |
| `--drop-id` | 关闭 | 无表头数据去掉第一列 |
| `--seed` | 0 | 随机种子（列抽样与遗传算法都受它控制） |
| `--max-depth` | 3 | 表达式树最大深度 |
| `--population` | 100 | 种群大小 |
| `--generations` | 20 | 遗传代数 |
| `--gp-loss` | `--phi/10` | 回归内部早停阈值，适应度低于它就不再进化 |
| `--top` | 20 | 最多打印多少条规则 |
| `--grid` | 关闭 | 按行数与列数扫描，写出 `row_sweep.csv` / `col_sweep.csv` |
| `--quiet` | 关闭 | 只打印统计，不打印表达式 |

### 输出示例

`python main.py --variant all --top 2`（glass.data 前 60 行、8 个属性、`φ=0.02`、seed 0）：

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

`x6 -> x1` 里的 `[9]` 是该列在原始数据中的列名：`glass.data` 无表头，去掉首列编号后
列名就是 `0..10`，`x1` 指本次抽样后的第 2 个属性。
