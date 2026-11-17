# -*- coding: utf-8 -*-
"""遗传编程符号回归

用表达式树（+、-、*、/、sin、cos 与常数）拟合 ``Y ≈ f(X)``，返回最优表达式与适应度（MSE）。

论文（EFDiscover, ICDE 2024）在符号回归上有两个优化，这里用开关表达：

* ``reuse_pop``：**增量发现**。上一个节点（EBTree 兄弟/父子节点）留下的种群作为
  下一轮回归的初始种群的一部分，沿着 EBTree 逐层复用，减少随机重启的浪费。
* ``early_stop``：**损失变化率早停**。相邻两代最优适应度的变化幅度低于
  ``loss_threshold / 10`` 时认为已收敛，提前结束迭代。

对应的消融开关见 ``SymRegression`` / ``SymRegression1`` / ``SymRegression2``。
"""
import copy
import math
import operator
import random

import numpy as np
import pandas as pd

# 论文使用的符号集合 L = {+ - * / sin cos}
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda x, y: x / y if y != 0 else 1.0,   # 除零保护
    'cos': math.cos,
    'sin': math.sin,
}
UNARY = ('cos', 'sin')


class Node:
    """表达式树节点：``value`` 为运算符、属性下标或常数"""

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        try:
            if isinstance(self.value, float):
                return self.value
            if isinstance(self.value, str) and self.value in OPERATORS:
                if self.value in UNARY:
                    vvalue = OPERATORS[self.value](self.left.evaluate(x))
                    # 子树全为常数时折叠成常数，避免重复计算
                    if isinstance(self.left.value, float):
                        self.value = float(vvalue)
                        self.left = None
                        self.right = None
                    return vvalue
                vvalue = OPERATORS[self.value](self.left.evaluate(x), self.right.evaluate(x))
                if isinstance(self.left.value, float) and isinstance(self.right.value, float):
                    self.value = float(vvalue)
                    self.left = None
                    self.right = None
                return vvalue
            if isinstance(self.value, (int, np.integer)):
                return x[self.value]
        except (KeyError, TypeError, ValueError):
            return 1.0     # 越界或非法取值时回退为常数，避免整轮回归崩溃
        return 1.0

    def __str__(self):
        if isinstance(self.value, float):
            return f"{self.value:.4g}"
        if isinstance(self.value, (int, np.integer)):
            return f"x{self.value}"
        if self.value in UNARY:
            return f"{self.value}({self.left})"
        return f"({self.left} {self.value} {self.right})"

    def __hash__(self):
        return hash((self.left, self.value, self.right))


def generate_random_tree(depth, num_features):
    """生成随机表达式树（最大深度 ``depth``）"""
    if depth <= 0:
        return Node(random.choice(list(num_features) + [random.uniform(-1, 1)]))
    op = random.choice(list(OPERATORS.keys()))
    if op in UNARY:
        return Node(op, left=generate_random_tree(depth - 1, num_features))
    return Node(op,
                left=generate_random_tree(depth - 1, num_features),
                right=generate_random_tree(depth - 1, num_features))


def initialize_population(size, depth, num_features):
    return [generate_random_tree(depth, num_features) for _ in range(max(int(size), 1))]


def _as_rows(X):
    """把 DataFrame / 行序列统一成逐行可索引的序列（避免反复 iterrows）"""
    if isinstance(X, pd.DataFrame):
        return [row for _, row in X.iterrows()]
    return list(X)


def fitness_function(tree, X, Y):
    """适应度：预测值与真实值的均方误差（MSE）"""
    rows = _as_rows(X)
    preds = np.fromiter((tree.evaluate(row) for row in rows), dtype=float, count=len(rows))
    return float(np.mean((preds - Y) ** 2))


def select(population, fitnesses, num_parents):
    """轮盘赌选择：适应度越小权重越大"""
    fit = np.asarray(fitnesses, dtype=float)
    fit = np.where(np.isfinite(fit), fit, np.finfo(np.float64).max)
    fit = np.maximum(np.abs(fit), 1e-12)
    weights = (1.0 / fit).tolist()
    return random.choices(population, weights=weights, k=num_parents)


def _iter_nodes(tree):
    """遍历表达式树的所有节点（含根）"""
    yield tree
    if tree.left is not None:
        yield from _iter_nodes(tree.left)
    if tree.right is not None:
        yield from _iter_nodes(tree.right)


def crossover(parent1, parent2):
    """子树交叉：把 parent2 的一棵随机子树换到 parent1 的随机位置上

    原实现直接返回某一个父代（等价于复制 + 变异），搜索能力太弱——像 ``a + b``
    这种可精确表达的规律都拟合不出来。这里换成标准的子树交叉，并整体深拷贝，
    避免多个个体共享同一棵树后互相污染。
    """
    child = copy.deepcopy(parent1)
    donor = copy.deepcopy(parent2)
    slot = random.choice(list(_iter_nodes(child)))
    graft = random.choice(list(_iter_nodes(donor)))
    slot.value, slot.left, slot.right = graft.value, graft.left, graft.right
    return child


def mutate(tree, mutation_rate, num_features, max_depth, depth=0):
    """变异：按概率整棵子树重生，否则递归向下

    ``depth`` 用于保证整棵表达式树不超过 ``max_depth``（论文要求表达式尽量简单）。
    """
    if depth >= max_depth:
        return Node(random.choice(list(num_features) + [random.uniform(-1, 1)]))
    if random.random() < mutation_rate:
        return generate_random_tree(max_depth - depth, num_features)
    if tree.left is not None:
        tree.left = mutate(tree.left, mutation_rate, num_features, max_depth, depth + 1)
    if tree.right is not None:
        tree.right = mutate(tree.right, mutation_rate, num_features, max_depth, depth + 1)
    return tree


def genetic_algorithm(X, Y, init_pop=None, pop_size=100, generations=20,
                      mutation_rate=0.1, loss_threshold=0.01, max_depth=3,
                      reuse_pop=True, early_stop=True, verbose=False):
    """遗传算法主循环

    返回 ``(carry_pop, best_individual, best_fitness)``：``carry_pop`` 是保留下来的一半
    种群，供下一个节点做增量发现；``reuse_pop=False`` 时恒为空列表。
    """
    num_features = list(X.columns.values)
    rows = _as_rows(X)

    if reuse_pop and init_pop:
        population = list(init_pop) + initialize_population(pop_size // 2, max_depth, num_features)
    else:
        population = initialize_population(pop_size, max_depth, num_features)

    best_individual = min(population, key=lambda ind: fitness_function(ind, rows, Y))
    best_fitness = fitness_function(best_individual, rows, Y)
    last_best = None

    for generation in range(generations):
        fitnesses = [fitness_function(ind, rows, Y) for ind in population]
        gen_best = float(min(fitnesses))
        if gen_best < best_fitness:
            best_fitness = gen_best
            best_individual = population[int(np.argmin(fitnesses))]

        if best_fitness < loss_threshold:            # 已足够好，提前结束
            break

        if early_stop and last_best is not None:     # 适应度变化率过小 → 已收敛
            if abs(gen_best - last_best) <= loss_threshold / 10:
                break
        last_best = gen_best

        new_population = [copy.deepcopy(best_individual)]      # 精英保留，保证不退化
        for _ in range(max(pop_size - 1, 1)):
            parents = select(population, fitnesses, 2)
            offspring = crossover(parents[0], parents[1])
            new_population.append(mutate(offspring, mutation_rate, num_features, max_depth))
        population = new_population

        if verbose:
            print(f"    generation {generation + 1}/{generations}: best_MSE={best_fitness:.6g}")

    carry = population[: max(pop_size // 2, 1)] if reuse_pop else []
    return carry, best_individual, best_fitness


def fit_model(data, target_column, input_columns, init_pop=None, loss_threshold=0.01,
              max_depth=3, reuse_pop=True, early_stop=True,
              population_size=100, generations=20, mutation_rate=0.1, verbose=False):
    """对外接口：``data`` 为 numpy 矩阵，``target_column`` / ``input_columns`` 为列下标"""
    input_columns = list(input_columns)
    X = data[:, input_columns]
    Y = data[:, target_column]
    data_X = pd.DataFrame({i: X[:, k] for k, i in enumerate(input_columns)})
    return genetic_algorithm(data_X, Y,
                             init_pop=init_pop or [],
                             pop_size=population_size,
                             generations=generations,
                             mutation_rate=mutation_rate,
                             loss_threshold=loss_threshold,
                             max_depth=max_depth,
                             reuse_pop=reuse_pop,
                             early_stop=early_stop,
                             verbose=verbose)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    n, m = 100, 5
    demo = np.random.rand(n, m)
    carry, best, fit = fit_model(demo, 4, [0, 2, 3], [], 0.01, 3)
    print(f"Best solution found: {best}")
    print(f"MSE: {fit}")
