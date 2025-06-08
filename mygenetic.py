import numpy as np
import random
import operator
import math
from pandas.core.frame import DataFrame

# 定义操作符
OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': lambda x, y: x / y if y != 0 else 1,  # 防止除以零
    #'^2': lambda x: x*x ,  # 防止负数的幂次和溢出
    #'^3': lambda x: x*x*x ,  # 防止负数的幂次和溢出
    'cos': math.cos,
    'sin': math.sin
}


# 定义表达式树节点类
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        try:

            if isinstance(self.value, float):
                return self.value
            elif isinstance(self.value, str) and self.value in OPERATORS:
                if self.value in ['cos', 'sin']:
                    #print(self.value,type(self.value),isinstance(self.value, int),isinstance(self.value, np.int64),self.left,self.right,self.left.evaluate(x))
                    xx=self.left.evaluate(x)
                    vvalue = OPERATORS[self.value](xx)
                    if isinstance(self.left.value, float):
                        self.value = float(vvalue)
                        self.left = None
                        self.right = None
                    return vvalue
                else:
                    vvalue = OPERATORS[self.value](self.left.evaluate(x), self.right.evaluate(x))
                    if isinstance(self.left.value, float) and isinstance(self.right.value, float):
                        self.value = float(vvalue)
                        self.left = None
                        self.right = None
                    return vvalue
            elif isinstance(self.value, int) or isinstance(self.value, np.int64):
                return x[self.value]
        except (KeyError):
            #print(self.value,x.columns.values,type(self.value),isinstance(self.value, int),isinstance(self.value, np.int64),self.left,self.right)
            return 1  # 对异常值返回一个默认值

    def __str__(self):
        if isinstance(self.value, float):
            return str(self.value)
        elif isinstance(self.value, int) or isinstance(self.value, np.int64):
            return f"x{self.value}"
        elif self.value in ['cos', 'sin']:
            return f"{self.value}({self.left})"
        elif self.value in ['^2','^3']:
            return f"({self.left}){self.value}"
        else:
            return f"({self.left} {self.value} {self.right})"

    def __hash__(self):
        return hash((self.left, self.value, self.right))


# 生成随机表达式树
def generate_random_tree(depth, num_features):
    if depth == 0:
        return Node(random.choice(list(num_features) + [random.uniform(-1, 1)]))
    else:
        operator = random.choice(list(OPERATORS.keys()))
        if operator in ['cos', 'sin','^2','^3']:
            return Node(operator, left=generate_random_tree(depth - 1, num_features))
        else:
            return Node(operator, left=generate_random_tree(depth - 1, num_features),
                        right=generate_random_tree(depth - 1, num_features))


# 评估表达式树的适应度
def fitness_function(tree, X, Y):
    predictions = np.array([tree.evaluate(x) for index, x in X.iterrows()])
    mse = np.mean((predictions - Y) ** 2)
    return mse


# 初始化种群
def initialize_population(size, depth, num_features):
    return [generate_random_tree(depth, num_features) for _ in range(size)]


# 选择操作
def select(population, fitnesses, num_parents):
    fitnesses = np.array(fitnesses)
    fitnesses[np.isinf(fitnesses) | np.isnan(fitnesses)] = np.finfo(np.float64).max  # 将无效的适应度值替换为最大浮点值
    weights = [1 / f for f in fitnesses]
    selected_parents = random.choices(population, weights=weights, k=num_parents)
    return selected_parents


# 交叉操作
def crossover(parent1, parent2):
    if random.random() < 0.5:
        return parent1
    else:
        return parent2


# 变异操作
def mutate(tree, mutation_rate, num_features, max_depth):
    if random.random() < mutation_rate:
        return generate_random_tree(max_depth, num_features)
    else:
        if tree.left is not None:
            tree.left = mutate(tree.left, mutation_rate, num_features, max_depth)
        if tree.right is not None:
            tree.right = mutate(tree.right, mutation_rate, num_features, max_depth)
        return tree


# 遗传算法主循环
def genetic_algorithm(X, Y, init_pop, pop_size, generations, mutation_rate, loss_threshold, max_depth):
    # print(X.shape)
    num_features = X.columns.values
    if init_pop == []:
        population = initialize_population(int(pop_size), max_depth, num_features)
    else:
        population = init_pop + initialize_population(int(pop_size / 2), max_depth, num_features)
    last_fitnesses=-1
    for generation in range(generations):
        fitnesses = [fitness_function(ind, X, Y) for ind in population]

        if min(fitnesses) < loss_threshold:
            break

        if last_fitnesses==-1:
            continue
        if loss_threshold/10 >= fitnesses-last_fitnesses >= -loss_threshold/10:
            break
        last_fitnesses=fitnesses
        new_population = []
        for _ in range(pop_size):
            parents = select(population, fitnesses, 2)
            offspring = crossover(parents[0], parents[1])
            offspring = mutate(offspring, mutation_rate, num_features, max_depth)
            new_population.append(offspring)

        population = new_population
        best_individual = min(population, key=lambda ind: fitness_function(ind, X, Y))

        # print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    best_individual = min(population, key=lambda ind: fitness_function(ind, X, Y))
    best_fitness = fitness_function(best_individual, X, Y)
    return population[:int(pop_size / 2)], best_individual, best_fitness


# 示例用法
def fit_model(data, target_column, input_columns, init_pop, loss_threshold=0.01, max_depth=3):
    X = data[:, input_columns]
    Y = data[:, target_column]
    population_size = 100
    num_generations = 20
    mutation_rate = 0.1
    data_X = {}
    for i in input_columns:
        data_X[i] = [id[i] for id in data]
    data_X = DataFrame(data_X)
    pop_now, best_solution, best_fitness = genetic_algorithm(data_X, Y, init_pop, population_size, num_generations,
                                                             mutation_rate, loss_threshold, max_depth)

    return pop_now, best_solution, best_fitness


if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    n, m = 100, 5
    data = np.random.rand(n, m)
    target_column = 4
    input_columns = [0, 2, 3]
    loss_threshold = 0.01
    max_depth = 3  # 表达式树的最大深度

    pop_now, best_solution, best_fitness = fit_model(data, target_column, input_columns, [], loss_threshold, max_depth)
    print(f"Best solution found: {best_solution}")
    print(best_fitness)
