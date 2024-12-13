

import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
import mygenetic
import time
def NRegression(data, input_columns, target_column):
    # 提取输入矩阵X和目标向量Y
    X = data[:, input_columns]
    Y = data[:, target_column]
    #print(X.shape,Y.shape)
    # 添加偏置项
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # 计算线性回归的参数 (theta)
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # 计算预测值
    Y_pred = X @ theta

    # 计算均方误差 (MSE)
    mse = np.mean((Y - Y_pred) ** 2)

    # 构建回归方程的表达式
    equation_terms = [f"{theta[0]:.4f}"] + [f"{theta[i+1]:.4f} * x{i+1}" for i in range(len(input_columns))]
    equation = " + ".join(equation_terms)

    return equation, mse

class SymRegression:
    def __init__(self,file_name,row,col):
        self.df=pd.read_csv(file_name)
        self.df=self.df.select_dtypes(include=["float64", "int64"])
        if row != -1:
            self.df = self.df.head(row)
        if col != -1:
            self.df = self.df.sample(n=col, axis='columns')
        self.cnt=0

    def Calc_Regression(self,LHS,RHS,pop,loss_threshold=0.1):
        data=self.df.values
        # data = sc.fit_transform(data)
        #self.cnt+=1
        if LHS==[]:
            return pop,1
        '''best_solution, best_fitness=NRegression(data, LHS, RHS)
        if best_fitness<loss_threshold:
            return pop,best_fitness'''
        #print(LHS,RHS,len(data[0]))
        pop_now,best_solution,best_fitness =mygenetic.fit_model(data, RHS,LHS, pop,loss_threshold, max_depth=3)
        self.cnt+=1
        #print(best_solution,best_fitness)
        return pop_now,best_fitness


    def Get_Regression(self,LHS,RHS,pop,loss_threshold=0.1):
        data=self.df.values
        # data = sc.fit_transform(data)
        pop_now,best_solution,best_fitness =mygenetic.fit_model(data, RHS,LHS, pop,loss_threshold, max_depth=3)
        return best_solution


if __name__=="__main__":
    # 示例用法
    np.random.seed(42)
    n, m = 100, 5
    data = np.random.rand(n, m)

    input_columns = [0, 1, 2, 3]  # 输入属性的列号
    target_column = 4  # 目标属性的列号

    equation, mse = NRegression(data, input_columns, target_column)
    print(f"Linear regression equation: {equation}")
    print(f"Mean Squared Error: {mse}")

    '''Score_final=[]
    import time
    for i in [500,1000,1500,2000,2500,3000]:
        Start=time.time()
        df = df.head(i)
        data = df.iloc[:, 3:]
        target = df.iloc[:, 1]
        print(type(data),type(target))
        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=2023)
        est_gp = SymbolicRegressor(population_size=5000,
                                   generations=20, stopping_criteria=0.01,
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   max_samples=0.9, verbose=1,
                                   parsimony_coefficient=0.01, random_state=0)
        est_gp.fit(train_x, train_y)
        End=time.time()
        Score_final.append(End-Start)
        print( est_gp._program)
    print(est_gp.score(valid_x, valid_y))  # 0.5967867578111098
    print(
        est_gp._program)  # sub(div(X5, 0.022), mul(add(mul(X10, X4), sub(X5, 0.502)), sub(div(X10, X12), add(div(mul(add(X5, X5), mul(X12, X10)), sub(0.479, 0.502)), sub(div(X11, X12), sub(0.479, 0.502))))))
    print(Score_final)
    from pydotplus.graphviz import graph_from_dot_data
    
    dot_data = est_gp._program.export_graphviz()
    #dot_data = dot_data.replace('\n', '')
    graph = graph_from_dot_data(dot_data)  # Create graph from dot data
    
    graph.write('D:/决策树.png')  # Write graphto PNG image
    
    for i in range(1,len(Score_final)):
        Score_final[i]=Score_final[i-1]+Score_final[i]*i
    
    import matplotlib.pyplot as plt
    
    x_axis_data = [500,1000,1500,2000,2500,3000]  # x
    y_axis_data = Score_final  # y
    
    plt.plot(x_axis_data, y_axis_data, linewidth=1, label='time')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
    
    plt.legend()  # 显示上面的label
    plt.xlabel('Cnt')  # x_label
    plt.ylabel('Time')  # y_label
    
    # plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.show()'''
