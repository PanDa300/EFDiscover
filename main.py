from EBTree import EBTree
from SymRegression import SymRegression
import numpy as np
import time
import pandas as pd
from SymRegression import SymRegression



def EFDiscover(filename, phi, row=-1, col=-1):
    Data = SymRegression(filename, row, col)
    data = Data.df
    x = data.corr()
    x = np.array(x)
    res = []
    sum_best_fitness = 0
    sum_cnt = 0
    for i in range(len(x)):
        print(i,len(x))
        id = []
        person = []
        for j in range(len(x)):
            if i == j:
                continue
            id.append(j)
            person.append(1 - abs(x[i][j]))
        if np.isnan(person[0]):
            continue
        id.sort()
        result_list = [i for _, i in sorted(zip(person, id))]
        EBTree_i = EBTree(i, Data.Calc_Regression, phi)
        for j in range(len(result_list)):
            EBTree_i.insert(result_list[j], result_list[j + 1:])
        LHS = EBTree_i.GetEFD()
        for j in LHS:
            pop_now, best_fitness = Data.Calc_Regression(j, i, [])
            res.append([j, i, Data.Get_Regression(j, i, []),best_fitness])
            sum_best_fitness += best_fitness
            sum_cnt += 1
    loss = 1
    if sum_cnt != 0:
        loss = sum_best_fitness / sum_cnt
    return res, Data.cnt, loss




def work_one(file_name):
    error_list = 0.5

    j = -1
    i = -1

    start = time.time()
    result, cnt, loss = EFDiscover(file_name, error_list, col=i, row=j)
    end = time.time()
    print(file_name, j, i, 11)
    print(len(result), end - start, cnt, loss)
    for i in result:
        print(i[0], i[1])
        print(f"Best solution found: {i[2]}")
        print('%.2f'%(i[3]))


if __name__ == "__main__":
    # work("abalone.data")
    # work("Rice_Cammeo_Osmancik.arff")
    # work_one("averaged_data.csv")
    # work("A.csv")
    work_one("glass.data")
