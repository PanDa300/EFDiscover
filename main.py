from EBTree import EBTree
from SymRegression import SymRegression
import numpy as np
import time
import pandas as pd
from SymRegression1 import SymRegression1
from SymRegression2 import SymRegression2


def EFDiscover(filename, phi, row=-1, col=-1):
    Data = SymRegression(filename, row, col)
    data = Data.df
    x = data.corr()
    x = np.array(x)
    res = []
    sum_best_fitness = 0
    sum_cnt = 0
    for i in range(len(x)):
        # print(i,len(x))
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
            res.append([j, i, Data.Get_Regression(j, i, [])])
            pop_now, best_fitness = Data.Calc_Regression(j, i, [])
            sum_best_fitness += best_fitness
            sum_cnt += 1
    loss = 1
    if sum_cnt != 0:
        loss = sum_best_fitness / sum_cnt
    return res, Data.cnt, loss

def EFDiscover1(filename, phi, row=-1, col=-1):
    Data = SymRegression1(filename, row, col)
    data = Data.df
    x = data.corr()
    x = np.array(x)
    res = []
    sum_best_fitness = 0
    sum_cnt = 0
    for i in range(len(x)):
        # print(i,len(x))
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
            res.append([j, i, Data.Get_Regression(j, i, [])])
            pop_now, best_fitness = Data.Calc_Regression(j, i, [])
            sum_best_fitness += best_fitness
            sum_cnt += 1
    loss = 1
    if sum_cnt != 0:
        loss = sum_best_fitness / sum_cnt
    return res, Data.cnt, loss

def EFDiscover2(filename, phi, row=-1, col=-1):
    Data = SymRegression2(filename, row, col)
    data = Data.df
    x = data.corr()
    x = np.array(x)
    res = []
    sum_best_fitness = 0
    sum_cnt = 0
    for i in range(len(x)):
        # print(i,len(x))
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
            res.append([j, i, Data.Get_Regression(j, i, [])])
            pop_now, best_fitness = Data.Calc_Regression(j, i, [])
            sum_best_fitness += best_fitness
            sum_cnt += 1
    loss = 1
    if sum_cnt != 0:
        loss = sum_best_fitness / sum_cnt
    return res, Data.cnt, loss


def work(file_name):

    row_result_fin = []
    row_time_fin = []
    row_cnt_fin = []
    row_loss_fin = []
    col_result_fin = []
    col_time_fin = []
    col_cnt_fin = []
    col_loss_fin = []
    error_list = 0.01
    row_list = [100,200,300,500,700,1000,1200,1500,1700,2000]       
    col_list = list(range(5, 12))

    j=error_list

    row_result = []
    row_time = []
    row_cnt = []
    row_loss = []
    col_result = []
    col_time = []
    col_cnt = []
    col_loss = []
    for i in row_list:
        start = time.time()
        result, cnt, loss = EFDiscover2(file_name, j, row=i,col=-1)
        end = time.time()
        print(file_name, j, i, 7)
        print(len(result), end - start, cnt, loss)
        row_result.append(len(result))
        row_time.append(end - start)
        row_cnt.append(cnt)
        row_loss.append(loss)
    for i in col_list:
        start = time.time()
        result, cnt, loss = EFDiscover2(file_name, j, row=200,col=i)
        end = time.time()
        print(file_name, j, 2000, i)
        print(len(result), end - start, cnt, loss)
        col_result.append(len(result))
        col_time.append(end - start)
        col_cnt.append(cnt)
        col_loss.append(loss)
    row_result_fin.append(row_result)
    row_time_fin.append(row_time)
    row_cnt_fin.append(row_cnt)
    row_loss_fin.append(row_loss)
    col_result_fin.append(col_result)
    col_time_fin.append(col_time)
    col_cnt_fin.append(col_cnt)
    col_loss_fin.append(col_loss)

    row_result = []
    row_time = []
    row_cnt = []
    row_loss = []
    col_result = []
    col_time = []
    col_cnt = []
    col_loss = []
    for i in row_list:
        start = time.time()
        result, cnt, loss = EFDiscover1(file_name, j, row=i,col=-1)
        end = time.time()
        print(file_name, j, i, 11)
        print(len(result), end - start, cnt, loss)
        row_result.append(len(result))
        row_time.append(end - start)
        row_cnt.append(cnt)
        row_loss.append(loss)
    for i in col_list:
        start = time.time()
        result, cnt, loss = EFDiscover1(file_name, j, row=200,col=i)
        end = time.time()
        print(file_name, j, 200, i)
        print(len(result), end - start, cnt, loss)
        col_result.append(len(result))
        col_time.append(end - start)
        col_cnt.append(cnt)
        col_loss.append(loss)
    row_result_fin.append(row_result)
    row_time_fin.append(row_time)
    row_cnt_fin.append(row_cnt)
    row_loss_fin.append(row_loss)
    col_result_fin.append(col_result)
    col_time_fin.append(col_time)
    col_cnt_fin.append(col_cnt)
    col_loss_fin.append(col_loss)

    row_result = []
    row_time = []
    row_cnt = []
    row_loss = []
    col_result = []
    col_time = []
    col_cnt = []
    col_loss = []
    for i in row_list:
        start = time.time()
        result, cnt, loss = EFDiscover(file_name, j, row=i,col=-1)
        end = time.time()
        print(file_name, j, i, 11)
        print(len(result), end - start, cnt, loss)
        row_result.append(len(result))
        row_time.append(end - start)
        row_cnt.append(cnt)
        row_loss.append(loss)
    for i in col_list:
        start = time.time()
        result, cnt, loss = EFDiscover(file_name, j, row=200,col=i)
        end = time.time()
        print(file_name, j, 200, i)
        print(len(result), end - start, cnt, loss)
        col_result.append(len(result))
        col_time.append(end - start)
        col_cnt.append(cnt)
        col_loss.append(loss)
    row_result_fin.append(row_result)
    row_time_fin.append(row_time)
    row_cnt_fin.append(row_cnt)
    row_loss_fin.append(row_loss)
    col_result_fin.append(col_result)
    col_time_fin.append(col_time)
    col_cnt_fin.append(col_cnt)
    col_loss_fin.append(col_loss)


    csv_name=['Cnt','Time','Sym-cnt','Loss']
    row_final=[row_result_fin,row_time_fin,row_cnt_fin,row_loss_fin]
    col_final=[col_result_fin,col_time_fin,col_cnt_fin,col_loss_fin]
    cnt=0
    for i in row_final:
        name=csv_name[cnt]
        dataX=np.array(i)
        dataX=dataX.T
        row_df = pd.DataFrame(dataX)
        row_df.index=row_list
        row_df.columns=['EFDiscover_nonSymLoss','EFDiscover_nonSymPop','EFDiscover']
        row_df.to_csv('row-'+name+'.csv',index=True)
        cnt+=1
    cnt=0
    for i in col_final:
        name=csv_name[cnt]
        dataX=np.array(i)
        dataX=dataX.T
        col_df = pd.DataFrame(dataX)
        col_df.index=col_list
        col_df.columns=['EFDiscover_nonSymLoss','EFDiscover_nonSymPop','EFDiscover']
        col_df.to_csv('col-'+name+'.csv',index=True)
        cnt+=1



    '''row_df = {'Row-cnt': row_list,
              'Cnt': row_result,
              'Time': row_time,
              'Sym-cnt': row_cnt,
              'Loss': row_loss}
    row_df = pd.DataFrame(row_df)

    row_df.to_csv('row-%.2f.csv' % (j), index=True)

    col_df = {'Row-cnt': col_list,
              'Cnt': col_result,
              'Time': col_time,
              'Sym-cnt': col_cnt,
              'Loss': col_loss}
    col_df = pd.DataFrame(col_df)

    col_df.to_csv('col-%.2f.csv' % (j), index=True)'''


def work_one(file_name):
    error_list = 0.01

    j = 100
    i = 11

    start = time.time()
    result, cnt, loss = EFDiscover(file_name, error_list, col=i, row=j)
    end = time.time()
    print(file_name, j, i, 11)
    print(len(result), end - start, cnt, loss)
    for i in result:
        print(i[0], i[1])
        print(f"Best solution found: {i[2]}")


if __name__ == "__main__":
    # work("abalone.data")
    # work("Rice_Cammeo_Osmancik.arff")
    # work("glass.data")
    work("A.csv")
    #work_one("glass.data")
