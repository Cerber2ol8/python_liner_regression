import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import os
#线性数据生成器

def random_data(count, dimension):
    X = np.zeros((count, dimension))
    #print(np.shape(X))
    for d in range(count):
        x = np.zeros(dimension).reshape(-1,1)
        for c in range(dimension):

            #正态分布
            #x[c] = np.random.normal()
            #均匀分布
            x[c] = np.random.uniform()
        
        X[d] = x.T
    return X


def generator(count=20, dimension=2):
    """生成数据的数量，维度
    """
    X = random_data(count,dimension)
    Y = np.zeros((count,1))

    #指定系数或者随机生成系数
    a = [k+2 for k in range(dimension)]
    #a = [np.random.uniform() for k in range(dimension)]


    for i in range(count):
        y = 0
        for j in range(dimension):
            
            x = X[i][j]
            y += x * a[j]


        y = y * (1 + np.random.normal(-0.1,0.1))
        Y[i] = y


    print('X Shape:{}, Y Shape:{}.'.format(np.shape(X),np.shape(Y)))

    return X,Y

def save_to_csv(X,Y):
    data = np.hstack((X,Y))
    #print(data)
    c = ['X' + str(i) for i in range(np.shape(X)[1])]
    c.append('label')
    #print(c)
    df = DataFrame(data=data, index=None, columns=c)

    #z = zip(['X' + str(i) for i in range(np.shape(X)[1])],[X.T[i].reshape(-1,1) for i in range(np.shape(X)[1])])
    #z = dict(z)
    #data = {'label':Y}
    #data.update(z)
    pwd = os.getcwd()
    df.to_csv(path_or_buf = pwd + '/data.csv')
    return df

if __name__ == '__main__':
    #生成数据的数量，维度
    X,Y = generator(20000,7)
    #保存数据
    df = save_to_csv(X,Y)
    print(df)
