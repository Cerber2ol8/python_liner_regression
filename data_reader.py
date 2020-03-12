import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import os


def read_data(path):
    """ return numpy.array
    """
    df = pd.read_csv(path)
    rows = df.shape[0]
    columns = df.shape[1] - 2
    X = np.zeros((rows,columns))
    for i in range(columns):
        x = df['X{}'.format(i)]
        x = np.array(x)
        X[:,i] = x
    Y = np.array(df['label']).reshape(-1,1)

    return np.hstack((X, Y))


if __name__ == '__main__':
    path = os.getcwd() + '/data.csv'

    X,Y = read_data(path)[:,:-1],read_data(path)[:,-1:]
    print('X:',X)
    print('Y:',Y)
