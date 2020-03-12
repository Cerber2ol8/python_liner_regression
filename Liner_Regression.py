import numpy as np
from scipy import stats
import data_reader
import os
import six

#线性回归模型主体


path = os.getcwd() + '/data.csv'
input,label = read_data(path)[:,:-1],read_data(path)[:,-1:]

#学习率
alpha = 0.001
#BACH_SIZE
BATCH_SIZE = 512
#数据集尺寸
input_size = np.shape(input)[1]
label_size = np.shape(label)[1]
m = np.shape(label)[0]

##正则化
#maximums, minimums, avgs = input.max(axis=0), input.min(axis=0), input.sum(axis=0)/m
#l_maximums, l_minimums, l_avgs = label.max(axis=0), label.min(axis=0), label.sum(axis=0)/m

#for i in six.moves.range(input_size):
#   input[:, i] = (input[:, i] - avgs[i]) / (maximums[i] - minimums[i]) 

#for i in six.moves.range(output_size):
#    label[:, i] = (label[:, i] - l_avgs[i]) / (l_maximums[i] - l_minimums[i]) 


ratio = 0.8
offset = int(np.shape(input)[0] * ratio)

train_data = input[:offset]
train_label = label[:offset]

test_data = input[offset:]
test_label = label[offset:]



print('input_size:{} \nlabel_size:{}'.format(input_size,label_size))
#exit()

#def reader(data, bach_size):
#    for d in data:
#        yield d
        
#    pass


def MSE_function(theta, X, y):
    
    diff = np.dot(X,theta) - y
    J = (1/(2*m)) * np.dot(np.transpose(diff),diff)
    return J

def predict_funtcion():
    pass

def gradient_function(theta, X, y):

    diff = np.dot(X,theta) - y
    delta_J = (1/m) * np.dot(np.transpose(X),diff)
    return delta_J

def gradient_descent(X, y, alpha):
    theta = np.ones(input_size).reshape(input_size,1)

    gradient = gradient_function(theta, X, y)
    #print(gradient)
    step = 0
    while not np.all(np.absolute(gradient) <= 1e-6):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        if step % 1000 == 0:
            error = MSE_function(theta, test_data, test_label)[0][0]
            print('当前损失:{}'.format(error))
        step +=1
    return theta



if __name__ == '__main__':
    optimal = gradient_descent(train_data, train_label, alpha)
    #error = MSE_function(optimal, input, label)
    print(optimal)