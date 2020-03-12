import numpy as np
import paddle 
import paddle.fluid as fluid
import sys
import os
import data_reader
from ploter import PlotData,Ploter
import math
#使用paddle的全连接网络

#读取本地csv数据
path = os.getcwd() + '/data.csv'
data = data_reader.read_data(path)

def reader_creator(data):
    def reader():
        for d in data:
            yield d[:-1], d[-1:] 

    return reader



#学习率
learning_rate = 0.001
#BACH_SIZE
BATCH_SIZE = 512
#数据集尺寸
input_size = np.shape(data[:,:-1])[1]
output_size = np.shape(data[:,-1:])[1]
m = np.shape(data)[0]

# 训练集和验证集的划分比例
ratio = 0.8 
offset = int(data.shape[0]*ratio)
train_data = data[:offset]
test_data = data[offset:]

train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(train_data), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(test_data), buf_size=500),
        batch_size=BATCH_SIZE)

#定义数据
input = fluid.data(name='input', dtype='float32',shape=[None, input_size])
label = fluid.data(name='label', dtype='float32',shape=[None, output_size])
prediction = fluid.layers.fc(input=input, size=1, act=None)


#优化器配置
cost = fluid.layers.square_error_cost(input=prediction, label=label)
avg_loss = fluid.layers.mean(cost) 
sgd = fluid.optimizer.SGD(learning_rate=learning_rate)
sgd.minimize(avg_loss)

#paddle program配置
main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序
test_program = main_program.clone(for_test=True)

#运算场所配置
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

#配置执行器
#executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)
#向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，调用run(...)执行program。
exe = fluid.Executor(place)


def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失

params_dirname = "liner_regression_paddle.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[input, label])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"

plot_prompt = Ploter(train_prompt, test_prompt)


exe_test = fluid.Executor(place)
num_epochs = 100

def train(num_epochs):
    pass 

step = 0
for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                    feed=feeder.feed(data_train),
                                    fetch_list=[avg_loss])
        if step % 10 == 0: # 每10个批次记录并输出一下训练损失
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plt.clf()
            plot_prompt.plot()
            plot_prompt.plt.pause(0.1) 
            plot_prompt.plt.ioff() 

            print("%s, Step %d, Cost %f" %
	                    (train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                        program=test_program,
                                        reader=test_reader,
                                        fetch_list=[avg_loss.name],
                                        feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            #plot_prompt.plot()

            print("%s, Step %d, Cost %f" %
	                    (test_prompt, step, test_metics[0]))
            if test_metics[0] < 6.0: # 如果准确率达到要求，则停止训练
                plot_prompt.plt.pause(0)
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        #if params_dirname is not None:
            #fluid.io.save_inference_model(params_dirname, ['x'], [prediction], exe)




if __name__ == '__main__':
    
    #train(num_epochs)

    pass
