import cupy as cp
import numpy as np
from shucflow.Operate import *
from shucflow.mnist import *
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
x_train = cp.array(x_train)
t_train = cp.array(t_train)
x_test = cp.array(x_test)
t_test = cp.array(t_test)

x = input_node(h=1,w=784,c=1,batch_size=100)
fc1 = FullConnect(inputs=x,w=50)
re = Relu(inputs=fc1)
fc2 = FullConnect(inputs=re,w=10)
sce = softmax_cross_entropy(inputs=fc2)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size , 1)
for i in range(iters_num):
    batch_mask = cp.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    x.value = x_batch
    sce.t = t_batch
    # print(fc1.batch_size)
    # fc1.forward()
    # print(fc1.value)
    default_Model.forward()
    default_Model.compute_grad()
    default_Model.backward()
    # print(fc1.batch_size)
    pre_y=cp.argmax(sce.y,axis=-1)
    lab_t = cp.argmax(t_batch,axis=-1)
    acc_mask=(pre_y == lab_t).astype(float)
    accuracy = cp.mean(acc_mask)
    # p_shape = cp.shape(pre_y)
    # la_shape = cp.shape(lab_t)
    print("the train accuracy is %f"%accuracy)
    print(sce.loss)
pass

