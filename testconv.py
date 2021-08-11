from shucflow.Operate import *
from shucflow.mnist import *
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,flatten=False,one_hot_label=True)
x_train = cp.array(x_train)
t_train = cp.array(t_train)
x_test = cp.array(x_test)
t_test = cp.array(t_test)

x = input_node(h=28,w=28,c=1,batch_size=100)
c1 = Conv2d(x,FN=30,FH=5,FW=5,pad=0,stride=1)
r1 = Relu(c1)
p1 = pooling(r1,PH=3,PW=3,stride=2,pad=0)
flat = flatten(p1)
fc1 = FullConnect(flat,w=100)
r2 = Relu(fc1)
fc2 = FullConnect(r2,w=10)
pre = softmax_cross_entropy(fc2)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size , 1)

for i in range(iters_num):
    batch_mask = cp.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    x.value = x_batch
    pre.t = t_batch

    default_Model.forward()
    default_Model.compute_grad()
    default_Model.backward(lr=0.001)


    pre_y = cp.argmax(pre.y, axis=-1)
    lab_t = cp.argmax(t_batch, axis=-1)
    acc_mask = (pre_y == lab_t).astype(float)
    accuracy = cp.mean(acc_mask)
    # p_shape = cp.shape(pre_y)
    # la_shape = cp.shape(lab_t)
    print("the train accuracy is %f" % accuracy)
    print(pre.loss)

pass
