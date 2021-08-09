import cupy as cp
# class operate():
#     def __init__(self):
#         self.children = []
#         self.parents = []
#         self.dtx = None
#         self.value = None
from  shucflow.Graph import *
from shucflow.util import *
class FullConnect():
    def __init__(self,inputs,w,model=default_Model):
        self.batch_size = None
        self.H = inputs.w
        self.weight = cp.random.normal(size=(self.H,w),scale=1,loc=0)
        self.bias = cp.random.normal(size=(w),scale=1,loc=0)

        # self.x = inputs.value
        self.children = []
        self.parents = [inputs]

        self.shape = None
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
        self.value = 0
        self.dw = 0
        self.db = 0
        self.dtx = {}
    def forward(self):
        # self.shape = cp.shape(self.parents[0].value)
        self.batch_size = cp.shape(self.parents[0].value)[0]

        self.value = cp.matmul(self.parents[0].value,self.weight) + self.bias
        return self.value

    def compute_grad(self):
        self.dtx[self.parents[0]] = 0

        for child in self.children:
            self.dtx[self.parents[0]] += cp.matmul(child.dtx[self],self.weight.T)
            self.dw += cp.matmul(self.parents[0].value.T,child.dtx[self])
            self.db += cp.sum(child.dtx[self],axis=0)
        return

    def apply_Grad(self,lr):
        self.weight = self.weight - lr * self.dw
        self.bias = self.bias - lr * self.db
        return

class softmax_cross_entropy():
    def __init__(self,inputs,model=default_Model):
        self.loss = None
        self.y = None
        self.value = None
        self.dtx = {}
        # self.x = inputs.value
        self.t = None
        self.batch_size = None
        self.parents = [inputs]
        self.children = []
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
    def softmax(self,x):
        if x.ndim == 2:
            x = x.T
            x = x - cp.max(x, axis=0)
            y = cp.exp(x) / cp.sum(cp.exp(x), axis=0)
            return y.T
        x = x - cp.max(x)  # 溢出对策
        return cp.exp(x) / cp.sum(cp.exp(x))
    def cross_entropy(self,y,t):
        # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
        t = t.argmax(axis=1)
        return -cp.sum(cp.log(y[cp.arange(self.batch_size),t] + 1e-7)) / self.batch_size
    def forward(self):
        self.batch_size = cp.shape(self.parents[0].value)[0]
        self.y = self.softmax(self.parents[0].value)
        self.value = self.y
        self.loss = self.cross_entropy(self.y,self.t)
        return self.loss
    def compute_grad(self):
        # 监督数据是one-hot-vector的情况
        self.dtx[self.parents[0]] = (self.y - self.t) / self.batch_size
        return self.dtx
    def apply_Grad(self,lr):
        return

class input_node():
    def __init__(self,h,w,c=3,batch_size=100,model=default_Model):
        self.value = None
        self.shape = None
        self.batch_size = batch_size
        self.channle = c
        self.children = []
        self.parents = None
        self.H=h
        self.W=w
        model.op_node_dict[self] = []
    def forward(self):
        return self.value
    def compute_grad(self):
        return
    def apply_Grad(self,lr):
        return

class Relu():
    def __init__(self,inputs,model=default_Model):
        # self.x = inputs.value
        self.parents = [inputs]
        self.children = []
        self.value = None
        self.mask = None
        self.dtx = {}
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
    def forward(self):
        self.mask = (self.parents[0].value <=0)
        self.value = self.parents[0].value
        self.value[self.mask] = 0
        return self.value
    def compute_grad(self):
        self.dtx[self.parents[0]] = 0
        for child in self.children:
            self.dtx[self.parents[0]] += child.dtx[self]
        self.dtx[self.parents[0]][self.mask] = 0
        return self.dtx
    def apply_Grad(self,lr):
        return

class Conv2d():
    def __init__(self,inputs,FN,FH,FW,stride=1,pad=0,model=default_Model):
        self.batch_size = inputs.batch_size
        self.channle = FN
        self.FN = FN
        self.stride = stride
        self.FH = FH
        self.FW = FW
        self.pad = pad
        self.weight = cp.random.normal(size=(FN,inputs.channle,FH,FW),scale=1,loc=0)
        self.bias = cp.random.normal(size=(FN),scale=1,loc=0)
        self.parents = [inputs]
        self.children = []
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
        self.value = None
        self.col = None
        self.col_w = None
        self.dw = 0
        self.db = 0
        self.dtx = {}
        self.H = 1 + int((inputs.H + 2 * self.pad - self.FH) / self.stride)
        self.W = 1 + int((inputs.W + 2 * self.pad - self.FW) / self.stride)

    def forward(self):
        self.col = im2col(self.parents[0].value,self.FH,self.FW,self.stride,self.pad)
        self.col_w = self.weight.reshape(self.FN,-1).T
        self.value = cp.matmul(self.col,self.col_w) + self.bias
        self.value = self.value.reshape(self.batch_size,self.H,self.W,-1).transpose(0,3,1,2)
        return self.value

    def compute_grad(self):
        self.dtx[self.parents[0]] = 0
        for child in self.children:
            dout = child.dtx[self].transpose(0,2,3,1).reshape(-1,self.FN)
            self.db += cp.sum(child.dtx[self], axis=0)
            self.dw += cp.matmul(self.col.T, dout)
            self.dtx[self.parents[0]] += cp.matmul(dout, self.col_w.T)
        self.dw = self.dw.transpose(1,0).reshape(self.FN,self.parents[0].channle,self.FH,self.FW)
        self.dtx[self.parents[0]] = col2im(self.dtx[self.parents[0]],self.parents[0].value.shape,self.FH,self.FW,self.stride,self.pad)
        return

    def apply_grad(self,lr):
        self.weight = self.weight - lr * self.dw
        self.bias = self.bias - lr * self.db
        return

class pooling():
    def __init__(self,inputs,PH,PW,stride=1,pad=0,model=default_Model):
        self.PH = PH
        self.PW = PW
        self.batch_size = inputs.batch_size
        self.pad = pad
        self.stride = stride
        self.channle = inputs.channle
        self.H = 1 + int((inputs.H + 2 * self.pad - self.PH) / self.stride)
        self.W = 1 + int((inputs.W + 2 * self.pad - self.PW) / self.stride)
        self.parents = [inputs]
        self.children = []
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
        self.value = None
        self.col = None
        self.arg_max = None
        self.dtx = {}

    def forward(self):
        col = im2col(self.parents[0].value,self.PH,self.PW,self.stride,self.pad)
        col = col.reshape(-1,self.PH * self.PW)

        self.arg_max = cp.argmax(col,axis=1)
        self.value = cp.max(col,axis=1)
        self.value = self.value.reshape(self.batch_size,self.H,self.W,self.channle).transpose(0,3,1,2)
        return self.value

    def compute_grad(self):
        pool_size = self.PH * self.PW
        dmax = cp.zeros((self.children[0].dtx[self].size,pool_size))
        for child in self.children:
            dout = child.dtx[self].transpose(0,2,3,1)
            dmax[cp.arange(self.arg_max.size),self.arg_max.flatten()] += dout.flatten()
        dmax = dmax.reshape(self.children[0].dtx[self].shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        self.dtx[self.parents[0]] = col2im(dcol,self.parents[0].value.shape,self.PH,self.PW,self.stride,self.pad)
        return self.dtx

    def apply_grad(self,lr):
        return

class flatten():
    def __init__(self,inputs,col,model=default_Model):
        self.batch_size = inputs.batch_size
        self.parents = [inputs]
        self.children = []
        inputs.children.append(self)
        model.op_node_dict[self] = []
        if model.op_node_dict.get(inputs) == None:
            model.op_node_dict[inputs] = [self]
        else:
            model.op_node_dict[inputs].append(self)
        self.w = inputs.channle * inputs.H * inputs.W
        self.value = None
        self.dtx = {}

    def forward(self):
        self.value = cp.reshape(self.parents[0].value,(self.batch_size,-1))
        return

    def compute_grad(self):
        self.dtx[self.parents[0]] = cp.reshape(self.children[0].dtx[self],(self.batch_size,self.parents[0].channle,self.parents[0].H,self.parents[0].W))
        return

    def apply_grad(self,lr):
        return
