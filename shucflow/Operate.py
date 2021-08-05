import cupy as cp
# class operate():
#     def __init__(self):
#         self.children = []
#         self.parents = []
#         self.dtx = None
#         self.value = None
from  shucflow.Graph import *
class FullConnect():
    def __init__(self,inputs,raw,col,model=default_Model):
        self.batch_size = None
        self.weight = cp.random.normal(size=(raw,col),scale=1,loc=0)
        self.bias = cp.random.normal(size=(col),scale=1,loc=0)

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
        self.value = None
        self.dw = None
        self.db = None
        self.dtx = {}
    def forward(self):
        # self.shape = cp.shape(self.parents[0].value)
        self.batch_size = cp.shape(self.parents[0].value)[0]

        self.value = cp.matmul(self.parents[0].value,self.weight) + self.bias
        return self.value

    def compute_grad(self):
        self.dtx[self.parents[0]] = 0
        self.dw = 0
        self.db = 0
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
    def __init__(self,model=default_Model):
        self.value = None
        self.shape = None
        self.children = []
        self.parents = None
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


