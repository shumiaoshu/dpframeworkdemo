class graph():
    def __init__(self):
        self.op_node_dict={}
        self.node_seq = []
        self.back_node_seq = []
    def topoSort(self,G):
        in_degree = dict((u,0) for u in G.keys())
        for u in G:
            for v in G[u]:
                in_degree[v] +=1
        Q=[u for u in in_degree.keys() if in_degree[u]==0]
        seq=[]
        while Q:
            s=Q.pop()
            seq.append(s)
            for u in G[s]:
                in_degree[u] -=1
                if in_degree[u]==0:
                    Q.append(u)
        return seq
    def forward(self):
        global node_seq
        self.node_seq = self.topoSort(self.op_node_dict)
        # self.node_seq[0].value = inputs
        for op_node in self.node_seq:
            op_node.forward()
    def compute_grad(self):
        self.back_node_seq = self.node_seq[::-1]
        for op_node in self.back_node_seq:
            op_node.compute_grad()
    def backward(self,lr=0.001):
        # lr = 0.0001
        for op_node in self.back_node_seq:
            op_node.apply_Grad(lr)
    def __call__(self):
        self.forward()
default_Model = graph()
if __name__ == "__main__":
    testgraph= graph()
    G = {
        'a': ['b', 'f'],
        'b': ['c', 'd', 'f'],
        'c': ['d'],
        'd': ['e', 'f'],
        'e': ['f'],
        'f': []
    }
    res = testgraph.topoSort(G)
    print(res[::-1])