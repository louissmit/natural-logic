class Leaf():
    def __init__(self, vec):
        self.vec = vec
    
    def do(self):
        return self.vec

class Tree(Leaf):        
    def __init__(self, left, right, nl=lambda x: max(0,x)):
        self.left = left
        self.right = right
        self.nl = np.vectorize(nl)
    def do(self):
        global T, M, b
        l, r = self.left.do(), self.right.do()
        return self.nl(l.T.dot(T).dot(r) + M.dot(np.append(l,r)) + b)


n=2
b = np.random.randn(n)
M = np.random.randn(n,2*n)
T = np.random.randn(n,n,n)

t = Tree(Leaf(np.random.randn(n)), Leaf(np.random.randn(n)))
t.do()
