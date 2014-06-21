from collections import Counter
from scipy.sparse import dok_matrix
import numpy as np

class Leaf():
    def __init__(self, vec, index=0):
        self.vec = vec
        self.i = index
    
    def do(self):
        return self.vec
    
    def grad(self, delta):
        """ Create the gradients of a node """
        global T, M, b, W
        gb, gM, gT = np.zeros(b.shape), np.zeros(M.shape), np.zeros(T.shape)
        gW = dok_matrix(W.shape) # Keep word gradients in a sparse array
        gW[self.i] = delta
        return (gW, gb, gM, gT)

class Tree(Leaf):
    def __init__(self, left, right, nl=lambda x: max(0,x), nld=lambda x: int(x>0)):
        self.left = left
        self.right = right
        self.nl = np.vectorize(nl) # transfer function
        self.nld = np.vectorize(nld) # transfer derivative
        this.vec = None # store activation
        
    def do(self):
        """ Forward pass """
        global T, M, b
        l, r = self.left.do(), self.right.do()
        this.vec = self.nl(l.T.dot(T).dot(r) + M.dot(np.append(l,r)) + b)
        return this.vec
    
    def grad(self, delta, out):
        """ Gradients """
        global T, M, b
        g = self.nld(out) * delta
        l, r = self.left.vec, self.right.vec
        
        delta_l = (T.dot(r).T + M[:,:M.shape[0]].T).dot(g)
        delta_r = (l.dot(T).T + M[:,M.shape[0]:].T).dot(g)
        (gWl, gbl, gMl, gTl) = self.left.grad(delta_l)
        (gWr, gbr, gMr, gTr) = self.right.grad(delta_r)
        
        gb = gbl + gbr + g
        gM = gMl + gMr + g[:, None] * np.append(l, r)
        gT = gTl + gTr + g[None, None].T * (l[:, None] * r)
        
        return (gWl + gWr, gb, gM, gT)
        


n=2 # vector size
m=1000 # dictionary size

# Globals:
W = dok_matrix((m,n)) # Keep word vectors in a sparse array?
b = np.random.randn(n)
M = np.random.randn(n,2*n)
T = np.random.randn(n,n,n)

# Test:
t = Tree(Leaf(np.random.randn(n)), Leaf(np.random.randn(n)))
t.do()
t.grad(np.random.randn(n), np.random.randn(n))
