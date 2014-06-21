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
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vec = None # store activation
        
    def do(self):
        """ Forward pass """
        global T, M, b, nl
        l, r = self.left.do(), self.right.do()
        self.vec = nl(l.T.dot(T).dot(r) + M.dot(np.append(l,r)) + b)
        return self.vec
    
    def grad(self, delta, out):
        """ Get the gradients of this tree """
        global T, M, b, nld
        g = nld(out) * delta
        l, r = self.left.vec, self.right.vec
        
        delta_l = (T.dot(r).T + M[:,:M.shape[1]/2].T).dot(g)
        delta_r = (l.dot(T).T + M[:,M.shape[1]/2:].T).dot(g)
        (gWl, gbl, gMl, gTl) = self.left.grad(delta_l)
        (gWr, gbr, gMr, gTr) = self.right.grad(delta_r)
        
        gb = gbl + gbr + g
        gM = gMl + gMr + g[:, None] * np.append(l, r)
        gT = gTl + gTr + g[None, None].T * (l[:, None] * r)
        
        return (gWl + gWr, gb, gM, gT)

def step(left_tree, right_tree, true_relation):
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    ## Run forward
    l, r = left_tree.do(), right_tree.do()
    comparison = nl2(l.T.dot(T2).dot(r) + M2.dot(np.append(l,r)) + b2)
    softmax = normalize(np.exp( S.dot(np.append(1, comparison)) ))

    cost = -np.log(softmax[true_relation])
    
    ## Get gradients
    # softmax
    diff = softmax - np.eye(c1)[true_relation]
    delta = ( nld2(np.append(1, comparison)) * S.T.dot(diff) )[1:]
    gS = np.append(1, comparison) * diff[:, None]
    # comparison
    g = nld2(comparison) * delta
    gb2 = g
    gM2 = g[:, None] * np.append(l, r)
    gT2 = g[None, None].T * (l[:, None] * r)
    # composition
    delta_l = (T2.dot(r).T + M2[:,:M2.shape[1]/2].T).dot(g)
    delta_r = (l.dot(T2).T + M2[:,M2.shape[1]/2:].T).dot(g)
    (gWl, gbl, gMl, gTl) = left_tree.grad( delta_l, l)
    (gWr, gbr, gMr, gTr) = right_tree.grad(delta_r, r)
    gb, gM, gT = (gbl + gbr, gMl + gMr, gTl + gTl)
    
    return cost, (gS, (gb2, gM2, gT2), (gb, gM, gT)), np.argmax(softmax)


n=2 # vector size
h=3 # hidden layer size
m=1000 # dictionary size
c1=4 # number of comparison classes
c2=3 # size of comparison layer

W = dok_matrix((m,n)) # Keep word vectors in a sparse array?

nl  = np.vectorize(lambda x: max(0.,x)) # composition transfer function
nld = np.vectorize(lambda x: float(x>0)) # composition transfer derivative
# Composition layer
b = np.random.randn(h)
M = np.random.randn(h,2*n)
T = np.random.randn(h,n,n)

nl2  = np.vectorize(lambda x: max(0.,x)) # comparison transfer function
nld2 = np.vectorize(lambda x: float(x>0)) # comparison transfer derivative
# Softmax layer
S  = np.random.randn(c1, c2+1)
# Comparison layer
b2 = np.random.randn(c2)
M2 = np.random.randn(c2,2*h)
T2 = np.random.randn(c2,h,h)

step(
    Tree(Leaf(np.random.randn(n)), Leaf(np.random.randn(n))),
    Tree(Leaf(np.random.randn(n)), Leaf(np.random.randn(n))),
    2
)
