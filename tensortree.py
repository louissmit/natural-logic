from collections import Counter
from scipy.sparse import dok_matrix
import numpy as np

def tensorLayer((l,r), (T, M, b), nl):
    """ Compute Inner Tensor Layer """
    return nl(l.T.dot(T).dot(r) + M.dot(np.append(l,r)) + b)

def tensorGrad ((l,r), (T, M, b), delta, nld, output):
    """ Compute Tensor Layer Gradients """
    g = nld(output) * delta
    gb = g
    gM = g[:, None] * np.append(l, r)
    gT = g[None, None].T * (l[:, None] * r)
    delta_l = (T.dot(r).T + M[:,:M.shape[1]/2].T).dot(g)
    delta_r = (l.dot(T).T + M[:,M.shape[1]/2:].T).dot(g)
    return (gT, gM, gb), (delta_l, delta_r)

class Leaf():
    def __init__(self, index):
        self.i = index
        self.vec = None # store activation

    def do(self, (p, W), nl):
        """ Look up word in vocabulary """
        self.vec = W[self.i]
        return self.vec
    
    def grad(self, delta, output, ((T, M, b), W), nld):
        """ Create the gradients of a node """
        gb, gM, gT = np.zeros(b.shape), np.zeros(M.shape), np.zeros(T.shape)
        # Keep word gradients in a sparse array, so we can add them up
        gW = dok_matrix(W.shape)
        gW[self.i] = delta
        return ((gT, gM, gb), gW)

class Tree(Leaf):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vec = None # store activation
        
    def do(self, param, nl):
        """ Forward pass """
        l, r = self.left.do(param, nl), self.right.do(param, nl)
        self.vec = tensorLayer((l,r), param[0], nl)
        return self.vec
    
    def grad(self, delta, output, param, nld):
        """ Get the gradients of this tree """
        l, r = self.left.vec, self.right.vec
        
        (gT, gM, gb), (delta_l, delta_r) = \
            tensorGrad ((l,r), param[0], delta, nld, output)

        ((gTl, gMl, gbl), gWl) = self.left.grad (delta_l, l, param, nld)
        ((gTr, gMr, gbr), gWr) = self.right.grad(delta_r, r, param, nld)
        
        return ((gTl + gTr + gT, gMl + gMr + gM, gbl + gbr + gb), gWl + gWr)


if __name__ == "__main__":
    nl  = np.vectorize(lambda x: max(0.,x))
    nld = np.vectorize(lambda x: float(x>0))

    n = 2
    v = 10
    b = np.random.randn(n)
    M = np.random.randn(n,2*n)
    T = np.random.randn(n,n,n)
    W = np.random.randn(v,n)

    gold = np.random.randn(n)

    tree = Tree(Tree(Leaf(0), Leaf(1)), Leaf(2))
    d =  tree.do(((T, M, b), W), nl)
    print d
    print tree.grad(gold-d, d, ((T, M, b), W), nld)




