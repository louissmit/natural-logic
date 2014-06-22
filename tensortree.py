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

def ntn_dtype(i, o):
    """ The dtype of the parameters of a Neural Tensor Network"""
    return np.dtype([('b', '<f8', (o,)),       # bias
                     ('M', '<f8', (o, 2 * i)), # matrix
                     ('T', '<f8', (o, i, i))]) # tensor

def random_dtype(dt):
    """ Random numpy array for a given dtype"""
    return np.array(tuple([random.randn(*a[2]) for a in dt.descr]), dtype=dt)

class Leaf():
    def __init__(self, vec, index=0, hyp, param):
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
        return ((gT, gM, gb), gW)

class Tree(Leaf):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vec = None # store activation
        
    def do(self):
        """ Forward pass """
        global T, M, b, nl
        l, r = self.left.do(), self.right.do()
        self.vec = tensorLayer((l,r), (T, M, b), nl)
        return self.vec
    
    def grad(self, delta, output):
        """ Get the gradients of this tree """
        global T, M, b, nld
        l, r = self.left.vec, self.right.vec
        
        (gT, gM, gb), (delta_l, delta_r) = \
            tensorGrad ((l,r), (T, M, b), delta, nld, output)

        ((gTl, gMl, gbl), gWl) = self.left.grad(delta_l)
        ((gTr, gMr, gbr), gWr) = self.right.grad(delta_r)
        
        return ((gTl + gTr + gT, gMl + gMr + gM, gbl + gbr + gb), gWl + gWr)

class SGD():
    def __init__(self, hyp, param = None):
        self.hyp = hyp
        if param:
            self.param = param

    def cost_and_grad(self, left_tree, right_tree, true_relation):
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm else v

        ## Run forward
        l, r = left_tree.do(), right_tree.do()
        comparison = tensorLayer((l,r), (T2, M2, b2), nl2)
        softmax = normalize(np.exp( S.dot(np.append(1, comparison)) ))

        cost = -np.log(softmax[true_relation])
        
        ## Get gradients
        # softmax
        diff = softmax - np.eye(c1)[true_relation]
        delta = ( nld2(np.append(1, comparison)) * S.T.dot(diff) )[1:]
        gS = np.append(1, comparison) * diff[:, None]
        # comparison
        (gT2, gM2, gb2), (delta_l, delta_r) = \
                tensorGrad ((l,r), (T2, M2, b2), delta, nld2, comparison)
        # composition
        ((gTl, gMl, gbl), gWl) = left_tree.grad (delta_l, l)
        ((gTr, gMr, gbr), gWr) = right_tree.grad(delta_r, r)
        ((gT, gM, gb), gW) = ((gTl + gTr, gMl + gMr, gbl + gbr), gWl + gWr)
        
        return cost, (gS, (gT2, gM2, gb2), (gT, gM, gb), gW), np.argmax(softmax)

class Parameters(): pass
class HyperParameters(): pass

if __name__ == "__main__":
    relu  = np.vectorize(lambda x: max(0.,x))
    relud = np.vectorize(lambda x: float(x>0))

    hyp = HyperParameters()
    hyp.vocab_size = 100
    hyp.word_size = 2
    hyp.composition_size = 3
    hyp.comparison_size = 4
    hyp.classes = 7
    hyp.composition_transfer = relu
    hyp.composition_backtrans = relud
    hyp.comparison_transfer = relu
    hyp.comparison_backtrans = relud

    param = Parameters()
    param.vocab = dok_matrix((hyp.vocab_size, hyp.word_size))
    param.composition = random_dtype(ntn_dtype(hyp.word_size, hyp.composition_size))
    param.comparison  = random_dtype(ntn_dtype(hyp.composition_size, hyp.comparison_size))
    param.softmax = np.random.randn(hyp.classes, hyp.comparison_size)





