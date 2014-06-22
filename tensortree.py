from collections import Counter
from scipy.sparse import dok_matrix
import numpy as np

def tensorLayer((l,r), (T, M, b), nl):
    """ Compute Inner Tensor Layer """
    print l
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

def init_ntn_params(i,o):
    """ Random Neural Tensor Network parameters"""
    return (
        np.random.randn(o,i,i), # T
        np.random.randn(o,2*i), # M
        np.random.randn(o)      # b
    )

class Leaf():
    def __init__(self, index):
        self.i = index
        self.vec = None # store activation

    def do(self, hyp, param):
        """ Look up word in vocabulary """
        self.vec = param.vocab[self.i]
        return self.vec
    
    def grad(self, delta, output, hyp, param):
        """ Create the gradients of a node """
        (T, M, b) = param.composition
        gb, gM, gT = np.zeros(b.shape), np.zeros(M.shape), np.zeros(T.shape)
        # Keep word gradients in a sparse array, so we can add them up
        gW = dok_matrix((hyp.vocab_size, hyp.word_size))
        gW[self.i] = delta
        return ((gT, gM, gb), gW)

class Tree(Leaf):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.vec = None # store activation
        
    def do(self, hyp, param):
        """ Forward pass """
        l, r = self.left.do(hyp, param), self.right.do(hyp, param)
        self.vec = tensorLayer((l,r), param.composition, hyp.composition_transfer)
        return self.vec
    
    def grad(self, delta, output, hyp, param):
        """ Get the gradients of this tree """
        l, r = self.left.vec, self.right.vec
        
        (gT, gM, gb), (delta_l, delta_r) = \
            tensorGrad ((l,r), param.composition, delta, hyp.composition_backtrans, output)

        ((gTl, gMl, gbl), gWl) = self.left.grad (delta_l, l, hyp, param)
        ((gTr, gMr, gbr), gWr) = self.right.grad(delta_r, r, hyp, param)
        
        return ((gTl + gTr + gT, gMl + gMr + gM, gbl + gbr + gb), gWl + gWr)

class SGD():
    def __init__(self, hyp):
        self.hyp = hyp

    def cost_and_grad(s, left_tree, right_tree, true_relation, param):
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm else v

        nl  = hyp.comparison_transfer
        nld = hyp.comparison_backtrans
        S = param.softmax

        ## Run forward
        l, r = left_tree.do(s.hyp, param), right_tree.do(s.hyp, param)
        comparison = tensorLayer((l,r), param.comparison, nl)
        softmax = normalize(np.exp( S.dot(np.append(1, comparison)) ))

        cost = -np.log(softmax[true_relation])
        
        ## Get gradients
        # softmax
        diff = softmax - np.eye(hyp.classes)[true_relation]
        delta = ( nld(np.append(1, comparison)) * S.T.dot(diff) )[1:]
        gS = np.append(1, comparison) * diff[:, None]
        # comparison
        (gT2, gM2, gb2), (delta_l, delta_r) = \
                tensorGrad ((l,r), param.comparison, delta, nld, comparison)
        # composition
        ((gTl, gMl, gbl), gWl) = left_tree.grad (delta_l, l, s.hyp, param)
        ((gTr, gMr, gbr), gWr) = right_tree.grad(delta_r, r, s.hyp, param)
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
    hyp.comparison_size = 4
    hyp.classes = 7
    hyp.composition_transfer = relu
    hyp.composition_backtrans = relud
    hyp.comparison_transfer = relu
    hyp.comparison_backtrans = relud

    param = Parameters()
    param.vocab = { 0: np.random.randn(hyp.word_size) }
    param.composition = init_ntn_params(hyp.word_size, hyp.word_size)
    param.comparison  = init_ntn_params(hyp.word_size, hyp.comparison_size)
    param.softmax = np.random.randn(hyp.classes, hyp.comparison_size+1)

    l = Leaf(0)
    r = Tree(Tree(Leaf(0), Leaf(0)), Leaf(0))

    sgd = SGD(hyp)
    print sgd.cost_and_grad(l,r,2, param)

