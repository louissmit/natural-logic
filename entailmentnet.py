import numpy as np
from tensortree import Leaf, Tree, tensorLayer, tensorGrad

class HyperParameters(): pass

class Net():
    def __init__(self, hyp):
        self.hyp = hyp
        def ntn_dims(i,o):
            return ((o, i, i), (o, 2*i), (o,))

        self.dims = ((hyp.classes, hyp.comparison_size+1),) \
            + ntn_dims(hyp.word_size, hyp.comparison_size) \
            + ntn_dims(hyp.word_size, hyp.word_size) \
            + ((hyp.vocab_size, hyp.word_size),)
        self.theta = np.random.randn(np.sum(np.prod(d) for d in self.dims))

    def params(self):
        i=0
        out = []
        for d in self.dims:
            j = np.prod(d)
            out.append( self.theta[i:i+j].reshape(*d) )
            i += j
        return tuple(out)

    # np.hstack(p.flat for p in par2.params())

    def cost_and_grad(s, left_tree, right_tree, true_relation):
        def normalize(v):
            norm = np.linalg.norm(v)
            assert not np.isinf(norm), 'softmax is too big'
            return v / norm if norm else v

        nl  = s.hyp.comparison_transfer
        nld = s.hyp.comparison_backtrans
        (S, T2, M2, b2, T, M, b, W) = s.params()

        ## Run forward
        l = left_tree.do (((T, M, b), W), nl)
        r = right_tree.do(((T, M, b), W), nl)
        comparison = tensorLayer((l,r), (T2, M2, b2), nl)
        softmax = normalize(np.exp( S.dot(np.append(1, comparison)) ))
        
        cost = -np.log(softmax[true_relation])
        
        ## Get gradients
        # softmax
        diff = softmax - np.eye(s.hyp.classes)[true_relation]
        delta = ( nld(np.append(1, comparison)) * S.T.dot(diff) )[1:]
        gS = np.append(1, comparison) * diff[:, None]
        # comparison
        (gT2, gM2, gb2), (delta_l, delta_r) = \
            tensorGrad ((l,r), (T2, M2, b2), delta, nld, comparison)
        # composition
        ((gTl, gMl, gbl), gWl) = left_tree.grad (delta_l, l, ((T, M, b), W), nld)
        ((gTr, gMr, gbr), gWr) = right_tree.grad(delta_r, r, ((T, M, b), W), nld)
        ((gT, gM, gb), gW) = ((gTl + gTr, gMl + gMr, gbl + gbr), gWl + gWr)

        ## Pack them into a vector
        theta_grad = np.hstack(g.flat for g in (gS, gT2, gM2, gb2, gT, gM, gb, gW.todense()))
        
        return cost, theta_grad, np.argmax(softmax)

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

    net = Net(hyp)

    l = Leaf(0)
    r = Tree(Tree(Leaf(0), Leaf(0)), Leaf(0))

    # cost, grad, prediction = net.cost_and_grad(l,r,2)
    # param.grad(grad)

    print net.cost_and_grad(l,r,2)

