import numpy as np
from tensortree import Leaf, Tree, tensorLayer, tensorGrad

class HyperParameters(): pass

class Net():
    def __init__(self, hyp):
        self.hyp = hyp
        def ntn_dims(i,o):
            return ((o, i, i), (o, 2*i), (o,))

        # softmax, comparison, composition, vocabulary
        self.dims = ((hyp.classes, hyp.comparison_size+1),) \
            + ntn_dims(hyp.word_size, hyp.comparison_size) \
            + ntn_dims(hyp.word_size, hyp.word_size) \
            + ((hyp.vocab_size, hyp.word_size),)
        self.theta = np.random.uniform(-0.01, 0.01, np.sum(np.prod(d) for d in self.dims))

    def params(self):
        """ Grab the slices of theta and assemble them """
        i=0
        out = []
        for d in self.dims:
            j = np.prod(d)
            out.append( self.theta[i:i+j].reshape(*d) )
            i += j
        return tuple(out)

    # np.hstack(p.flat for p in par2.params())

    def soft_max(self, l, r, true_relation, comparison, params):
        def normalize(v):
            norm = np.linalg.norm(v)
            assert not np.isinf(norm), 'softmax is too big'
            return v / norm if norm else v

        (S, T2, M2, b2, T, M, b, W) = params

        softmax = normalize(np.exp(S.dot(np.append(1, comparison))))
        cost = -np.log(softmax[true_relation])
        return softmax, cost

    def predict(self, left_tree, right_tree):
        nl  = self.hyp.composition_transfer
        nld = self.hyp.composition_backtrans
        nl2  = self.hyp.comparison_transfer
        nld2 = self.hyp.comparison_backtrans
        params = self.params()
        (S, T2, M2, b2, T, M, b, W) = params

        ## Run forward
        l = left_tree.do (((T, M, b), W), nl)
        r = right_tree.do(((T, M, b), W), nl)

        comparison = tensorLayer((l,r), (T2, M2, b2), nl2)

        softmax = self.soft_max(l, r, 0, comparison, params)[0]
        return np.argmax(softmax)

    def cost_and_grad(s, left_tree, right_tree, true_relation):
        nl  = s.hyp.composition_transfer
        nld = s.hyp.composition_backtrans
        nl2  = s.hyp.comparison_transfer
        nld2 = s.hyp.comparison_backtrans
        params = s.params()
        (S, T2, M2, b2, T, M, b, W) = params

        ## Run forward
        l = left_tree.do (((T, M, b), W), nl)
        r = right_tree.do(((T, M, b), W), nl)

        comparison = tensorLayer((l,r), (T2, M2, b2), nl2)

        softmax, cost = s.soft_max(l, r, true_relation, comparison, params)

        ## Get gradients
        # softmax
        diff = softmax - np.eye(s.hyp.classes)[true_relation]
        delta = ( nld2(np.append(1, comparison)) * S.T.dot(diff) )[1:]
        gS = np.append(1, comparison) * diff[:, None] # outer product?
        # comparison
        (gT2, gM2, gb2), (delta_l, delta_r) = \
            tensorGrad ((l,r), (T2, M2, b2), delta, nld2, comparison)
        # composition
        ((gTl, gMl, gbl), gWl) = left_tree.grad (delta_l, l, ((T, M, b), W), nld)
        ((gTr, gMr, gbr), gWr) = right_tree.grad(delta_r, r, ((T, M, b), W), nld)
        ((gT, gM, gb), gW) = ((gTl + gTr, gMl + gMr, gbl + gbr), gWl + gWr)

        ## Pack them into a vector
        theta_grad = np.hstack(g.flat for g in (gS, gT2, gM2, gb2, gT, gM, gb, gW.todense()))
        
        return cost, theta_grad, np.argmax(softmax)

    def adaGrad(self, data):
        # http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        train_indices = np.arange(int(0.85 * len(data)))
        test_indices = np.arange(int(0.85 * len(data)), len(data))
        master_stepsize = 0.2 # for example
        fudge_factor = 0.001 # for numerical stability
        historical_grad = np.zeros(self.theta.size)
        historical_cost = float('inf')
        converged = False
        test_frequency = 5
        iters = 0

        while not converged:
            if iters % test_frequency == 0:
                test_correct, test_total = 0., 0.
                for (left_tree, right_tree, true_relation) in [data[i] for i in test_indices]:
                    pred = self.predict(left_tree, right_tree)
                    test_correct += int(pred == true_relation)
                    test_total += 1
                print 'test accuracy: ', test_correct / test_total

            np.random.shuffle(train_indices)
            batch = train_indices[:self.hyp.batch_size]
            # Get gradient of batch
            grad = np.zeros(self.theta.size)
            cost, correct, total = 0., 0., 0.
            for (left_tree, right_tree, true_relation) in [data[i] for i in batch]:
                c, g, pred = self.cost_and_grad(left_tree, right_tree, true_relation)
                grad += g
                cost += c
                correct += int(pred == true_relation)
                total += 1
            # Normalize, Regularize
            cost = cost / self.hyp.batch_size
            cost += self.hyp.l2_lambda/2 * np.sum(np.square(self.theta))
            grad = grad / self.hyp.batch_size
            grad += self.hyp.l2_lambda * self.theta

            print (correct/total), cost, grad
            converged = abs(historical_cost - cost) < 1e-8
            historical_cost = cost

            # Perform adaGrad update
            historical_grad += np.square(grad)
            adjusted_grad = grad / (fudge_factor + np.sqrt(historical_grad))
            self.theta -= master_stepsize * adjusted_grad
            iters += 1


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
    hyp.l2_lambda = 0.0002

    net = Net(hyp)

    l = Leaf(0)
    r = Tree(Tree(Leaf(0), Leaf(0)), Leaf(0))

    # cost, grad, prediction = net.cost_and_grad(l,r,2)
    # param.grad(grad)

    print net.cost_and_grad(l,r,2)

