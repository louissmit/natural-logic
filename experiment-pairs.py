import numpy as np
from dataset import DataSet
from entailmentnet import HyperParameters, Net
from tensortree import Leaf

data = DataSet('wordpairs-v2.tsv')

relu  = np.vectorize(lambda x: max(0.,x))
relud = np.vectorize(lambda x: float(x>0))

hyp = HyperParameters()
hyp.vocab_size = len(data.vocab)
hyp.word_size = 16
hyp.comparison_size = 20
hyp.classes = len(data.rels)
hyp.composition_transfer = relu
hyp.composition_backtrans = relud
hyp.comparison_transfer = relu
hyp.comparison_backtrans = relud
hyp.batch_size = 100

net = Net(hyp)
print net.dims
pairs = [(Leaf(w1), Leaf(w2), r) for (r,w1,w2) in data.pairs]
net.adaGrad(pairs)