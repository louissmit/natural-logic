import numpy as np
from dataset import DataSet
from entailmentnet import HyperParameters, Net
from tensortree import Leaf

data = DataSet('wordpairs-v2.tsv')

# relu  = np.vectorize(lambda x: max(0.,x) + 0.01 * min(0.,x))
relu = lambda x: np.maximum(x, np.zeros(x.shape) + 0.01 * np.minimum(x, np.zeros(x.shape)))
# relud = np.vectorize(lambda x: float(x>=0) + 0.01 * float(x<0))
relud = lambda x: (x >= 0).astype(float) + 0.01 * (x < 0).astype(float)
tanh  = lambda x: np.tanh(x)
tanhd = lambda x: (1-(np.tanh(x)**2))
hyp = HyperParameters()
hyp.vocab_size = len(data.vocab)
hyp.word_size = 16
hyp.comparison_size = 20
hyp.classes = len(data.rels)
hyp.composition_transfer = tanh
hyp.composition_backtrans = tanhd
hyp.comparison_transfer = relu
hyp.comparison_backtrans = relud
hyp.batch_size = 100
hyp.l2_lambda = 0.0002

net = Net(hyp)
print net.dims
pairs = [(Leaf(w1), Leaf(w2), r) for (r,w1,w2) in data.pairs]
net.adaGrad(pairs)