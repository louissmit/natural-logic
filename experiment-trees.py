import numpy as np
from dataset import DataSet
from entailmentnet import HyperParameters, Net
from tensortree import Leaf

data = DataSet('wordpairs-v2.tsv')
data.load_sents('data-4')

relu  = np.vectorize(lambda x: max(0.,x) + 0.01 * min(0.,x))
relud = np.vectorize(lambda x: float(x>=0) + 0.01 * float(x<0))
tanh  = np.vectorize(lambda x: np.tanh(x))
tanhd = np.vectorize(lambda x: (1-(np.tanh(x)**2)))

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

datapoints = [datapoint for dataset in data.sets.values() for datapoint in dataset]
print len(datapoints), 'data points'
net.adaGrad(datapoints)