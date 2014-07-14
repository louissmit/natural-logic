import numpy as np
from dataset import DataSet
from entailmentnet import HyperParameters, Net
from tensortree import Leaf

data = DataSet('data-sick/vocab', relations_file='data-sick/vocab-relations')
data.load_sents('data-sick')



relu = lambda x: np.maximum(x, np.zeros(x.shape) + 0.01 * np.minimum(x, np.zeros(x.shape)))
relud = lambda x: (x >= 0).astype(float) + 0.01 * (x < 0).astype(float)
tanh  = lambda x: np.tanh(x)
tanhd = lambda x: (1-(np.tanh(x)**2))

hyp = HyperParameters()
hyp.vocab_size = len(data.vocab)
hyp.word_size = 32
hyp.comparison_size = 40
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
datapoints = list(data.sample_balanced('SICK', 1000))
print len(datapoints), 'data points'
net.adaGrad(datapoints)