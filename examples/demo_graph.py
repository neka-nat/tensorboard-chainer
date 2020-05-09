import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from datetime import datetime
from tb_chainer import SummaryWriter, name_scope, within_name_scope

np.random.seed(123)

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    @within_name_scope('MLP')
    def __call__(self, x):
        with name_scope('linear1', self.l1.params()):
            h1 = F.relu(self.l1(x))
        with name_scope('linear2', self.l2.params()):
            h2 = F.relu(self.l2(h1))
        with name_scope('linear3', self.l3.params()):
            o = self.l3(h2)
        return o

model = L.Classifier(MLP(1000, 10))

res = model(chainer.Variable(np.random.rand(1, 784).astype(np.float32)),
            chainer.Variable(np.random.rand(1).astype(np.int32)))

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
writer.add_graph([res])
writer.add_all_variable_images([res], pattern='.*MLP.*')
writer.add_all_parameter_histograms([res], pattern='.*MLP.*')

writer.close()
