import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from datetime import datetime
from tensorboard import SummaryWriter

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = L.Classifier(MLP(1000, 10))

res = model(chainer.Variable(np.random.rand(1, 784).astype(np.float32)), chainer.Variable(np.random.rand(1).astype(np.int32)))

writer = SummaryWriter('runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))
writer.add_graph(model, res)

writer.close()
