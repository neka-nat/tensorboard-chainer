def test_demo():
    import demo

#def test_demo_graph():
    #import demo_graph

#def test_demo_embedding():
    #import demo_embedding

def test_name_scope():
    import chainer
    import numpy as np
    from tensorboard import name_scope
    with name_scope("test"):
        x = chainer.Variable(np.zeros((10, 10)))
        y = chainer.functions.activation.leaky_relu.LeakyReLU()(x)

    assert y.creator.name_scope == "test"
    assert y.node.name_scope == "test"
    assert x.node.name_scope == "test"
