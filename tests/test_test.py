def test_demo():
    from examples import demo

def test_demo_graph():
    from examples import demo_graph

#def test_demo_embedding():
    #from examples import demo_embedding

def test_name_scope():
    import chainer
    import numpy as np
    from tb_chainer import name_scope
    with name_scope("test"):
        x = chainer.Variable(np.zeros((10, 10)))
        y = chainer.functions.activation.leaky_relu.leaky_relu(x)

    assert y.creator.name_scope == "test"
    assert y.node.name_scope == "test"
    assert x.node.name_scope == "test"
