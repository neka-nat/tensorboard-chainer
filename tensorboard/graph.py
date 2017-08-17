from .src.graph_pb2 import GraphDef
from .src.node_def_pb2 import NodeDef
from .src.versions_pb2 import VersionDef
from .src.attr_value_pb2 import AttrValue
from .src.tensor_shape_pb2 import TensorShapeProto
from collections import defaultdict
import chainer.variable
import chainer.computational_graph as c
global _id2name
global _type2ids

def add_id(obj):
    if not id(obj) in _type2ids[type(obj)]:
        _type2ids[type(obj)].append(id(obj))

def make_name(obj):
    if hasattr(obj, '_variable') and obj._variable is not None:
        if id(obj._variable()) in _id2name:
            return 'Parameter_' + _id2name[id(obj._variable())]
    add_id(obj)
    if isinstance(obj, chainer.variable.VariableNode):
        return 'Valiable_' + str(_type2ids[type(obj)].index(id(obj))) + ' ' + obj.label
    return obj.label + '_' + str(_type2ids[type(obj)].index(id(obj)))

def make_list_of_nodes(fn):
    list_of_nodes = []
    g = c.build_computational_graph([fn])
    for n in g.nodes:
        inputs = []
        for e1, e2 in g.edges:
            if e2 == n:
                inputs.append(make_name(e1))
        attr_shape = []
        if hasattr(n, 'shape'):
            attr_shape = list(n.shape)
        list_of_nodes.append({'name':make_name(n), 'op':n.label,
                              'inputs':inputs, 'attr.shape':attr_shape})
    return list_of_nodes

def graph(model, lastVar):
    global _id2name
    global _type2ids
    nodes = []
    _type2ids = defaultdict(list)
    _id2name = {id(m):n[1:].replace('/', '.') for n, m in model.namedparams()}
    list_of_nodes = make_list_of_nodes(lastVar)
    for node in list_of_nodes:
        shape_str = str(node['attr.shape']).encode(encoding='utf_8')
        nodes.append(NodeDef(name=node['name'], op=node['op'], input=node['inputs'], attr={'shape':AttrValue(s=shape_str)}))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
