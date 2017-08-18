from .src.graph_pb2 import GraphDef
from .src.node_def_pb2 import NodeDef
from .src.versions_pb2 import VersionDef
from .src.attr_value_pb2 import AttrValue
from .src.tensor_shape_pb2 import TensorShapeProto
from .src import types_pb2 as dt
from collections import defaultdict
import numpy as np
import chainer.variable
import chainer.computational_graph as c
global _id2name
global _type2ids

def convert_dtype(dtype):
    if dtype == np.float32:
        return dt.DT_FLOAT
    elif dtype == np.float64:
        return dt.DT_DOUBLE
    elif dtype == np.int32:
        return dt.DT_INT32
    elif dtype == np.uint8:
        return dt.DT_UINT8
    elif dtype == np.int16:
        return dt.DT_INT16
    elif dtype == np.int8:
        return dt.DT_INT8
    elif dtype == np.dtype('S1'):
        return dt.DT_STRING
    else:
        raise ValueError('Unsupported type.')

def add_id(obj):
    if not id(obj) in _type2ids[type(obj)]:
        _type2ids[type(obj)].append(id(obj))

def is_intermidiate_variable(obj):
    if isinstance(obj, chainer.variable.VariableNode) and \
      hasattr(obj, 'creator') and obj.creator is not None:
        return True
    return False

def make_name(obj, max_rank):
    target = obj
    if obj.rank < max_rank and is_intermidiate_variable(obj):
        target = obj.creator
    if hasattr(target, '_variable') and target._variable is not None:
        if id(target._variable()) in _id2name:
            return 'Parameter_' + _id2name[id(target._variable())]
    add_id(target)
    if isinstance(target, chainer.variable.VariableNode):
        return 'Valiable_' + str(_type2ids[type(target)].index(id(target))) + ' ' + target.label
    return target.label + '_' + str(_type2ids[type(target)].index(id(target)))

def make_list_of_nodes(fn):
    list_of_nodes = []
    g = c.build_computational_graph([fn])
    max_rank = max([n.rank for n in g.nodes])
    for n in g.nodes:
        if n.rank < max_rank and is_intermidiate_variable(n):
            continue
        inputs = []
        for e1, e2 in g.edges:
            if e2 == n:
                inputs.append(make_name(e1, max_rank))
        attr_shape = []
        if hasattr(n, 'shape'):
            attr_shape = list(n.shape)
        dtype = dt.DT_INVALID
        if hasattr(n, 'dtype'):
            dtype = convert_dtype(n.dtype)
        list_of_nodes.append({'name': make_name(n, max_rank), 'op': n.label,
                              'inputs': inputs,
                              'attr.shape': attr_shape,
                              'attr.dtype': dtype})
    return list_of_nodes

def make_attr(shape, dtype):
    dim_list = [TensorShapeProto.Dim(size=s) for s in shape]
    if len(dim_list) == 0:
        return None
    return {'shape': AttrValue(shape=TensorShapeProto(dim=dim_list)),
            'dtype': AttrValue(type=dtype)}

def graph(model, lastVar):
    global _id2name
    global _type2ids
    nodes = []
    _type2ids = defaultdict(list)
    _id2name = {id(m):n[1:].replace('/', '.') for n, m in model.namedparams()}
    list_of_nodes = make_list_of_nodes(lastVar)
    for node in list_of_nodes:
        nodes.append(NodeDef(name=node['name'], op=node['op'],
                             input=node['inputs'],
                             attr=make_attr(node['attr.shape'], node['attr.dtype'])))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
