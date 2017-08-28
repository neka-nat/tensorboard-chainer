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

def add_id(dic, name, obj):
    if not id(obj) in dic[name]:
        dic[name].append(id(obj))

def base_name(obj):
    name_scope = (obj.name_scope + '/') if hasattr(obj, 'name_scope') else ''
    if hasattr(obj, '_variable') and obj._variable is not None:
        if isinstance(obj._variable(), chainer.Parameter):
            return name_scope + (('Parameter_' + obj.name) if obj.name is not None else 'Parameter')
    if isinstance(obj, chainer.variable.VariableNode):
        return name_scope + 'Variable_' + obj.label
    return name_scope + obj.label

def make_name(obj, dic):
    bn = base_name(obj)
    if len(dic[bn]) == 1:
        return bn
    else:
        return bn + '_' + str(dic[bn].index(id(obj)))

def make_list_of_nodes(fn):
    list_of_nodes = []
    g = c.build_computational_graph([fn])
    type2ids = defaultdict(list)
    for n in g.nodes:
        add_id(type2ids, base_name(n), n)
    for n in g.nodes:
        inputs = []
        for e1, e2 in g.edges:
            if e2 == n:
                inputs.append(make_name(e1, type2ids))
        attr_shape = []
        if hasattr(n, 'shape'):
            attr_shape = list(n.shape)
        dtype = dt.DT_INVALID
        if hasattr(n, 'dtype'):
            dtype = convert_dtype(n.dtype)
        list_of_nodes.append({'name': make_name(n, type2ids),
                              'op': n.__class__.__name__,
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

def graph(lastVar):
    nodes = []
    list_of_nodes = make_list_of_nodes(lastVar)
    for node in list_of_nodes:
        nodes.append(NodeDef(name=node['name'], op=node['op'],
                             input=node['inputs'],
                             attr=make_attr(node['attr.shape'], node['attr.dtype'])))
    return GraphDef(node=nodes, versions=VersionDef(producer=22))
