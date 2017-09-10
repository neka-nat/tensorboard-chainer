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

class NodeName:
    """Class that creates the node's name from the list of nodes on the network.
    Give unique names to unique nodes on the network.
    Attributes:
        name_to_id :A dictionary in which the key is the object name and the value
                    is list of the object IDs.
    """
    def __init__(self, nodes):
        self.name_to_id = defaultdict(list)
        for n in nodes:
            name = NodeName.base_name(n)
            if not id(n) in self.name_to_id[name]:
                self.name_to_id[name].append(id(n))

    @staticmethod
    def base_name(obj):
        name_scope = (obj.name_scope + '/') if hasattr(obj, 'name_scope') else ''
        if hasattr(obj, '_variable') and obj._variable is not None:
            if isinstance(obj._variable(), chainer.Parameter):
                return name_scope + (('Parameter_' + obj.name) if obj.name is not None else 'Parameter')
        if isinstance(obj, chainer.variable.VariableNode):
            return name_scope + 'Variable_' + obj.label
        return name_scope + obj.label

    def name(self, obj):
        """Return the name of the object.
        Args:
            obj :A object on the network
        """
        bn = NodeName.base_name(obj)
        if len(self.name_to_id[bn]) == 1:
            return bn
        else:
            return bn + '_' + str(self.name_to_id[bn].index(id(obj)))

def make_list_of_nodes(fn):
    list_of_nodes = []
    g = c.build_computational_graph(fn)
    node_name = NodeName(g.nodes)
    for n in g.nodes:
        inputs = []
        for e1, e2 in g.edges:
            if e2 == n:
                inputs.append(node_name.name(e1))
        attr_shape = []
        if hasattr(n, 'shape'):
            attr_shape = list(n.shape)
        dtype = dt.DT_INVALID
        if hasattr(n, 'dtype'):
            dtype = convert_dtype(n.dtype)
        list_of_nodes.append({'name': node_name.name(n),
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
