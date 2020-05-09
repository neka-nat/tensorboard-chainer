from .src.graph_pb2 import GraphDef
from .src.node_def_pb2 import NodeDef
from .src.versions_pb2 import VersionDef
from .src.attr_value_pb2 import AttrValue
from .src.tensor_shape_pb2 import TensorShapeProto
from .src import types_pb2 as dt
from .ordered_set import OrderedSet
import heapq
from collections import defaultdict
import numpy as np
import chainer.variable
import chainer.function_node
import chainer.computational_graph as c


def build_computational_graph(
        outputs, remove_split=True, variable_style='default',
        function_style='default', rankdir='TB', remove_variable=False,
        show_name=True):
    """Builds a graph of functions and variables backward-reachable from outputs.
    Args:
        outputs (:class:`~chainer.Variable`, \
        :class:`~chainer.variable.VariableNode`, \
        :class:`~chainer.FunctionNode`, or :class:`list`): node(s) from which
            the graph is constructed.
            Each element of outputs must be either :class:`~chainer.Variable`
            object, :class:`~chainer.variable.VariableNode` object, or
            :class:`~chainer.FunctionNode` object.
        remove_split(bool): It must be ``True``. This argument is left for
            backward compatibility.
        variable_style(dict or 'default'): Dot node style for variable.
            Possible keys are 'shape', 'color', 'fillcolor', 'style' etc.
            If the special value ``'default'`` is specified, the default
            configuration will be used.
        function_style(dict or 'default'): Dot node style for function.
            Possible keys are 'shape', 'color', 'fillcolor', 'style' etc.
            If the special value ``'default'`` is specified, the default
            configuration will be used.
        rankdir (str): Direction of the graph that must be
            TB (top to bottom), BT (bottom to top), LR (left to right)
            or RL (right to left).
        remove_variable (bool): If ``True``, :class:`VariableNode`\\ s are
            removed from the resulting computational graph. Only
            :class:`FunctionNode`\\ s are shown in the output.
        show_name (bool): If ``True``, the ``name`` attribute of each node is
            added to the label of the node. Default is ``True``.
    Returns:
        ComputationalGraph: A graph consisting of nodes and edges that
        are backward-reachable from at least one of ``outputs``.
        If ``unchain_backward`` was called in some variable in the
        computational graph before this function, backward step is
        stopped at this variable.
        For example, suppose that computational graph is as follows::
                |--> f ---> y
            x --+
                |--> g ---> z
        Let ``outputs = [y, z]``.
        Then the full graph is emitted.
        Next, let ``outputs = [y]``. Note that ``z`` and ``g``
        are not backward-reachable from ``y``.
        The resulting graph would be following::
            x ---> f ---> y
        See :class:`TestGraphBuilder` for details.
    .. note::
       The default configuration for ``variable_style`` is
       ``{'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}`` and
       the default configuration for ``function_style`` is
       ``{'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}``.
    .. note::
        The default behavior of :class:`~chainer.ComputationalGraph` has been
        changed from v1.23.0, so that it ouputs the richest representation of
        a graph as default, namely, styles are set and names of functions and
        variables are shown. To reproduce the same result as previous versions
        (<= v1.22.0), please specify `variable_style=None`,
        `function_style=None`, and `show_name=False` explicitly.
    """
    if not remove_split:
        raise ValueError('remove_split=False is not supported anymore')

    output_types = (
        chainer.variable.Variable, chainer.variable.VariableNode,
        chainer.function_node.FunctionNode)

    if isinstance(outputs, output_types):
        outputs = [outputs]
    else:
        if not all(isinstance(o, output_types) for o in outputs):
            raise TypeError(
                'element of outputs must be either Variable, VariableNode, '
                ' or FunctionNode.')

    cands = []
    seen_edges = OrderedSet()
    nodes = OrderedSet()
    push_count = [0]

    def add_cand(cand):
        heapq.heappush(cands, (-cand.rank, push_count[0], cand))
        push_count[0] += 1

    for o in outputs:
        if isinstance(o, chainer.variable.Variable):
            o = o.node
        add_cand(o)
        nodes.add(o)

    while cands:
        _, _, cand = heapq.heappop(cands)
        if isinstance(cand, chainer.variable.VariableNode):
            creator = cand.creator_node
            if creator is not None and (creator, cand) not in seen_edges:
                add_cand(creator)
                seen_edges.add((creator, cand))
                nodes.add(creator)
                nodes.add(cand)
        elif isinstance(cand, chainer.function_node.FunctionNode):
            for input_ in cand.inputs:
                if input_ is not cand and (input_, cand) not in seen_edges:
                    add_cand(input_)
                    seen_edges.add((input_, cand))
                    nodes.add(input_)
                    nodes.add(cand)
    return c.ComputationalGraph(
        list(nodes), list(seen_edges), variable_style,
        function_style, rankdir, remove_variable, show_name)


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
    g = build_computational_graph(fn)
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
