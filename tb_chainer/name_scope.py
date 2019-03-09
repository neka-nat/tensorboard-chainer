import sys
import inspect
import chainer
from chainer import function_node
from chainer import variable
import functools
from types import MethodType
if sys.version_info >= (3, 0):
    def method_wraper(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return functools.wraps(f)(wrapper)
    gen_method = lambda m, i, c: method_wraper(m)
else:
    gen_method = lambda m, i, c: MethodType(m, i, c)

def _copy_method(f, c):
    g = gen_method(f, None, c)
    return g

def _init_with_name_scope(self, *args, **kargs):
    self.name_scope = kargs['_name_scope']
    org_init = kargs['_org_init']
    retain_data = kargs['_retain_data']
    del kargs['_name_scope']
    del kargs['_org_init']
    del kargs['_retain_data']
    org_init(self, *args, **kargs)
    if retain_data and isinstance(self, variable.VariableNode):
        self.retain_data()

def _new_with_name_scope(cls, *args, **kargs):
    ns = kargs['_name_scope']
    org_new = kargs['_org_new']
    self = org_new(cls)
    self.name_scope = ns
    return self

_org_classes = []
_copy_org_inits = []

def register_functions(funcs):
    """Register function nodes to use name_scope.
    Args:
        funcs (list): List of function nodes
    """
    global _org_classes, _copy_org_inits
    _org_classes.extend(funcs)
    for c in _org_classes:
        if c == variable.VariableNode:
            _copy_org_inits.append(_copy_method(c.__init__, c))
        else:
            _copy_org_inits.append(c.__new__)

register_functions([function_node.FunctionNode, variable.VariableNode])

class name_scope(object):
    """Class that creates hierarchical names for operations and variables.
    Args:
        name (str): Name for setting namespace.
        values (list): Variable in the namespace.
        retain_data (bool): Hold the data in the variable.
    Example:
        You can set namespace using "with" statement.
        In the following example, no namespace is set for the variable 'X', but
        the variable 'Y' and the relu function are set to the namespace "test".

            x = chainer.Variable(...)
            with name_scope('test'):
               y = F.relu(x)
    """
    stack = []
    def __init__(self, name, values=[], retain_data=True):
        self.stack.append(name)
        self._org_inits = []
        self._retain_data = retain_data
        for v in values:
            v.node.name_scope = '/'.join(self.stack)

    def __enter__(self):
        for idx, c in enumerate(_org_classes):
            if c == variable.VariableNode:
                self._org_inits.append(c.__init__)
                c.__init__ = gen_method(functools.partial(_init_with_name_scope,
                                                          _name_scope='/'.join(self.stack),
                                                          _org_init=_copy_org_inits[idx],
                                                          _retain_data=self._retain_data),
                                        None, c)
            else:
                self._org_inits.append(c.__new__)
                c.__new__ = classmethod(functools.partial(_new_with_name_scope,
                                                          _name_scope='/'.join(self.stack),
                                                          _org_new=_copy_org_inits[idx]))
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        for idx, c in enumerate(_org_classes):
            if c == variable.VariableNode:
                c.__init__ = self._org_inits[idx]
            else:
                c.__new__ = classmethod(functools.partial(_new_with_name_scope,
                                                          _name_scope='/'.join(self.stack[:-1]),
                                                          _org_new=_copy_org_inits[idx]))
        self.stack.pop(-1)

def within_name_scope(name, retain_data=True):
    """Decorator for link class methods.
    Args:
        name (str): Name for setting namespace.
        retain_data (bool): Hold the data in the variable.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            with name_scope(name, self.params(), retain_data=retain_data):
                res = func(self, *args, **kwargs)
            return res
        return wrapper
    return decorator
