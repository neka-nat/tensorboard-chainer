from chainer import function
from chainer import variable
import functools
from types import MethodType

def _init_with_name_scope(self, *args, **kargs):
    self.name_scope = kargs['name_scope']
    org_init = kargs['org_init']
    del kargs['name_scope']
    del kargs['org_init']
    org_init(self, *args, **kargs)

_org_func_init = function.Function.__init__
_org_val_init = variable.VariableNode.__init__

class name_scope(object):
    """Class that creates hierarchical names for operations and variables.
    Args:
        name (str): Name for setting namespace.
        values (list): Variable in the namespace.
    Example:
        You can set namespace using "with" statement.
        In the following example, no namespace is set for the variable 'X', but
        the variable 'Y' and the relu function are set to the namespace "test".

            x = chainer.Variable(...)
            with name_scope('test'):
               y = F.relu(x)
    """
    stack = []
    def __init__(self, name, values=[]):
        self.stack.append(name)
        for v in values:
            v.node.name_scope = '/'.join(self.stack)

    def __enter__(self):
        self._org_func_init = function.Function.__init__
        function.Function.__init__ = MethodType(functools.partial(_init_with_name_scope,
                                                                  name_scope='/'.join(self.stack),
                                                                  org_init=_org_func_init),
                                                None, function.Function)
        self._org_val_init = variable.VariableNode.__init__
        variable.VariableNode.__init__ = MethodType(functools.partial(_init_with_name_scope,
                                                                      name_scope='/'.join(self.stack),
                                                                      org_init=_org_val_init),
                                                    None, variable.VariableNode)
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        function.Function.__init__ = self._org_func_init
        variable.VariableNode.__init__ = self._org_val_init
        self.stack.pop(-1)
