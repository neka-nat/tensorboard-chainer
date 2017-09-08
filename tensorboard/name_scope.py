import sys
import chainer
from chainer import function
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

def _copy_method(c):
    g = gen_method(c.__init__, None, c)
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

_org_classes = [function.Function,
                chainer.functions.activation.clipped_relu.ClippedReLU,
                chainer.functions.activation.crelu.CReLU,
                chainer.functions.activation.elu.ELU,
                chainer.functions.activation.leaky_relu.LeakyReLU,
                chainer.functions.array.concat.Concat,
                chainer.functions.array.copy.Copy,
                chainer.functions.array.expand_dims.ExpandDims,
                chainer.functions.array.pad.Pad,
                chainer.functions.array.permutate.Permutate,
                chainer.functions.array.reshape.Reshape,
                chainer.functions.array.squeeze.Squeeze,
                chainer.functions.array.swapaxes.Swapaxes,
                chainer.functions.array.tile.Tile,
                chainer.functions.array.transpose.Transpose,
                chainer.functions.evaluation.accuracy.Accuracy,
                chainer.functions.evaluation.classification_summary.ClassificationSummary,
                chainer.functions.evaluation.r2_score.R2_score,
                chainer.functions.loss.contrastive.Contrastive,
                chainer.functions.loss.cross_covariance.CrossCovariance,
                chainer.functions.loss.ctc.ConnectionistTemporalClassification,
                chainer.functions.loss.decov.DeCov,
                chainer.functions.loss.hinge.Hinge,
                chainer.functions.loss.huber_loss.HuberLoss,
                chainer.functions.loss.negative_sampling.NegativeSamplingFunction,
                chainer.functions.loss.sigmoid_cross_entropy.SigmoidCrossEntropy,
                chainer.functions.loss.softmax_cross_entropy.SoftmaxCrossEntropy,
                chainer.functions.loss.triplet.Triplet,
                chainer.functions.math.basic_math.AddConstant,
                chainer.functions.math.basic_math.SubFromConstant,
                chainer.functions.math.basic_math.MulConstant,
                chainer.functions.math.basic_math.DivFromConstant,
                chainer.functions.math.basic_math.PowVarConst,
                chainer.functions.math.basic_math.PowConstVar,
                chainer.functions.math.basic_math.MatMulVarConst,
                chainer.functions.math.basic_math.MatMulConstVar,
                chainer.functions.math.sum.Sum,
                chainer.functions.connection.convolution_2d.Convolution2DFunction,
                chainer.functions.connection.convolution_nd.ConvolutionND,
                chainer.functions.connection.deconvolution_2d.Deconvolution2DFunction,
                chainer.functions.connection.deconvolution_nd.DeconvolutionND,
                chainer.functions.normalization.batch_normalization.BatchNormalizationFunction,
                chainer.functions.normalization.l2_normalization.NormalizeL2,
                chainer.functions.normalization.local_response_normalization.LocalResponseNormalization,
                chainer.functions.pooling.average_pooling_2d.AveragePooling2D,
                chainer.functions.pooling.average_pooling_nd.AveragePoolingND,
                chainer.functions.pooling.max_pooling_2d.MaxPooling2D,
                chainer.functions.pooling.max_pooling_nd.MaxPoolingND,
                chainer.functions.pooling.pooling_2d.Pooling2D,
                chainer.functions.pooling.pooling_nd._PoolingND,
                chainer.functions.pooling.unpooling_2d.Unpooling2D,
                chainer.functions.pooling.unpooling_nd.UnpoolingND,
                variable.VariableNode]
_copy_org_inits = [_copy_method(c) for c in _org_classes]

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
    def __init__(self, name, values=[], retain_data=False):
        self.stack.append(name)
        self._org_inits = []
        self._retain_data = retain_data
        for v in values:
            v.node.name_scope = '/'.join(self.stack)

    def __enter__(self):
        for idx, c in enumerate(_org_classes):
            self._org_inits.append(c.__init__)
            c.__init__ = gen_method(functools.partial(_init_with_name_scope,
                                                      _name_scope='/'.join(self.stack),
                                                      _org_init=_copy_org_inits[idx],
                                                      _retain_data=self._retain_data),
                                    None, c)
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        for idx, c in enumerate(_org_classes):
            c.__init__ = self._org_inits[idx]
        self.stack.pop(-1)

def within_name_scope(name, retain_data=False):
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
