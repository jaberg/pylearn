from itertools import izip

from theano import tensor

from .base import IterativeOptimizerBase, dict_merge

class Basic(IterativeOptimizerBase):
    """
    The basic stochastic gradient descent algorithm.

    #TODO: how to mark this up?
    p_i \rightarrow p_i - \lambda_i \frac{d cost}{d p_i}

    The $p_i$ are the ``parameters``, the ``lambda_i`` are the step_size[s] and 
    the gradients $\frac{d cost}{d p_i}$ may be specified explicitly.
    """
    def __call__(self, parameters, 
            cost,
            gradients=None,
            stop=None, 
            updates=None,
            step_size=None):
        # DOCSTRING IS DEFINED BELOW

        if step_size is None:
            raise TypeError('step_size must be scalar-like or a list of scalar-like things', step_size)

        if gradients is None:
            gradients = tensor.grad(cost, parameters)

        if len(gradients) != len(parameters):
            raise ValueError(('number of gradients (%i)'
                ' does not match number of parameters (%i)')%(
                    len(gradients), len(parameters)))

        try:
            l_step_size = len(step_size)
        except: #TODO: be precise here
            l_step_size = None

        if l_step_size is None:
            step_size = [step_size for p in parameters]
        elif len(step_size) != len(parameters):
            raise ValueError('number of step_sizes (%i)'
                             ' does not match number of parameters (%i)' %
                             (len(step_size), len(parameters)))

        learning_updates = dict(
                [(p, p - s * g) 
                    for (p, g, s) in izip(parameters, gradients, step_size)])

        if updates:
            rval = dict_merge(learning_updates, updates)
        else:
            rval = learning_updates

        return rval
    __call__.__doc__ = IterativeOptimizerBase.__doc__ + """
    :param step_size: scalar-like step size to be used for all parameters, or a list of
        scalar-like step sizes to be used for each parameter.
    """
basic = Basic()
#TODO: 
# - SGD with stopping heuristics
# - SGD with momentum
# - SGD with annealed step_size

