
class IterativeOptimizerBase(object):
    # Note to implementers:
    # inherit from this object and implement __call__.
    # See sgd.py for example.

    """
    :param parameters: list or tuple of Theano variables 
        that we want to optimize iteratively.  If we are minimizing f(x), then
        together, these variables represent 'x'.  Typically these are shared
        variables and their values are the initial values for the minimization
        algorithm.

    :param cost: scalar-valued Theano variable that computes an exact or noisy
        estimate of cost (what are the conditions on the noise?).
        Some algorithms might need an exact cost, some algorithms might ignore
        the cost if the gradients are given.

    :param gradients: list or tuple of Theano variables representing the
        gradients on the corresponding parameters.  These default to
        tensor.grad(cost, parameters).

    :param stop: a shared variable (scalar integer) that (if provided) will be
        updated to say when the iterative minimization algorithm has finished
        (1) or requires more iterations (0).

    :param updates: a dictionary to update with the (var, new_value) items
        associated with the iterative algorithm.  The default is a new empty
        dictionary.  A KeyError is raised in case of key collisions.

    :param kwargs: algorithm-dependent arguments

    :returns: a dictionary mapping each parameter to an expression that it
       should take in order to carry out the optimization procedure.

       If all the parameters are shared variables, then this dictionary may be
       passed as the ``updates`` argument to theano.function.

       There may be more key,value pairs in the dictionary corresponding to
       internal variables that are part of the optimization algorithm.

    """


def dict_merge(a, b):
    """
    Return the union of dictionaries `a` and `b`.
    
    This function will raise a KeyError if `a` and `b` map any one key to
    different objects (as determined by __eq__).

    This function does not modify `a` nor `b`, but will return something like
    a shallow copy of each.
    """
    r = dict(a)
    for (k, v) in b.iteritems():
        if k in a and v != a[k]:
            raise KeyError(k)
        r[k] = v
    return r
