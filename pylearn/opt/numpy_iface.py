class minimize_iterator(object):
    """
    Attributes
     - x  - the current best estimate of the minimum
     - f  - the function being minimized
     - df - f's derivative function
     - opt_algo - the optimization algorithm at work (a serializable, callable
       object with the signature of iterative_optimizer above).

    """
    def __init__(self, x0, f, df, opt_algo, **kwargs):
        """Initialize state (arguments as in minimize()) """
        self.x0 = x0
        self.x = x0.copy()
        self.f = f
        self.df = df
        self.opt_algo = opt_algo
        self.kwargs = kwargs
        raise NotImplementedError()
    def __iter__(self):
        return self
    def next(self):
        """Take a step of minimization and return self raises StopIteration when
        the algorithm is finished with minimization

        """
        raise NotImplementedError()

def minimize(x0, f, df, opt_algo, **kwargs):
    """
    Return a point x_new with the same type as x0 that minimizes function `f`
    with derivative `df`.

    This is supposed to provide an interface similar to scipy's minimize
    routines, or MATLAB's.

    :type x0: numpy ndarray or list of numpy ndarrays.
    :param x0: starting point for minimization

    :type f: python callable mapping something like x0 to a scalar
    :param f: function to minimize

    :type df: python callable mapping something like x0 to the derivative of f at that point
    :param df: derivative of `f`

    :param opt_algo: one of the functions that implements the
    `iterative_optimizer` interface.

    :param kwargs: passed through to `opt_algo`

    """

    it = minimize_iterator(x0, f, df, opt_algo, **kwargs)
    for ii in it:
        pass
    return it.x
