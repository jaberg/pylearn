import numpy
import scipy.optimize
#import scipy.linalg
import theano
import theano.tensor as TT
from theano_linalg import solve, cholesky, diag, matrix_inverse, det

#TODO: Match names to scikits.learn

def dots(*args):
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval

class SquaredExponentialKernel(object):
    """

    K(x,y) = exp(-0.5 ||x-y||^2 / l^2)

    Attributes:

        log_lenscale - log(2 l^2)
        lenscale - 2 l^2

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(2*(l**2))
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def params(self):
        return [self.log_lenscale]

    def K(self, x, y):
        l = TT.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * TT.dot(x, y.T))
        K = TT.exp(-d/l)
        return K

class ExponentialKernel(object):
    """
    K(x,y) = exp(- ||x-y|| / l)

    Attributes:

        log_lenscale - log(l)

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(l)
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def params(self):
        return [self.log_lenscale]

    def K(self, x, y):
        l = TT.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * TT.dot(x, y.T))
        K = TT.exp(-TT.sqrt(d)/l)
        return K

class CategoryKernel(object):
    """
    K(x,y) is 1 if x==y else c

    Attributes:
        
        raw_c - 
        c - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def params(self):
        return [self.raw_c]

class ConvexMixtureKernel(object):
    """

    Attributes:
    
        kernels -
        raw_coefs - 
        coefs - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def alloc(cls, kernels, coefs=None):
        if coefs is None:
            raw_coefs = theano.shared(numpy.zeros(len(kernels)))
        else:
            raise NotImplementedError()
        return cls(
                kernels=kernels,
                coefs=TT.nnet.softmax(raw_coefs),
                raw_coefs = raw_coefs,
                )

    def params(self):
        rval = [self.raw_coefs]
        for k in self.kernels():
            rval.extend(k.params())
        return rval

    def K(self, x, y):
        # get the kernel matrix from each sub-kernel
        # stack them up
        # multiply by coefs
        # sum down to one kernel
        # return
        pass
    def K_each_pair(self, x, y):
        # get the kernel value from each sub-kernel
        # stack them up
        # multiply by coefs
        # sum down to one kernel
        # return
        pass


#
# To move into a NEW FILE pylearn/gp/regression.py
#

class GPR_math(object):
    """
    Formulae for Gaussian Process Regression

    """
    def __init__(self, kernel, x, y, var_y):
        self.kernel = kernel
        self.x = TT.as_tensor_variable(x)
        self.y = TT.as_tensor_variable(y)
        self.var_y = TT.as_tensor_variable(var_y)

    def s_nll(self):
        """
        Marginal negative log likelihood of model

        K - gram matrix (matrix-like)
        y - the training targets (vector-like)
        var_y - the variance of uncertainty about y (vector-like)

        :note: See RW.pdf page 37, Eq. 2.30.

        """

        y = self.y
        n = y.shape[0]
        K = self.kernel.K(self.x, self.x)
        rK = K + self.var_y * TT.eye(self.y.shape[0])

        nll = ( 0.5 * dots(y, matrix_inverse(rK), y)
            + 0.5 * TT.log(det(rK)) 
            + n / 2.0 * TT.log(2*numpy.pi))
        return nll

    def s_mean(self, x):
        K = self.kernel.K(self.x, self.x)
        rK = K + self.var_y * TT.eye(self.y.shape[0])
        alpha = TT.dot(matrix_inverse(rK), self.y)

        K_x = self.kernel.K(self.x, x)
        y_x = TT.dot(alpha, K_x)
        return y_x

    def s_variance(self, x):
        K = self.kernel.K(self.x, self.x)
        rK = K + self.var_y * TT.eye(self.y.shape[0])
        L = cholesky(rK)
        K_x = self.kernel.K(self.x, x)
        v = solve(L, K_x)
        var_x = 1 - (v**2).sum(axis=0)
        return var_x

    def minimize_nll(self):
        """
        Fit GPR kernel parameters by minimizing magininal nll.

        Returns: None

        Side effect: chooses optimal kernel parameters.
        """

        nll = theano.function([], self.s_nll())
        dnll_dparams = theano.function([], TT.grad(self.s_nll(), self.kernel.params()))

        def get_pt():
            #TODO: handle non-scalar parameters...
            return numpy.asarray([p.value for p in self.kernel.params()])
        def set_pt(pt):
            params = self.kernel.params()
            assert len(pt) == len(params)
            for pt_i, p_i in zip(pt, params):
                p_i.value = pt_i
        def f(pt):
            set_pt(pt)
            return nll()
        def df(pt):
            set_pt(pt)
            dparams = dnll_dparams()
            return numpy.asarray(dparams)
        start_pt = get_pt()
        best_pt = scipy.optimize.fmin_ncg(f, start_pt, df)
        set_pt(best_pt)

    def mean(self, x):
        """
        Compute mean at points in x_new
        """
        try:
            self._mean
        except AttributeError:
            s_x = TT.matrix()
            self._mean = theano.function([s_x], self.s_mean(s_x))
        return self._mean(x)

    def variance(self, x):
        """
        Compute variance at points in x_new
        """
        try:
            self._variance
        except AttributeError:
            s_x = TT.matrix()
            self._variance = theano.function([s_x], self.s_variance(s_x))
        return self._variance(x)

    def mean_variance(self, x):
        """
        Compute mean and variance at points in x_new
        """
        try:
            self._mean_variance
        except AttributeError:
            s_x = TT.matrix()
            self._mean_variance = theano.function([s_x],
                    [self.s_mean(s_x), self.s_variance(s_x)])
        return self._mean_variance(x)

