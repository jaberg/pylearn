import numpy
import scipy.optimize
#import scipy.linalg
import theano
import theano.tensor as TT
from theano_linalg import solve, cholesky, diag, matrix_inverse, det, PSD_hint

#TODO: Match names to scikits.learn

def dots(*args):
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval

def value(x):
    try:
        return x.get_value()
    except AttributeError:
        return x

class SquaredExponentialKernel(object):
    """

    K(x,y) = exp(-0.5 ||x-y||^2 / l^2)

    Attributes:

        log_lenscale - log(2 l^2)

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)
    def lenscale(self, thing=None):
        if thing is None:
            thing = self.log_lenscale
        return numpy.sqrt(numpy.exp(value(thing)) / 2.0)
    def __str__(self):
        l = self.lenscale()
        (low,high), = self.param_bounds()
        if low is not None:
            low = self.lenscale(low)
        if high is not None:
            high = self.lenscale(high)
        return "SquaredExponentialKernel{l=%s,bounds=(%s,%s)}"%(
                    str(l),str(low), str(high))

    @classmethod
    def alloc(cls, l=1, l_min=1e-4, l_max=1000):
        log_l = numpy.log(2*(l**2))
        log_lenscale = theano.shared(log_l)
        if l_min is None:
            log_lenscale_min = None
        else:
            log_lenscale_min = numpy.log(2*(l_min**2))
        if l_max is None:
            log_lenscale_max = None
        else:
            log_lenscale_max = numpy.log(2*(l_max**2))
        return cls(log_lenscale=log_lenscale,
                log_lenscale_min=log_lenscale_min,
                log_lenscale_max=log_lenscale_max)

    def params(self):
        return [self.log_lenscale]
    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        ll2 = TT.exp(self.log_lenscale) #2l^2
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * TT.dot(x, y.T))
        K = TT.exp(-d/ll2)
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
    def __str__(self):
        l = numpy.exp(self.log_lenscale.value)
        return "ExponentialKernel{l=%s}"%str(l)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(l)
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def params(self):
        return [self.log_lenscale]
    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        l = TT.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * TT.dot(x, y.T))
        K = TT.exp(-TT.sqrt(d)/l)
        return K

class CategoryKernel(object):
    """
    K(x,y) is 1 if x==y else exp(-1/l)

    The idea is that it's like a SquaredExponentialKernel
    where every point is a distance of 1 from every other one, 
    except itself.

    Attributes:
        
        l - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.l.ndim!=0:
            raise TypeError('log_denom must be scalar', self.l)
    def lenscale(self, thing=None):
        if thing is None:
            thing = self.l
        return value(thing)
    def __str__(self):
        l = self.lenscale()
        (a,b), = self.param_bounds()
        return "CategoryKernel{l=%s,bounds=(%s,%s)}"%(
                str(l), str(a), str(b))

    @classmethod
    def alloc(cls, l=1.0, l_min=1e-5, l_max=100.):
        l = theano.shared(l)
        return cls(l=l,
                l_min=l_min,
                l_max=l_max,
                )

    def params(self):
        return [self.l]
    def param_bounds(self):
        return [(self.l_min, self.l_max)]

    def K(self, x, y):
        xx = x.reshape((x.shape[0],))
        yy = y.reshape((y.shape[0],))
        xx = xx.dimshuffle(0,'x') # drop cols because there should only be 1
        yy = yy.dimshuffle(0)     # drop cols because there should only be 1
        K = TT.exp(-TT.neq(xx,yy)/self.l)
        return K

class ConvexMixtureKernel(object):
    """

    Attributes:
    
        kernels -
        element_ranges - each kernel looks at these elements (default ALL)
        feature_names - 
        raw_coefs - 
        coefs - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __str__(self):
        coefs = self.coefs_f()
        ks = [str(k) for k in self.kernels]
        return 'ConvexMixtureKernel{%s}'%(','.join(['%s*%s'%(str(c),s) for c,s in zip(coefs, ks)]))
    def summary(self):
        import StringIO
        ss = StringIO.StringIO()
        coefs = self.coefs_f()
        print >> ss,  "ConvexMixtureKernel:"
        for c, k,fname in zip(coefs,self.kernels, self.feature_names):
            print >> ss,  "  %f * %s '%s'" %(c, str(k), fname)
        return ss.getvalue()
    @classmethod
    def alloc(cls, kernels, coefs=None, element_ranges=None, feature_names=None):
        if coefs is None:
            raw_coefs = theano.shared(numpy.zeros(len(kernels)))
            print "HAAACK"
            raw_coefs.get_value(borrow=True)[0] += 1 
        else:
            raise NotImplementedError()
        coefs=TT.nnet.softmax(raw_coefs.dimshuffle('x',0))[0]
        coefs_f = theano.function([], coefs)
        return cls(
                kernels=kernels,
                coefs=coefs,
                coefs_f = coefs_f, #DEBUG
                raw_coefs = raw_coefs,
                element_ranges=element_ranges,
                feature_names = feature_names,
                )

    def params(self):
        rval = [self.raw_coefs]
        for k in self.kernels:
            rval.extend(k.params())
        return rval
    def param_bounds(self):
        rval = [(self.raw_coefs_min, self.raw_coefs_max)]
        for k in self.kernels:
            rval.extend(k.param_bounds())
        return rval

    def K(self, x, y):
        # get the kernel matrix from each sub-kernel
        if self.element_ranges is None:
            Ks = [kernel.K(x,y) for kernel in  self.kernels]
        else:
            assert len(self.element_ranges) == len(self.kernels)
            Ks = [kernel.K(x[:,er[0]:er[1]],y[:,er[0]:er[1]])
                    for (kernel,er) in zip(self.kernels, self.element_ranges)]
        # stack them up
        Kstack = TT.stack(*Ks)
        # multiply by coefs
        # and sum down to one kernel
        K = TT.sum(self.coefs.dimshuffle(0,'x','x') * Kstack,
                axis=0)
        return K

class ProductKernel(object):
    """

    Attributes:
    
        kernels -
        element_ranges - each kernel looks at these elements (default ALL)
        feature_names - 
        raw_coefs - 
        coefs - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __str__(self):
        ks = [str(k) for k in self.kernels]
        return 'ProductKernel{%s}'%(','.join(['%s*%s'%(str(c),s) for c,s in zip(coefs, ks)]))
    def summary(self):
        import StringIO
        ss = StringIO.StringIO()
        print >> ss,  "ProductKernel:"
        for k,fname in zip(self.kernels, self.feature_names):
            print >> ss,  "  %s '%s'" %(str(k), fname)
        return ss.getvalue()
    @classmethod
    def alloc(cls, kernels, element_ranges=None, feature_names=None):
        return cls(
                kernels=kernels,
                element_ranges=element_ranges,
                feature_names = feature_names,
                )

    def params(self):
        rval = []
        for k in self.kernels:
            rval.extend(k.params())
        return rval
    def param_bounds(self):
        rval = []
        for k in self.kernels:
            rval.extend(k.param_bounds())
        return rval

    def K(self, x, y):
        # get the kernel matrix from each sub-kernel
        if self.element_ranges is None:
            Ks = [kernel.K(x,y) for kernel in  self.kernels]
        else:
            assert len(self.element_ranges) == len(self.kernels)
            Ks = [kernel.K(x[:,er[0]:er[1]],y[:,er[0]:er[1]])
                    for (kernel,er) in zip(self.kernels, self.element_ranges)]
        # stack them up
        Kstack = TT.stack(*Ks)
        # multiply by coefs
        # and sum down to one kernel
        K = TT.prod(Kstack, axis=0)
        return K

#
# To move into a NEW FILE pylearn/gp/regression.py
#
#TODO: make an Op-friendly version of the minimize_nll function
#      that does not update shared variables by side-effect
#TODO: Talk to Olivier about how to make this a learner

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
        rK = PSD_hint(rK)

        nll = ( 0.5 * dots(y, matrix_inverse(rK), y)
            + 0.5 * TT.log(det(rK)) 
            + n / 2.0 * TT.log(2*numpy.pi))
        return nll

    def s_mean(self, x):
        K = self.kernel.K(self.x, self.x)
        rK = K + self.var_y * TT.eye(self.y.shape[0])
        rK = PSD_hint(rK)
        alpha = TT.dot(matrix_inverse(rK), self.y)

        K_x = self.kernel.K(self.x, x)
        y_x = TT.dot(alpha, K_x)
        return y_x

    def s_variance(self, x):
        K = self.kernel.K(self.x, self.x)
        rK = K + self.var_y * TT.eye(self.y.shape[0])
        rK = PSD_hint(rK)
        L = cholesky(rK)
        K_x = self.kernel.K(self.x, x)
        v = solve(L, K_x)
        var_x = 1 - (v**2).sum(axis=0)
        return var_x

    def minimize_nll(self, maxiter=None):
        """
        Fit GPR kernel parameters by minimizing magininal nll.

        Returns: None

        Side effect: chooses optimal kernel parameters.
        """

        cost = self.s_nll() + 0.1 * sum([(p**2).sum() for p in self.kernel.params()])

        nll = theano.function([], cost)
        #theano.printing.debugprint(nll)
        dnll_dparams = theano.function([], TT.grad(cost, self.kernel.params()))
        params = self.kernel.params()
        param_bounds = self.kernel.param_bounds()
        lbounds = []
        ubounds = []
        for lb,ub in param_bounds:
            lbounds.extend(numpy.asarray(value(lb)).flatten())
            ubounds.extend(numpy.asarray(value(ub)).flatten())
        bounds = numpy.asarray([lbounds, ubounds]).T

        def get_pt():
            #TODO: handle non-scalar parameters...
            rval = []
            for p in params:
                v = p.get_value().flatten()
                rval.extend(v)
            return  numpy.asarray(rval)
        def set_pt(pt):
            i = 0
            for p in params:
                shape = p.get_value(borrow=True).shape
                size = numpy.prod(shape)
                p.set_value(pt[i:i+size].reshape(shape))
                i += size
            assert i == len(pt)
            print self.kernel.summary()

        def f(pt):
            #print 'f', pt
            set_pt(pt)
            return nll()
        def df(pt):
            #print 'df', pt
            set_pt(pt)
            dparams = dnll_dparams()
            rval = []
            for dp in dparams:
                rval.extend(dp.flatten())
            rval =  numpy.asarray(rval)
            #print numpy.sqrt((rval**2).sum())
            return rval
        start_pt = get_pt()
        # WEIRD: I was using fmin_ncg here
        #        until I used a low multiplier on the sum-squared-error regularizer
        #        on the 'cost' above, which threw ncg into an inf loop!?
        #best_pt = scipy.optimize.fmin_cg(f, start_pt, df, maxiter=maxiter, epsilon=.02)
        best_pt, best_value, best_d = scipy.optimize.fmin_l_bfgs_b(f, start_pt, df, maxfun=3*maxiter, 
                #bounds=[(-10, 10)]*len(start_pt))
                bounds=bounds)
        print 'best_value', best_value

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

