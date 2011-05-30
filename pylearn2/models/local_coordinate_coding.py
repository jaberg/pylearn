from theano import function, shared
from framework.optimization import linear_conj_grad_r as cg
from framework.optimization import featuresign as fs
import numpy as N
import theano.tensor as T

class LocalCoordinateCoding:
    def __init__(self, nvis, nhid, coeff):
        self.nvis = nvis
        self.nhid = nhid
        self.coeff = float(coeff)
        self.rng = N.random.RandomState([1,2,3])

        self.redo_everything()

    def get_output_channels(self):
        return self.nhid

    def redo_everything(self):
        self.W = shared(self.rng.randn(self.nhid,self.nvis), name='W')
        self.W.T.name = 'W.T'

    def weights_format(self):
        return ['h','v']

    def optimize_gamma(self, example):

        #variable names chosen to follow the arguments to l1ls_featuresign

        Y = N.zeros((self.nvis,1))
        Y[:,0] = example

        #print Y.shape

        c = 1e-10+N.square(self.W.get_value(borrow=True)-example).sum(axis=1)

        A = self.W.get_value(borrow=True).T/c

        x = fs.l1ls_featuresign(A,Y,self.coeff)[:,0]

        #print 'c.shape '+str(c.shape)
        #print 'x.shape '+str(x.shape)

        g = x/c

        #print 'g.shape '+str(g.shape)

        return g


    def learn(self, dataset, batch_size):
        #TODO-- this results in compilation happening every time learn is called
        #should cache the compilation results, including those inside cg


        X = dataset.get_design_matrix()
        m = X.shape[0]
        assert X.shape[1] == self.nvis

        gamma = N.zeros((batch_size,self.nhid))


        cur_gamma = T.vector(name='cur_gamma')
        cur_v = T.vector(name='cur_v')
        recons = T.dot(cur_gamma,self.W)
        recons.name = 'recons'

        recons_diffs = cur_v - recons
        recons_diffs.name = 'recons_diffs'

        recons_diff_sq = T.sqr(recons_diffs)
        recons_diff_sq.name = 'recons_diff'

        recons_error = T.sum( recons_diff_sq )
        recons_error.name = 'recons_error'

        dict_dists = T.sum(T.sqr(self.W - cur_v),axis=1)
        dict_dists.name = 'dict_dists'

        abs_gamma = abs(cur_gamma)
        abs_gamma.name = 'abs_gamma'

        weighted_dists = T.dot(abs_gamma,dict_dists)
        weighted_dists.name = 'weighted_dists'

        penalty = self.coeff * weighted_dists
        penalty.name = 'penalty'


        #prevent directions of absolute flatness in the hessian
        #W_sq = T.sqr(self.W)
        #W_sq.name = 'W_sq'
        #debug =  T.sum(W_sq)
        debug = 1e-10 * T.sum(dict_dists)
        debug.name = 'debug'

        #J = debug
        J = recons_error + penalty + debug
        J.name = 'J'

        Jf = function([cur_v, cur_gamma],J)

        start = self.rng.randint(m-batch_size+1)
        batch_X = X[start:start+batch_size,:]

        #TODO-- optimize gamma
        print 'optimizing gamma'
        for i in xrange(batch_size):
            #print str(i+1)+'/'+str(batch_size)
            gamma[i,:] = self.optimize_gamma(batch_X[i,:])

        print 'max min'
        print N.abs(gamma).min(axis=0).max()
        print 'min max'
        print N.abs(gamma).max(axis=0).max()

        #Optimize W
        print 'optimizing W'
        cg.linear_conj_grad_r_hack(J, [self.W], [cur_v, cur_gamma], [batch_X, gamma], max_iters = 3)

        err = 0.

        for i in xrange(batch_size):
            err += Jf(batch_X[i,:],gamma[i,:])

        assert not N.isnan(err)
        assert not N.isinf(err)

        print 'err: '+str(err)
    #
#
