import theano
import theano.tensor as TT


import numpy

from theano_linalg import solve, cholesky, diag, matrix_inverse, det

def dots(*args):
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval

#TODO: set this based on variance in estimates
log_observation_noise=theano.shared(0.00)
observation_noise = TT.exp(log_observation_noise)

#TODO: guess this based on some fraction of the data variance or smth
log_lenscale=theano.shared(0.0)
lenscale = TT.exp(log_lenscale)

def rbf_kernel(x0, x1, l=1.0):
    d = ((x0**2).sum(axis=1).dimshuffle(0,'x')
            + (x1**2).sum(axis=1)
            - 2 * TT.dot(x0, x1.T))
    K = TT.exp(-d/l)
    return K

x_train = TT.matrix()
y_train = TT.vector()

x_test = TT.matrix()

K = (rbf_kernel(x_train, x_train, l=lenscale)
        + observation_noise * TT.eye(x_train.shape[0]))

L = cholesky(K)

alpha = solve(L.T, solve(L, y_train))

K_test = rbf_kernel(x_train, x_test, l=lenscale)
mean_test = TT.dot(alpha, K_test)

v_test = solve(L, K_test)
var_test = 1 - (v_test**2).sum(axis=0)

gpr_fn = theano.function(
        [x_train, y_train, x_test],
        [mean_test, var_test])

#Marginal likelihood
lik = ( -0.5 * dots(y_train, matrix_inverse(K), y_train)
    - 0.5 * TT.log(det(K)) 
    - x_train.shape[1] / 2.0 * TT.log(2*numpy.pi))
nll = -lik

gpr_nll = theano.function([x_train,y_train],
        nll)

dl, dobsnoise = TT.grad(nll, [log_lenscale, log_observation_noise])
gpr_fit = theano.function([x_train, y_train], nll,
        updates={
            log_lenscale: lenscale - 0.1 * dl,
            log_observation_noise: observation_noise - 0.0001 * dobsnoise,
            })
gpr_df = theano.function([x_train, y_train], [dl, dobsnoise])


N_train_pts = 800

rng = numpy.random.RandomState(2346)
x = (numpy.arange(N_train_pts) * 1.0).reshape(N_train_pts,1)
y = rng.randn(N_train_pts) + x[:,0]

y = y - y.mean() 

def f(pt):
    log_lenscale.value = pt[0]
    log_observation_noise.value = pt[1]
    print pt[0], pt[1]
    return gpr_fit(x,y)
def df(pt):
    log_lenscale.value = pt[0]
    log_observation_noise.value = pt[1]
    dl,dn = gpr_df(x,y)
    dn *= 0
    return numpy.asarray([dl,dn])

import scipy.optimize
l,o = scipy.optimize.fmin_ncg(f, [0,numpy.log(1.01)], df)
print 'Optimal vals', numpy.exp(l), numpy.exp(o)
log_lenscale.value = l
log_observation_noise.value = o

print 'likelihood', gpr_nll(x,y)
if 0:
    log_lenscale.value = 10.0
    for i in xrange(10):
        for i in xrange(1000):
            gpr_fit(x,y)
        print 'fitting', gpr_fit(x,y), lenscale.value, observation_noise.value


xstar = (numpy.arange(0,N_train_pts,0.1)).reshape(N_train_pts/0.1,1)
meanstar, Vfstar = gpr_fn(x,y,xstar)
stddev = numpy.sqrt(Vfstar)


import matplotlib.pyplot as plt
plt.scatter(x[:,0],y)
plt.plot(xstar[:,0],meanstar, c='g')
plt.fill_between(xstar[:,0], meanstar + stddev, meanstar - stddev, color='g', alpha=0.3)
plt.show()

