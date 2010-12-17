import theano
import theano.tensor as TT

from theano_linalg import solve, cholesky

observation_noise=.01
lenscale=1.1

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

import numpy

rng = numpy.random.RandomState(234)
x = (numpy.arange(10) * 1.0).reshape(10,1)
y = rng.randn(10)
y_mean = y - y.mean()
xstar = (numpy.arange(0,10,0.1)).reshape(100,1)

meanstar, Vfstar = gpr_fn(x,y,xstar)
stddev = numpy.sqrt(Vfstar)

import matplotlib.pyplot as plt
plt.scatter(x[:,0],y_mean)
plt.plot(xstar[:,0],meanstar, c='g')
plt.fill_between(xstar[:,0], meanstar + stddev, meanstar - stddev, color='g', alpha=0.3)
plt.show()

