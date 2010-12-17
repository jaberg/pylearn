
import numpy
import matplotlib.pyplot as plt
import scipy.linalg
try:
    solve_triangular = scipy.linalg.solve_triangular
except:
    solve_triangular = scipy.linalg.solve

observation_noise = 0.01
lenscale=0.1

rng = numpy.random.RandomState(234)

x = numpy.arange(10) * 1.0
y = rng.randn(10)

y_mean = y - y.mean()

def SC_K(l,r, scale=1.0, lenscale=1.0):
    l = l.reshape(len(l), 1)
    r = r.reshape(1, len(r))

    d = (l**2 + r**2
            - 2 * l * r)
    K = scale * numpy.exp(-d/lenscale)
    return K


K = SC_K(x, x, lenscale=lenscale) + observation_noise*numpy.identity(10)

L_factor = scipy.linalg.cho_factor(K)
alpha = scipy.linalg.cho_solve(L_factor, y)

L = scipy.linalg.cholesky(K, lower=True)
alpha2 = solve_triangular(L, y)
alpha2 = solve_triangular(L.T, alpha2)

xstar = numpy.arange(0,10,0.1)
Kstar = SC_K(x, xstar, lenscale=lenscale)
meanstar = numpy.dot(alpha, Kstar)
plt.plot(xstar,meanstar, c='r')

meanstar = numpy.dot(alpha2, Kstar)
plt.plot(xstar,meanstar, c='g')

#Variance:

v = solve_triangular(L,Kstar)
print v.shape
Vfstar = 1 - (v**2).sum(axis=0)

stddev = numpy.sqrt(Vfstar)

plt.fill_between(xstar, meanstar + stddev, meanstar - stddev, color='b', alpha=0.3)

plt.scatter(x,y_mean)

plt.show()

