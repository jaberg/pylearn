import numpy
import matplotlib.pyplot as plt
import kernels

N_train_pts = 8

rng = numpy.random.RandomState(2346)
x0 = (numpy.arange(N_train_pts) * 1.0).reshape(N_train_pts,1)
y0 = rng.randn(N_train_pts) +x0[:,0]
x1 = (numpy.arange(N_train_pts) * 1.0).reshape(N_train_pts,1)+0.5
y1 = rng.randn(N_train_pts) +x0[:,0]

x = numpy.vstack([x0,x1])
y = numpy.hstack([y0,y1])
y = y - y.mean() 

x_new = (numpy.arange(0,N_train_pts,0.1)).reshape(N_train_pts/0.1,1)

if 0:
    kernel = kernels.ConvexMixtureKernel.alloc([
        kernels.SquaredExponentialKernel.alloc(),
        kernels.ExponentialKernel.alloc()
            ])
elif 1:
    kernel = kernels.ConvexMixtureKernel.alloc(
            kernels = [
                kernels.SquaredExponentialKernel.alloc(),
                kernels.ExponentialKernel.alloc()
            ])
else:
    kernel = kernels.ExponentialKernel.alloc()
# fit the kernels
gpr = kernels.GPR_math(kernel, x, y, var_y=0.2)
gpr.minimize_nll(maxiter=50)
meanstar, Vfstar = gpr.mean_variance(x_new)
# get the predictions at new points
stddev = numpy.sqrt(Vfstar)

plt.scatter(x[:,0],y)
plt.plot(x_new[:,0],meanstar, c='g')
plt.fill_between(x_new[:,0], meanstar + stddev, meanstar - stddev, color='g', alpha=0.3)
plt.show()

