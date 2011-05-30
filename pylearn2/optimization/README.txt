.. _v2planning_optimization:

Optimization API
================

Members: Bergstra, Lamblin, Delalleau, Glorot, Breuleux, Bordes
Leader: Bergstra


Description
-----------

This API is for iterative optimization algorithms, such as:

 - stochastic gradient descent (incl. momentum, annealing)
 - delta bar delta
 - conjugate methods
 - L-BFGS
 - "Hessian Free"
 - SGD-QN
 - TONGA

The API includes an iterative interface based on Theano, and a one-shot
interface similar to SciPy and MATLAB that is based on Python and Numpy, that
only uses Theano for the implementation.


Theano Interface
-----------------

The theano interface to optimization algorithms is to ask for a dictionary of
updates that can be used in theano.function.  Implementations of iterative
optimization algorithms should be global functions with a signature like
'iterative_optimizer'.

.. code-block:: python

    def iterative_optimizer(parameters, 
            cost=None,
            gradients=None,
            stop=None, 
            updates=None,
            **kwargs):
        """
        :param parameters: list or tuple of Theano variables 
            that we want to optimize iteratively.  If we're minimizing f(x), then
            together, these variables represent 'x'.  Typically these are shared
            variables and their values are the initial values for the minimization
            algorithm.

        :param cost: scalar-valued Theano variable that computes an exact or noisy estimate of
            cost  (what are the conditions on the noise?).  Some algorithms might
            need an exact cost, some algorithms might ignore the cost if the
            gradients are given.

        :param gradients: list or tuple of Theano variables representing the gradients on
            the corresponding parameters.  These default to tensor.grad(cost,
            parameters).

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


Numpy Interface
---------------

The numpy interface to optimization algorithms is supposed to mimick
scipy's.  Its arguments are numpy arrays, and functions that manipulate numpy
arrays.

.. code-block:: python

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


There is also a numpy-based wrapper to the iterative algorithms.
This can be more useful than minimize() because it doesn't hog program
control.  Technically minimize() is probably implemented using this
minimize_iterator interface.

.. code-block:: python

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
            """Initialize state (arguments as in minimize())
            """
        def __iter__(self):
            return self
        def next(self):
            """Take a step of minimization and return self raises StopIteration when
            the algorithm is finished with minimization

            """


JB replies: I don't think so, but for the few lines of code required, I think it would be nice to
provide an function that matches scipy's API.

OD: Looks like the reply above was pasted in the wrong place... where was it
supposed to go?

Examples
--------

Simple stochastic gradient descent could be called like this:

.. code-block:: python

    sgd([p], gradients=[g], step_size=.1)

and this would return

.. code-block:: python

    {p:p-.1*g}


Simple stochastic gradient descent with extra updates:

.. code-block:: python

    sgd([p], gradients=[g], updates={a:b}, step_size=.1)

will return

.. code-block:: python

    {a:b, p:p-.1*g}


If the parameters collide with keys in a given updates dictionary an exception
will be raised:

.. code-block:: python

    sgd([p], gradients=[g], updates={p:b}, step_size=.1)

will raise a ``KeyError``.

