
import numpy

import theano.tensor

from pylearn.opt.sgd import basic

def test_simple():
    # Test that simple use of this minimization algorithm is correct
    #
    rng = numpy.random.RandomState(44)
    s_m = theano.tensor.shared(rng.randn(5,4))
    cost = ((theano.tensor.dot(s_m, s_m.T) - theano.tensor.eye(5))**2).sum()
    learn_updates = basic([s_m], cost, step_size=0.01)
    learn_fn = theano.function([], cost, updates=learn_updates)

    errs = [learn_fn() for i in xrange(10)]
    assert errs[0] > 139
    assert errs[-1] < 2.5
    
#TODO: write the following tests in a way that they can be run *FOR ANY ALGO*
#      do not hard-code them for sgd.basic
#TODO:  - test that step_sizes are used correctly when given as list
#TODO:  - test that each exception is raised when appropriate
#TODO:  - test that gradients can be given explicitly

