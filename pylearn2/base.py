"""Base class for the components in other modules."""
# Standard library imports
import inspect

# Standard library imports
import cPickle
import os.path

# Third-party imports
import theano
from theano import tensor
from theano.sparse import SparseType

# Local imports
from .utils import subdict

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class Block(object):
    """
    Basic building block for deep architectures.
    """
    def params(self):
        """
        Get the list of learnable parameters in a Block.

        Returns
        -------
        param_list : list of shared variables
            A list of *shared* learnable parameters that are, in the
            implementor's judgment, typically learned in this model.
        """
        # NOTE: We return list(self._params) rather than self._params
        # in order to explicitly make a copy, so that the list isn't
        # absentmindedly modified. If a user really knows what they're
        # doing they can modify self._params.
        if None in self._params:
            raise ValueError('some parameters of %s not initialized' %
                             str(self))
        return list(self._params)

    def __call__(self, inputs):
        raise NotImplementedError('__call__')

    def save(self, save_file):
        """
        Dumps the entire object to a pickle file.
        Individual classes should override __getstate__ and __setstate__
        to deal with object versioning in the case of API changes.
        """
        save_dir = os.path.dirname(save_file)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fhandle = open(save_file, 'w')
        cPickle.dump(self, fhandle, -1)
        fhandle.close()

    @classmethod
    def fromdict(cls, conf, **kwargs):
        """ Alternative way to build a block, by using a dictionary """
        arglist = []
        kwargs.update(conf)
        # Loop over all superclasses of cls
        # NB : Supposes that "cls" is the first element returned by "getmro()"
        for elem in inspect.getmro(cls):
            # Extend arglist with arguments of elem.__init__
            argspec = inspect.getargspec(elem.__init__)
            arglist.extend(argspec[0])
            # If a keyworkds argument is not expected, then break the loop
            if argspec[2] is None:
                break
        # Build the class with appropriated arguments
        return cls(**subdict(kwargs, arglist))

    @classmethod
    def load(cls, load_file):
        """Load a serialized block."""
        if not os.path.isfile(load_file):
            raise IOError('File %s does not exist' % load_file)
        obj = cPickle.load(open(load_file))
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError('unpickled object was of wrong class: %s' %
                            obj.__class__)

    def function(self, name=None):
        """ Returns a compiled theano function to compute a representation """
        inputs = tensor.matrix()
        return theano.function([inputs], self(inputs), name=name)

    def invalid(self):
        return None in self._params


class StackedBlocks(Block):
    """
    A stack of Blocks, where the output of a block is the input of the next.
    """
    def __init__(self, layers):
        """
        Build a stack of layers.

        Parameters
        ----------
        layers: list of Blocks
            The layers to be stacked, ordered
            from bottom (input) to top (output)
        """
        self._layers = layers
        # Do not duplicate the parameters if some are shared between layers
        self._params = set([p for l in self._layers for p in l.params()])

    def layers(self):
        return list(self._layers)

    def __len__(self):
        return len(self._layers)

    def __call__(self, inputs):
        """
        Return the output representation of all layers, including the inputs.

        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be encoded. Assumed to be 2-tensors, with the
            first dimension indexing training examples and the second indexing
            data dimensions.

        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            A list of theano symbolic (or list thereof), each containing
            the representation at one level. The first element is the input.
        """
        # Build the hidden representation at each layer
        repr = [inputs]

        for layer in self._layers:
            outputs = layer(repr[-1])
            repr.append(outputs)

        return repr

    def function(self, name=None, repr_index=-1, sparse_input=False):
        """
        Compile a function computing representations on given layers.

        Parameters
        ----------
        name: string
            name of the function
        repr_index: int
            Index of the hidden representation to return.
            0 means the input, -1 the last output.
        """

        if sparse_input:
            inputs = SparseType('csr', dtype=floatX)()
        else:
            inputs = tensor.matrix()

        return theano.function(
                [inputs],
                outputs=self(inputs)[repr_index],
                name=name)

    def concat(self, name=None, start_index=-1, end_index=None):
        """
        Compile a function concatenating representations on given layers.

        Parameters
        ----------
        name: string
            name of the function
        start_index: int
            Index of the hidden representation to start the concatenation.
            0 means the input, -1 the last output.
        end_index: int
            Index of the hidden representation from which to stop
            the concatenation. We must have start_index < end_index.
        """
        inputs = tensor.matrix()
        return theano.function([inputs],
            outputs=tensor.concatenate(self(inputs)[start_index:end_index]),
            name=name)

    def append(self, layer):
        """
        Add a new layer on top of the last one
        """
        self.layers.append(layer)
        self._params.update(layer.params())


class Optimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """
    def updates(self):
        """Return symbolic updates to apply."""
        raise NotImplementedError()
