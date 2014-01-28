import numpy as np
from random import sample

from utils.utils import imagesc, isunique, vec_to_arr


class Network(object):
    def __repr__(self):
        return '%s (%ix%i)' % (self.__class__.__name__, self.n_v, self.n_h)

    def init_weights(self):
        self.w = np.random.uniform(size=(self.n_v, self.n_h))

    def act_fn(self, x): return x # linear activation function by default, i.e. no transformation

    def propagate_fwd(self, act1, w12, b2):
        # W = (N_LOWER x N_UPPER), ACT2 = (NPATTERNS x N_UPPER), B2 = (N_UPPER,)
        inp2 = np.dot(act1, w12) + b2
        act2 = self.act_fn(inp2)
        return inp2, act2

    def propagate_back(self, act2, w12, b1):
        # W = (N_LOWER x N_UPPER), ACT2 = (NPATTERNS x N_UPPER), B1 = (N_LOWER,)
        inp1 = np.dot(w12, act2.T) + b1
        # return INP1 as (NPATTERNS x N_LOWER)
        inp1 = inp1.T
        act1 = self.act_fn(inp1)
        return inp1, act1

    def calculate_error(self, target): raise NotImplementedError

    def update_weights(self, target): raise NotImplementedError

    def test_trial(self): raise NotImplementedError

    def learn_trial(self): raise NotImplementedError


class Patternset(object):
    def __init__(self, iset, shape=None):
        # ISET (inputs) = list of (X x Y) numpy arrays
        #
        # check they're all the same shape
        assert isunique([i.shape for i in iset])
        if shape is None:
            # if it's a (N,) vector, turn it into a (N,1) array
            if len(iset[0].shape) == 1: iset = [vec_to_arr(i) for i in iset]
        else:
            iset = [i.reshape(shape) for i in iset]
        self.shape = iset[0].shape
        self.iset = iset

    def __repr__(self):
        return '%s I(%ix%i)x%i' % (
            self.__class__.__name__,
            self.shape[0], self.shape[1], len(self.iset))

    def get(self, p): return self.iset[p].ravel()

    def getmulti(self, ps): return [self.iset[p].ravel() for p in ps]

    def imshow(self, x, dest=None): imagesc(x.reshape(self.shape), dest=dest)

    def __len__(self): return len(self.iset)


class Minibatch(object):
    def __init__(self, pset, n):
        self.pset = pset
        self.n = n
        self.patterns = np.array(self.pset.getmulti(sample(range(len(self.pset)), self.n)))

