import numpy as np

from utils import imagesc, isunique, vec_to_arr


class Network(object):
    def __repr__(self):
        return '%s (%ix%i)' % (self.__class__.__name__, self.n_v, self.n_h)

    def init_weights(self):
        self.w = np.random.uniform(size=(self.n_v, self.n_h))

    def act_fn(self, x): return x # linear activation function by default, i.e. no transformation

    def propagate_fwd(self, v):
        h_inp = np.dot(v, self.w) + self.b
        h_act = self.act_fn(h_inp)
        return h_inp, h_act

    def propagate_back(self, h):
        v_inp = np.dot(self.w, h) + self.a
        v_act = self.act_fn(v_inp)
        return v_inp, v_act

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

    def imshow(self, x, dest=None): imagesc(x.reshape(self.shape), dest=dest)
