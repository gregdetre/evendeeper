import numpy as np


class Network(object):
    def __init__(self, v_shape, h_shape, lrate):
        def to_shape(sh):
            # makes sure that you get a tuple back, for use as a shape in initializing arrays
            if isinstance(sh, int): return (sh,)
            elif all(isinstance(x, int) for x in sh): return sh
            else: raise Exception('Unknown shape type %s' % sh)

        self.v_shape, self.h_shape = to_shape(v_shape), to_shape(h_shape)
        self.n_v, self.n_h = np.prod(self.v_shape), np.prod(self.h_shape)
        self.v, self.h = np.empty(self.n_v), np.empty(self.n_h)
        self.lrate = lrate
        self.init_weights()

    def __repr__(self):
        return '%s (%ix%i)' % (self.__class__.__name__, self.n_v, self.n_h)

    def init_weights(self):
        self.w = np.random.rand(len(self.v), len(self.h))

    def act_fn(self, x): return x # linear activation function by default, i.e. no transformation

    def propagate_fwd(self, v):
        self.v = v.reshape(self.n_v,)
        self.h = self.act_fn(np.dot(self.v, self.w))

    def propagate_back(self, h):
        self.h = h.reshape(self.n_h)
        self.v = self.act_fn(np.dot(self.w, self.h))

    def calculate_error(self, target):
        target = target.reshape(self.n_h,)
        return sumsq(target - self.h)

    def update_weights(self, target):
        dw = np.empty(self.w.shape)
        for i in range(len(self.v)):
            for j in range(len(self.h)):
                dw[i,j] = self.lrate * self.v[i] * (target[j] - self.h[j])
        self.w += dw
        return dw

    def test_trial(self, input, target):
        input = input.reshape(self.n_v)
        target = target.reshape(self.n_h)
        self.propagate_fwd(input)
        error = self.calculate_error(target)
        return error

    def learn_trial(self, input, target):
        input = input.reshape(self.n_v)
        target = target.reshape(self.n_h)
        error = self.test_trial(input, target)
        dw = self.update_weights(target)
        return error, dw

    def show_v(self): return self.v.reshape(self.v_shape)
    def show_h(self): return self.h.reshape(self.h_shape)


class Patternset(object):
    def __init__(self, iset):
        # ISET (inputs) = list of (X x Y) numpy arrays
        #
        # check they're all the same shape
        assert isunique([i.shape for i in iset])
        self.shape = iset[0].shape
        assert len(self.shape) == 2 # each pattern 2-dimensional
        self.iset = iset

    def __repr__(self):
        return '%s I(%ix%i)x%i' % (
            self.__class__.__name__,
            self.iset.shape[0], self.iset.shape[1], len(self.iset))

    def get(self, p): return self.iset[p]

    def imshow(self, p): imagesc(self.get(p))
