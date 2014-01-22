from ipdb import set_trace as pause
import numpy as np
import random

from utils import imagesc, sumsq


"""
ipython nn.py --pylab
"""


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

    def propagate_fwd(self, v):
        self.v = v.reshape(self.n_v,)
        self.h = np.dot(self.v, self.w)

    def propagate_back(self, h):
        self.h = h.reshape(self.n_h)
        self.v = np.dot(self.w, self.h)

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
    def __init__(self, iset, tset):
        # ISET (inputs) = X x Y x NPATTERNS
        # TSET (targets) = P x Q x NPATTERNS
        assert len(iset.shape) == 3 # 3-dimensional
        assert len(tset.shape) == 3 # 3-dimensional
        assert iset.shape[2] == tset.shape[2] # NPATTERNS should be the same
        self.iset = iset
        self.tset = tset

    def __repr__(self):
        return '%s I(%ix%ix%i), T(%ix%ix%i)' % (
            self.__class__.__name__,
            self.iset.shape[0], self.iset.shape[1], self.iset.shape[2],
            self.tset.shape[0], self.tset.shape[1], self.tset.shape[2])

    def get(self, p):
        return self.iset[:,:,p], self.tset[:,:,p]

    def imshow(self, p):
        # based on my deeplearning/utils.py
        imagesc(self.iset[:,:,p])


def create_patternset():
    input0 = np.array([[.5,1,.2,0,0,0],
                       [.5,1,.2,0,0,0]])
    target0 = np.array([input0[0,:]])
    input1 = np.array([[0,.5,1,.2,0,0],
                       [0,.5,1,.2,0,0]])
    target1 = np.array([input1[0,:]])
    input2 = np.array([[0,0,.5,1,.2,0],
                       [0,0,.5,1,.2,0]])
    target2 = np.array([input2[0,:]])
    input3 = np.array([[0,0,0,.5,1,.2],
                       [0,0,0,.5,1,.2]])
    target3 = np.array([input3[0,:]])
    inputs = np.empty((2,6,4))
    inputs[:,:,0] = input0
    inputs[:,:,1] = input1
    inputs[:,:,2] = input2
    inputs[:,:,3] = input3
    targets = np.empty((1,6,4))
    targets[:,:,0] = target0
    targets[:,:,1] = target1
    targets[:,:,2] = target2
    targets[:,:,3] = target3
    pset = Patternset(inputs, targets)
#     pset.imshow(0)
#     pause()
#     pset.imshow(1)
#     pause()
#     pset.imshow(2)
#     pause()
#     pset.imshow(3)
#     pause()
    return pset


if __name__ == "__main__":
    net = Network((2,6), (1,6), 0.1)
    pset = create_patternset()
    print net
    print pset

    for p in range(4):
        print 'Before training, error on P%i = %.2f' % (p, net.test_trial(*pset.get(p)))

    npatterns = 4

    for e in range(100):
        p = random.randint(0,npatterns-1)
        input, target = pset.get(p)
        error, dw = net.learn_trial(input, target)
        print 'After %i epochs, error on P%i = %.2f' % (e, p, error)
        # imagesc(net.show_h())

    for p in range(4):
        print 'End of training, error on P%i = %.2f' % (p, net.test_trial(*pset.get(p)))
        imagesc(net.show_h())
        pause()

