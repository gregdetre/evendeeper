from ipdb import set_trace as pause
from math import ceil, sqrt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import time

from base import Network, Minibatch, Patternset
from datasets import load_mnist
from utils.utils import imagesc, sigmoid, sumsq, vec_to_arr

# TODO
#
# commit
# rename _act to _prob
# check learning rate is right for numcases
# create Epoch, TrainEpoch, TestEpoch, ValidationEpochadd 
# momentum
# PCD
# validation crit
# remove window manager comments
# init vis bias with hinton practical tip
# compare parameters


class RbmNetwork(Network):
    def __init__(self, n_v, n_h, lrate, wcost, v_shape=None, plot=True):
        self.n_v, self.n_h = n_v, n_h
        self.lrate = lrate
        self.w = self.init_weights(n_v, n_h)
        self.a = np.zeros(shape=(n_v,)) # bias to visible
        self.b = np.zeros(shape=(n_h,)) # bias to hidden
        self.v_shape = v_shape or (1,n_v)
        self.wcost = wcost

        self.fignum_layers = 1
        self.fignum_weights = 2
        self.fignum_dweights = 3
        self.fignum_errors = 4
        self.fignum_biases = 5
        self.fignum_dbiases = 6
        if plot:
            plt.figure(figsize=(5,7), num=self.fignum_layers)
            plt.figure(figsize=(9,6), num=self.fignum_weights) # 6,4
            plt.figure(figsize=(9,6), num=self.fignum_dweights)
            plt.figure(figsize=(3,2), num=self.fignum_errors)
            plt.figure(figsize=(3,2), num=self.fignum_biases)
            plt.figure(figsize=(3,2), num=self.fignum_dbiases)
            
        self.pause = False

    def init_weights(self, n_v, n_h, scale=0.01):
        # return np.random.uniform(size=(n_v, n_h), high=scale)
        return np.random.normal(size=(n_v, n_h), loc=0, scale=scale)

    def act_fn(self, x): return sigmoid(x)

    def test_trial(self, v_plus):
        h_plus_inp, h_plus_act,  = self.propagate_fwd(v_plus)
        v_minus_inp, v_minus_act = self.propagate_back(h_plus_act)
        # ERROR = (NPATTERNS,)
        error = self.calculate_error(v_minus_act, v_plus)
        return error, v_minus_act

    def learn_trial(self, v_plus):
        d_w, d_a, d_b = self.update_weights(v_plus)
        self.w += d_w
        self.a += d_a
        self.b += d_b
        if self.pause: pause()
        return d_w, d_a, d_b

    def calculate_error(self, actual, desired):
        return sumsq(actual - desired)

    def gibbs_step(self, v_plus):
        h_plus_inp, h_plus_act = self.propagate_fwd(v_plus)
        h_plus_state = self.samplestates(h_plus_act)
        v_minus_inp, v_minus_act = self.propagate_back(h_plus_state)
        v_minus_state = self.samplestates(v_minus_act)
        # h_minus_inp, h_minus_act = self.propagate_fwd(v_minus_state)
        h_minus_inp, h_minus_act = self.propagate_fwd(v_minus_act)
        h_minus_state = self.samplestates(h_minus_act)
        return \
            h_plus_inp, h_plus_act, h_plus_state, \
            v_minus_inp, v_minus_act, v_minus_state, \
            h_minus_inp, h_minus_act, h_minus_state

    def samplestates(self, x): return x > np.random.uniform(size=x.shape)

    def update_weights(self, v_plus):
        n_in_minibatch = float(v_plus.shape[0])
        h_plus_inp, h_plus_act, h_plus_state, \
            v_minus_inp, v_minus_act, v_minus_state, \
            h_minus_inp, h_minus_act, h_minus_state = self.gibbs_step(v_plus)
        d_w = np.zeros(self.w.shape)
        d_a = np.mean(self.lrate * (v_plus-v_minus_act) - self.wcost * self.a,
                      axis=0)
        d_b = np.mean(self.lrate * (h_plus_state-h_minus_act) - self.wcost * self.b,
                      axis=0)
        diff_plus_minus = np.dot(v_plus.T, h_plus_act) - np.dot(v_minus_act.T, h_minus_act)
        d_w = self.lrate * (diff_plus_minus/n_in_minibatch - self.wcost*self.w)
        return d_w, d_a, d_b

    def plot_layers(self, v_plus, ttl=None):
        v_bias = net.a.reshape(self.v_shape)
        h_bias = vec_to_arr(net.b)
        h_plus_inp, h_plus_act, h_plus_state, \
            v_minus_inp, v_minus_act, v_minus_state, \
            h_minus_inp, h_minus_act, h_minus_state = self.gibbs_step(v_plus)
        lmin, lmax = None, None

        v_plus = v_plus.reshape(self.v_shape)
        h_plus_inp = vec_to_arr(h_plus_inp)*1.
        h_plus_act = vec_to_arr(h_plus_act)*1.
        h_plus_state = vec_to_arr(h_plus_state)*1.
        v_minus_inp = v_minus_inp.reshape(self.v_shape)
        v_minus_act = v_minus_act.reshape(self.v_shape)
        v_minus_state = v_minus_state.reshape(self.v_shape)
        h_minus_inp = vec_to_arr(h_minus_inp)*1.
        h_minus_act = vec_to_arr(h_minus_act)*1.
        h_minus_state = vec_to_arr(h_minus_state)*1.

        # fig = plt.figure(figsize=(6,9))
        # fig = plt.gcf()
        fig = plt.figure(self.fignum_layers)
        plt.clf()
        if ttl: fig.suptitle(ttl)
        gs = gridspec.GridSpec(16,2)
        # top left downwards
        ax = fig.add_subplot(gs[    0,0]); im = imagesc(h_plus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_state')
        ax = fig.add_subplot(gs[    1,0]); im = imagesc(h_plus_act, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_act')
        ax = fig.add_subplot(gs[    2,0]); im = imagesc(h_plus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_inp')
        ax = fig.add_subplot(gs[ 5: 8,0]); im = imagesc(v_plus, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_plus'); fig.colorbar(im) # , ticks=[lmin, lmax])
        # top right downwards
        ax = fig.add_subplot(gs[    0,1]); im = imagesc(h_minus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_state')
        ax = fig.add_subplot(gs[    1,1]); im = imagesc(h_minus_act, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_act')
        ax = fig.add_subplot(gs[    2,1]); im = imagesc(h_minus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_inp')
        ax = fig.add_subplot(gs[ 5: 8,1]); im = imagesc(v_minus_state*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_state'); fig.colorbar(im) # , ticks=[lmin, lmax])
        ax = fig.add_subplot(gs[ 9:12,1]); im = imagesc(v_minus_act*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_act'); fig.colorbar(im) # , ticks=[lmin, lmax])
        ax = fig.add_subplot(gs[13:16,1]); im = imagesc(v_minus_inp*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_inp'); fig.colorbar(im) # , ticks=[lmin, lmax])
        plt.draw()

    def plot_biases(self, v_bias, h_bias, fignum, ttl=None):
        vmin = None # min(map(min, [v_bias, h_bias]))
        vmax = None # max(map(max, [v_bias, h_bias]))
        v_bias = v_bias.reshape(self.v_shape)
        h_bias = vec_to_arr(h_bias)
        fig = plt.figure(fignum)
        plt.clf()
        if ttl: fig.suptitle(ttl + '. range=%.2f to %.2f' % (vmin or float('NaN'), vmax or float('NaN')))
        gs = gridspec.GridSpec(1,2)
        ax = fig.add_subplot(gs[0,0]); im = imagesc(v_bias.reshape(self.v_shape), dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v bias'); fig.colorbar(im)
        ax = fig.add_subplot(gs[0,1]); im = imagesc(h_bias, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h bias'); fig.colorbar(im)
        plt.draw()

    def plot_weights(self, w, fignum, ttl=None):
        vmin, vmax = min(w.ravel()), max(w.ravel())
        fig = plt.figure(fignum)
        plt.clf()
        if ttl: fig.suptitle(ttl + '. range=%.2f to %.2f' % (vmin, vmax))
        nsubplots = int(ceil(sqrt(self.n_h)))
        gs = gridspec.GridSpec(nsubplots, nsubplots)
        for hnum in range(self.n_h):
            x,y = divmod(hnum, nsubplots)
            ax = fig.add_subplot(gs[x,y])
            im = imagesc(w[:,hnum].reshape(self.v_shape), dest=ax, vmin=vmin, vmax=vmax)
            # ax.set_title('to H#%i' % hnum)
        fig.colorbar(im)
        plt.draw()
    
    def plot_errors(self, errors):
        plt.figure(self.fignum_errors)
        plt.clf()
        plt.plot(errors)
        plt.ylim(ymin=0, ymax=max(errors)*1.1)
        plt.draw()


def create_random_patternset(shape=(8,2), npatterns=5):
    iset = [np.random.rand(*shape) for _ in range(npatterns)]
    return Patternset(iset)

def create_mnist_patternset(npatterns=None):
    print 'Loading %s MNIST patterns' % (str(npatterns) if npatterns else 'all')
    mnist_ds = load_mnist(filen='../rbm/minst_train.csv', nrows=npatterns)
    assert mnist_ds.X.shape[0] == npatterns
    pset = Patternset(mnist_ds.X, shape=(28,28))
    print 'done'
    return pset


if __name__ == "__main__":
    np.random.seed()

    lrate = 0.005
    wcost = 0.0002
    nhidden = 400
    npatterns = 50000
    train_minibatch_errors = []
    n_train_epochs = 100000
    plot_every_n = 1000
    should_plot = lambda n: not n % plot_every_n # e.g. 0, 100, 200, 300, 400, ...
    # plot_every_logn = 10
    # should_plot = lambda n: log(n,plot_every_logn).is_integer() # e.g. 10th, 100th, 1000th, 10000th, ...
    n_in_minibatch = 20

    # pset = create_random_patternset(npatterns=npatterns)
    pset = create_mnist_patternset(npatterns=npatterns)

    net = RbmNetwork(np.prod(pset.shape), nhidden, lrate, wcost, v_shape=pset.shape)
    # pset = create_random_patternset()
    print net
    print pset

    for epochnum in range(n_train_epochs):
        minibatch = Minibatch(pset, n_in_minibatch)

        [d_w, d_a, d_b] = net.learn_trial(minibatch.patterns)
        errors, _ = net.test_trial(minibatch.patterns)
        assert errors.shape == (n_in_minibatch,)
        minibatch_error = np.mean(errors)
        train_minibatch_errors.append(minibatch_error)

        msg = 'At E#%i, error = %.2f' % (epochnum, minibatch_error)
        print msg
        
        if should_plot(epochnum):
            pattern0 = minibatch.patterns[0].reshape(1, net.n_v)
            net.plot_layers(pattern0, ttl=msg)
            # net.plot_weights(net.w, net.fignum_weights, 'Weights to hidden at E#%i' % epochnum)
            # net.plot_weights(d_w, net.fignum_dweights, 'D weights to hidden at E#%i' % epochnum)
            net.plot_errors(train_minibatch_errors)
            # net.plot_biases(net.a, net.b, net.fignum_biases, 'Biases at E#%i' % epochnum)
            # net.plot_biases(d_a, d_b, net.fignum_dbiases, 'D biases at E#%i' % epochnum)

    for patnum in range(npatterns):
        pattern = pset.get(patnum).reshape(1, net.n_v)
        error, _ = net.test_trial(pattern)
        print 'End of training (E#%i), error = %.2f' % (n_train_epochs, error)
        pset.imshow(v_minus)

    print '  '.join(['%.2f' % error for error in train_minibatch_errors])
    plt.figure()
    net.plot_errors(train_minibatch_errors)
    pause()

