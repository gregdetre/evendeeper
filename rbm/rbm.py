from copy import copy
from ipdb import set_trace as pause
from math import ceil, sqrt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import time

from base import create_mnist_patternset, Minibatch, Network, Patternset
from utils.utils import imagesc, sigmoid, sumsq, vec_to_arr

# TODO
#
# why isn't original update_weights working???
# parallelise parallel tempering
# is parallel tempering working???
# rename _act to _prob
# make sure lrate divides by minibatch consistently
# make sure we're using momentum
# make sure we're using weight cost
# create Epoch, TrainEpoch, TestEpoch, ValidationEpochadd 
# PCD
# validation crit
# init vis bias with hinton practical tip


class RbmNetwork(Network):
    def __init__(self, n_v, n_h, lrate, wcost, momentum, v_shape=None, plot=True):
        self.n_v, self.n_h = n_v, n_h
        self.lrate = lrate
        self.w = self.init_weights(n_v, n_h)
        self.a = np.zeros(shape=(n_v,)) # bias to visible
        self.b = np.zeros(shape=(n_h,)) # bias to hidden
        self.d_w = np.zeros(shape=self.w.shape)
        self.d_a = np.zeros(shape=self.a.shape)
        self.d_b = np.zeros(shape=self.b.shape)
        self.v_shape = v_shape or (1,n_v)
        self.wcost = wcost
        self.momentum = momentum

        self.plot = plot
        self.fignum_layers = 1
        self.fignum_weights = 2
        self.fignum_dweights = 3
        self.fignum_errors = 4
        self.fignum_biases = 5
        self.fignum_dbiases = 6
        if self.plot:
            plt.figure(figsize=(5,7), num=self.fignum_layers)
            plt.figure(figsize=(9,6), num=self.fignum_weights) # 6,4
            # plt.figure(figsize=(9,6), num=self.fignum_dweights)
            plt.figure(figsize=(3,2), num=self.fignum_errors)
            plt.figure(figsize=(3,2), num=self.fignum_biases)
            # plt.figure(figsize=(3,2), num=self.fignum_dbiases)

    def init_weights(self, n_v, n_h, scale=0.01):
        # return np.random.uniform(size=(n_v, n_h), high=scale)
        return np.random.normal(size=(n_v, n_h), loc=0, scale=scale)

    def propagate_fwd(self, v):
        return super(RbmNetwork, self).propagate_fwd(v, self.w, self.b)

    def propagate_back(self, h):
        return super(RbmNetwork, self).propagate_back(h, self.w, self.a)

    def act_fn(self, x): return sigmoid(x)

    def test_trial(self, v_plus):
        h_plus_inp, h_plus_prob,  = self.propagate_fwd(v_plus)
        v_minus_inp, v_minus_prob = self.propagate_back(h_plus_prob)
        # ERROR = (NPATTERNS,)
        error = self.calculate_error(v_minus_prob, v_plus)
        return error, v_minus_prob

    def learn_trial(self, v_plus):
        n_in_minibatch = float(v_plus.shape[0])

        d_w, d_a, d_b = self.update_weights(v_plus)

        # d_w, d_a, d_b = self.update_weights_pt(v_plus)

        d_w = (self.lrate/n_in_minibatch)*(d_w - self.wcost*self.w) + self.momentum*self.d_w
        d_a = (self.lrate/n_in_minibatch)*(d_a - self.wcost*self.a) + self.momentum*self.d_a
        d_b = (self.lrate/n_in_minibatch)*(d_b - self.wcost*self.b) + self.momentum*self.d_b

        self.w = self.w + d_w
        self.a = self.a + d_a
        self.b = self.b + d_b
        self.d_w, self.d_a, self.d_b = d_w, d_a, d_b

    def calculate_error(self, actual, desired):
        return sumsq(actual - desired)

    def gibbs_step(self, v_plus):
        h_plus_inp, h_plus_prob = self.propagate_fwd(v_plus)
        h_plus_state = self.samplestates(h_plus_prob)
        v_minus_inp, v_minus_prob = self.propagate_back(h_plus_state)
        v_minus_state = self.samplestates(v_minus_prob)
        # h_minus_inp, h_minus_prob = self.propagate_fwd(v_minus_state)
        h_minus_inp, h_minus_prob = self.propagate_fwd(v_minus_prob)
        h_minus_state = self.samplestates(h_minus_prob)
        return \
            h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_minus_state, \
            h_minus_inp, h_minus_prob, h_minus_state

    def samplestates(self, x): 
        return x > np.random.uniform(size=x.shape)

    def update_weights(self, v_plus):
        n_in_minibatch = float(v_plus.shape[0])
        h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_minus_state, \
            h_minus_inp, h_minus_prob, h_minus_state = self.gibbs_step(v_plus)
        diff_plus_minus = np.dot(v_plus.T, h_plus_prob) - np.dot(v_minus_prob.T, h_minus_prob)
        d_w = diff_plus_minus
        d_a = np.mean(v_plus-v_minus_prob, axis=0)
        d_b = np.mean(h_plus_state-h_minus_prob, axis=0)
        return d_w, d_a, d_b

    def update_weights_pt(self, v_plus):
        M = 10 # number of parallel chains

        n_mb = v_plus.shape[0]

        T = np.arange(1, M+1) # linear
        invT = 1.0/np.arange(1, M+1) # linear

        d_w = np.zeros_like(self.w)
        d_a = np.zeros(shape=self.a.shape)
        d_b = np.zeros(shape=self.b.shape)

        h_minus_acts = np.zeros((M, n_mb, self.n_h))
        h_plus_acts = np.zeros((M, n_mb, self.n_h))
        v_minus_acts = np.zeros((M, n_mb, self.n_v))
        v_minus_states = np.zeros((M, n_mb, self.n_v))
        h_plus_states = np.zeros((M, n_mb, self.n_h))
        
        w_orig = self.w.copy() # backup weights
        a_orig = self.a.copy() # backup visible bias
        b_orig = self.b.copy() # backup hidden bias

        for m in range(M):
           # smoothing
           self.w = w_orig * (1.0/T[m])
           self.a = a_orig * (1.0/T[m])
           self.b = b_orig * (1.0/T[m])
           # perform CD1
           _, h_plus_act, h_plus_state, \
               _, v_minus_act, v_minus_state, \
               _, h_minus_act, _ = self.gibbs_step(v_plus)
           h_plus_states[m] = h_plus_state
           h_plus_acts[m] = h_plus_act
           v_minus_acts[m] = v_minus_act
           v_minus_states[m] = v_minus_state
           h_minus_acts[m] = h_minus_act

        # swapping
        for m in range(1, M, 2):
            v = v_minus_acts
            h = h_minus_acts

            def dofor(ratio, rand, v_orig, h_orig):
                v = copy(v_orig)
                h = copy(h_orig)
                for n in range(ratio.shape[0]):
                    if ratio[n] > rand[n]:
                        v[m][n], v[m-1][n] = v[m-1][n], v[m][n]
                        h[m][n], h[m-1][n] = h[m-1][n], h[m][n]
                return v,h

            def domat(ratio, rand, v_orig, h_orig):
                v = copy(v_orig)
                h = copy(h_orig)
                # RGR = RATIO_GT_RAND = is RATIO >
                # RAND?. we're going to convert to ints
                # (i.e. 0 and 1), so that we can use it for indexing
                rgr = (ratio > rand).astype(int)
                rgr_ix = rgr.nonzero()
                v[m-rgr,rgr_ix,:], v[m-(1-rgr),rgr_ix,:] = v[m-(1-rgr),rgr_ix,:], v[m-rgr,rgr_ix,:]
                h[m-rgr,rgr_ix,:], h[m-(1-rgr),rgr_ix,:] = h[m-(1-rgr),rgr_ix,:], h[m-rgr,rgr_ix,:]
                return v,h

            ratio = self.metropolis_ratio(invT[m], invT[m-1], v[m], v[m-1], h[m], h[m-1])
            rand = np.random.uniform(size=ratio.shape)
            v1, h1 = dofor(ratio, rand, v, h)
            v2, h2 = domat(ratio, rand, v, h)
            assert np.array_equal(v1, v2)
            assert np.array_equal(h1, h2)
            v, h = v1, h1

        for m in range(2, M, 2):
            v = v_minus_acts
            h = h_minus_acts
            ratio = self.metropolis_ratio(invT[m], invT[m-1], v[m], v[m-1], h[m], h[m-1])
            for n in range(ratio.shape[0]):
                if ratio[n] > np.random.uniform():
                    v[m][n], v[m-1][n] = v[m-1][n], v[m][n]
                    h[m][n], h[m-1][n] = h[m-1][n], h[m][n]

        #d_w = d_w + self.lrate * (diff_plus_minus/n_in_minibatch - self.wcost*self.w)
        #d_a = d_a + self.lrate * (v_sample-v_minus_acts[0]) - \
        #        self.wcost * self.a
        #d_b = d_b + self.lrate * (h_plus_state[0]-h_minus_acts[0]) - \
        #        self.wcost * self.b
        diff_plus_minus = np.dot(v_plus.T, h_plus_acts[0]) - np.dot(v_minus_acts[0].T, h_minus_acts[0])
  
        d_w = d_w + diff_plus_minus
        d_a = d_a + np.mean(v_plus - v_minus_states[0], axis=0) # d_a is visible bias (b)
        d_b = d_b + np.mean(h_plus_acts[0] - h_minus_acts[0], axis=0) # d_b is hidden bias (c)

        self.w = w_orig # restore weights
        self.a = a_orig # restore visible bias
        self.b = b_orig # restore hidden bias

        return d_w, d_a, d_b

    def metropolis_ratio(self, invT_curr, invT_prev, v_curr, v_prev, h_curr, h_prev):
        ratio = ((invT_curr-invT_prev) *
                 (self.energy_fn(v_curr, h_curr) - self.energy_fn(v_prev, h_prev)))
        return np.minimum(np.ones_like(ratio), ratio)

    def energy_fn(self, v, h):
        def dofor():
            n_mb = v.shape[0] # number in minibatch
            assert n_mb == h.shape[0]
            energy = np.zeros((n_mb,))
            for mb in range(n_mb):
                E_w = np.dot(np.dot(v[mb], self.w), h[mb].T)
                E_vbias = np.vdot(v[mb], self.a)
                E_hbias = np.vdot(h[mb], self.b)
                energy[mb] = -E_w - E_vbias - E_hbias
            return energy

        def domat():
            E_w = np.dot(np.dot(v, self.w), h.T)
            E_vbias = np.dot(v, self.a)
            E_hbias = np.dot(h, self.b)
            energy = -E_w - E_vbias - E_hbias
            return energy.mean(axis=1)

        # do1 = dofor()
        do2 = domat()
        # assert np.allclose(do1, do2)
        return do2

    def plot_layers(self, v_plus, ttl=None):
        if not self.plot: return
        v_bias = self.a.reshape(self.v_shape)
        h_bias = vec_to_arr(self.b)
        h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_minus_state, \
            h_minus_inp, h_minus_prob, h_minus_state = self.gibbs_step(v_plus)
        lmin, lmax = None, None

        v_plus = v_plus.reshape(self.v_shape)
        h_plus_inp = vec_to_arr(h_plus_inp)*1.
        h_plus_prob = vec_to_arr(h_plus_prob)*1.
        h_plus_state = vec_to_arr(h_plus_state)*1.
        v_minus_inp = v_minus_inp.reshape(self.v_shape)
        v_minus_prob = v_minus_prob.reshape(self.v_shape)
        v_minus_state = v_minus_state.reshape(self.v_shape)
        h_minus_inp = vec_to_arr(h_minus_inp)*1.
        h_minus_prob = vec_to_arr(h_minus_prob)*1.
        h_minus_state = vec_to_arr(h_minus_state)*1.

        # fig = plt.figure(figsize=(6,9))
        # fig = plt.gcf()
        fig = plt.figure(self.fignum_layers)
        plt.clf()
        if ttl: fig.suptitle(ttl)
        gs = gridspec.GridSpec(16,2)
        # top left downwards
        ax = fig.add_subplot(gs[    0,0]); im = imagesc(h_plus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_state')
        ax = fig.add_subplot(gs[    1,0]); im = imagesc(h_plus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_prob')
        ax = fig.add_subplot(gs[    2,0]); im = imagesc(h_plus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_plus_inp')
        ax = fig.add_subplot(gs[ 5: 8,0]); im = imagesc(v_plus, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_plus'); fig.colorbar(im) # , ticks=[lmin, lmax])
        # top right downwards
        ax = fig.add_subplot(gs[    0,1]); im = imagesc(h_minus_state, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_state')
        ax = fig.add_subplot(gs[    1,1]); im = imagesc(h_minus_prob, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_prob')
        ax = fig.add_subplot(gs[    2,1]); im = imagesc(h_minus_inp, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('h_minus_inp')
        ax = fig.add_subplot(gs[ 5: 8,1]); im = imagesc(v_minus_state*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_state'); fig.colorbar(im) # , ticks=[lmin, lmax])
        ax = fig.add_subplot(gs[ 9:12,1]); im = imagesc(v_minus_prob*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_prob'); fig.colorbar(im) # , ticks=[lmin, lmax])
        ax = fig.add_subplot(gs[13:16,1]); im = imagesc(v_minus_inp*1, dest=ax, vmin=lmin, vmax=lmax); ax.set_title('v_minus_inp'); fig.colorbar(im) # , ticks=[lmin, lmax])
        plt.draw()

    def plot_biases(self, v_bias, h_bias, fignum, ttl=None):
        if not self.plot: return
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
        if not self.plot: return
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
        if not self.plot: return
        plt.figure(self.fignum_errors)
        plt.clf()
        plt.plot(errors)
        plt.ylim(ymin=0, ymax=max(errors)*1.1)
        plt.draw()


def create_random_patternset(shape=(8,2), npatterns=5):
    patterns = [np.random.rand(*shape) for _ in range(npatterns)]
    return Patternset(patterns)


if __name__ == "__main__":
    np.random.seed()

    lrate = 0.01
    wcost = 0.0002
    nhidden = 100
    npatterns = 1000
    train_minibatch_errors = []
    n_train_epochs = 10000
    n_in_minibatch = 10
    momentum = 0.9
    plot = False
    plot_every_n = 1000
    should_plot = lambda n: not n % plot_every_n # e.g. 0, 100, 200, 300, 400, ...
    # plot_every_logn = 10
    # should_plot = lambda n: log(n,plot_every_logn).is_integer() # e.g. 10th, 100th, 1000th, 10000th, ...

    # pset = create_random_patternset(npatterns=npatterns)
    pset = create_mnist_patternset(npatterns=npatterns)

    net = RbmNetwork(np.prod(pset.shape), nhidden, lrate, wcost, momentum, v_shape=pset.shape, plot=plot)
    # pset = create_random_patternset()
    print net
    print pset

    for epochnum in range(n_train_epochs):
        minibatch = Minibatch(pset, n_in_minibatch)

        net.learn_trial(minibatch.patterns)
        errors, _ = net.test_trial(minibatch.patterns)
        assert errors.shape == (n_in_minibatch,)
        minibatch_error = np.mean(errors)
        train_minibatch_errors.append(minibatch_error)

        msg = 'At E#%i, error = %.2f' % (epochnum, minibatch_error)
        print msg
        
        if should_plot(epochnum):
            pattern0 = minibatch.patterns[0].reshape(1, net.n_v)
            net.plot_layers(pattern0, ttl=msg)
            net.plot_weights(net.w, net.fignum_weights, 'Weights to hidden at E#%i' % epochnum)
            # net.plot_weights(net.d_w, net.fignum_dweights, 'D weights to hidden at E#%i' % epochnum)
            net.plot_errors(train_minibatch_errors)
            net.plot_biases(net.a, net.b, net.fignum_biases, 'Biases at E#%i' % epochnum)
            # net.plot_biases(net.d_a, net.d_b, net.fignum_dbiases, 'D biases at E#%i' % epochnum)

    for patnum in range(npatterns):
        pattern = pset.get(patnum).reshape(1, net.n_v)
        error, v_minus = net.test_trial(pattern)
        print 'End of training (E#%i), error = %.2f' % (n_train_epochs, error)
        pset.imshow(v_minus)

    print '  '.join(['%.2f' % error for error in train_minibatch_errors])
    plt.figure()
    net.plot_errors(train_minibatch_errors)
    pause()

