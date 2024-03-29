from copy import copy
from ipdb import set_trace as pause
from math import ceil, sqrt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import time

from base import create_mnist_patternsets, Minibatch, Network, Patternset
from utils.utils import imagesc, isfunction, sigmoid, sumsq, vec_to_arr
from utils.stopwatch import Stopwatch

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
    def __init__(self, n_v, n_h, lrate, wcost, momentum, n_temperatures=1, n_sampling_steps=1, v_shape=None, plot=True):
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

        # number of parallel tempering chains
        self.n_temperatures = n_temperatures if isfunction(n_temperatures) else lambda net: n_temperatures
        if self.n_temperatures(self) == 1: print 'Using single tempering'
        else: print 'Using parallel tempering with %d chains' % self.n_temperatures(self)

        # number of sampling steps in CD-k
        self.n_sampling_steps = n_sampling_steps \
                           if isfunction(n_sampling_steps) \
                           else lambda net: n_sampling_steps

        self.trial_num = 0
        self.plot = plot
        self.fignum_layers = 1
        self.fignum_weights = None # 2
        self.fignum_dweights = None # 3
        self.fignum_errors = 4
        self.fignum_biases = None # 5
        self.fignum_dbiases = None # 6
        if self.plot:
            plt.figure(figsize=(5,7), num=self.fignum_layers)
            if self.fignum_dweights: plt.figure(figsize=(9,6), num=self.fignum_weights) # 6,4
            if self.fignum_dweights: plt.figure(figsize=(9,6), num=self.fignum_dweights)
            plt.figure(figsize=(4,3), num=self.fignum_errors)
            if self.fignum_biases: plt.figure(figsize=(3,2), num=self.fignum_biases)
            if self.fignum_dbiases: plt.figure(figsize=(3,2), num=self.fignum_dbiases)

    def init_weights(self, n_v, n_h, scale=0.01):
        # return np.random.uniform(size=(n_v, n_h), high=scale)
        return np.random.normal(size=(n_v, n_h), loc=0, scale=scale)

    def propagate_fwd(self, v, w=None, a=None, b=None):
        if w is None: w = self.w
        if a is None: a = self.a
        if b is None: b = self.b
        return super(RbmNetwork, self).propagate_fwd(v, w, b)

    def propagate_back(self, h, w=None, a=None, b=None):
        if w is None: w = self.w
        if a is None: a = self.a
        if b is None: b = self.b
        return super(RbmNetwork, self).propagate_back(h, w, a)

    def act_fn(self, x): return sigmoid(x)

    def test_trial(self, v_plus):
        h_plus_inp, h_plus_prob,  = self.propagate_fwd(v_plus)
        v_minus_inp, v_minus_prob = self.propagate_back(h_plus_prob)
        # ERROR = (NPATTERNS,)
        error = self.calculate_error(v_minus_prob, v_plus)
        return error, v_minus_prob

    def learn_trial(self, v_plus):
        n_mb = v_plus.shape[0]
        self.trial_num += n_mb
        lrate_over_n_mb = self.lrate/float(n_mb)

        M = self.n_temperatures(self)
        assert(M>0)
        if M == 1:
            d_w, d_a, d_b = self.update_weights(v_plus) # use single tempering
        else:
            d_w, d_a, d_b = self.update_weights_pt(v_plus, M) # use parallel tempering

        d_w = lrate_over_n_mb*(d_w - self.wcost*self.w) + self.momentum*self.d_w
        d_a = lrate_over_n_mb*(d_a - self.wcost*self.a) + self.momentum*self.d_a
        d_b = lrate_over_n_mb*(d_b - self.wcost*self.b) + self.momentum*self.d_b

        self.w = self.w + d_w
        self.a = self.a + d_a
        self.b = self.b + d_b
        self.d_w, self.d_a, self.d_b = d_w, d_a, d_b

    def calculate_error(self, actual, desired):
        return sumsq(actual - desired)

    def k_gibbs_steps(self, v_plus, w=None, a=None, b=None):        
        k = self.n_sampling_steps(self)
        assert(k>0) # do at least one step

        for t in range(k):
            # forward propagation
            h_plus_inp, h_plus_prob = self.propagate_fwd(v_plus, w, a, b) 
            h_plus_state = self.samplestates(h_plus_prob)
            # backward propagation
            v_minus_inp, v_minus_prob = self.propagate_back(h_plus_state, w, a, b)
            v_plus = self.samplestates(v_minus_prob) # = v_minus_state
        h_minus_inp, h_minus_prob = self.propagate_fwd(v_minus_prob, w, a, b)
        h_minus_state = self.samplestates(h_minus_prob)
        return \
            h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_plus, \
            h_minus_inp, h_minus_prob, h_minus_state

    def samplestates(self, x): 
        return x > np.random.uniform(size=x.shape)

    def update_weights(self, v_plus):
        n_in_minibatch = float(v_plus.shape[0])
        h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_minus_state, \
            h_minus_inp, h_minus_prob, h_minus_state = self.k_gibbs_steps(v_plus)
        diff_plus_minus = np.dot(v_plus.T, h_plus_prob) - np.dot(v_minus_prob.T, h_minus_prob)
        d_w = diff_plus_minus
        d_a = np.mean(v_plus-v_minus_prob, axis=0)
        d_b = np.mean(h_plus_state-h_minus_prob, axis=0)
        return d_w, d_a, d_b

    def update_weights_pt(self, v_plus, M):
        n_mb = v_plus.shape[0]

        T = np.arange(1, M+1)
        invT = 1.0/T

        h_minus_acts = np.zeros((M, n_mb, self.n_h))
        h_plus_acts = np.zeros((M, n_mb, self.n_h))
        v_minus_acts = np.zeros((M, n_mb, self.n_v))
        v_minus_states = np.zeros((M, n_mb, self.n_v))      

        for m in range(M):
           # perform CDk
           _, h_plus_act, _, \
               _, v_minus_act, v_minus_state, \
               _, h_minus_act, _ = self.k_gibbs_steps(v_plus, self.w*invT[m], self.a*invT[m], self.b*invT[m])
           h_plus_acts[m] = h_plus_act
           v_minus_acts[m] = v_minus_act
           v_minus_states[m] = v_minus_state
           h_minus_acts[m] = h_minus_act

        v = v_minus_acts
        h = h_minus_acts
        # swapping, in 2 stages
        for m in range(1, M, 2):
            ratio = self.metropolis_ratio(invT[m], invT[m-1], v[m], v[m-1], h[m], h[m-1])
            rand = np.random.uniform(size=ratio.shape)
            if random.choice(ratio > rand):
                v[m], v[m-1] = v[m-1], v[m]
                h[m], h[m-1] = h[m-1], h[m]
        for m in range(2, M, 2):
            ratio = self.metropolis_ratio(invT[m], invT[m-1], v[m], v[m-1], h[m], h[m-1])
            rand = np.random.uniform(size=ratio.shape)
            if random.choice(ratio > rand):
                v[m], v[m-1] = v[m-1], v[m]
                h[m], h[m-1] = h[m-1], h[m]

        diff_plus_minus = np.dot(v_plus.T, h_plus_acts[0]) - np.dot(v_minus_acts[0].T, h_minus_acts[0])
        d_w = diff_plus_minus
        d_a = np.mean(v_plus - v_minus_states[0], axis=0) # d_a is visible bias (b)
        d_b = np.mean(h_plus_acts[0] - h_minus_acts[0], axis=0) # d_b is hidden bias (c)
        return d_w, d_a, d_b

    def metropolis_ratio(self, invT_curr, invT_prev, v_curr, v_prev, h_curr, h_prev):
        ratio = ((invT_curr-invT_prev) *
                 (self.energy_fn(v_curr, h_curr) - self.energy_fn(v_prev, h_prev)))
        return np.minimum(np.ones_like(ratio), ratio)

    def energy_fn(self, v, h):
        E_w = np.dot(np.dot(v, self.w), h.T)
        E_vbias = np.dot(v, self.a)
        E_hbias = np.dot(h, self.b)
        energy = -E_w - E_vbias - E_hbias
        return energy.mean(axis=1)


    def plot_layers(self, v_plus, ttl=None):
        if not self.plot: return
        v_bias = self.a.reshape(self.v_shape)
        h_bias = vec_to_arr(self.b)
        h_plus_inp, h_plus_prob, h_plus_state, \
            v_minus_inp, v_minus_prob, v_minus_state, \
            h_minus_inp, h_minus_prob, h_minus_state = self.k_gibbs_steps(v_plus)
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
    
    def plot_errors(self, train_errors, valid_errors, test_errors):
        if not self.plot: return
        plt.figure(figsize=(9,6), num=self.fignum_errors)
        plt.clf()
        epochrange = range(len(train_errors))
        assert len(train_errors) == len(valid_errors)
        assert len(train_errors) == len(test_errors)
        train_line = plt.plot(train_errors, label='Train')
        valid_line = plt.plot(valid_errors, label='Valid')
        test_line = plt.plot(test_errors, label='Test')
        plt.legend()
        max_error = max(max(train_errors), max(test_errors), max(valid_errors))
        plt.ylim(ymin=0, ymax=max_error*1.1)
        plt.draw()

    def save_error_plots(self, train_errors, valid_errors, test_errors, filename='errors.png'):
        self.plot_errors(train_errors, valid_errors, test_errors)
        plt.figure(self.fignum_errors).savefig(filename)


def create_random_patternset(shape=(8,2), npatterns=5):
    patterns = [np.random.rand(*shape) for _ in range(npatterns)]
    return Patternset(patterns)


if __name__ == "__main__":
    np.random.seed()

    lrate = 0.01
    wcost = 0.0002
    nhidden = 500
    n_trainpatterns = 5000
    n_validpatterns = 1000
    n_train_epochs = 100000
    n_in_minibatch = 5
    momentum = 0.9
    #n_temperatures = 1
    def n_temperatures(net):
        """
        Returns integer; 1 for single tempering
        """
        #M = net.trial_num / 500
        return 1# None #M if M > 1 else None

    #n_sampling_steps = 1
    def n_sampling_steps(net):
        """
        Returns integer; number of sampling steps used in CD-k
        """
        return 1

    plot = True
    plot_every_n = 100
    should_plot = lambda n: not n % plot_every_n # e.g. 0, 100, 200, 300, 400, ...
    print_every_n = 100
    should_print = lambda n: not n % print_every_n # e.g. 0, 100, 200, 300, 400, ...

    train_pset, valid_pset, test_pset = create_mnist_patternsets(n_trainpatterns=n_trainpatterns, n_validpatterns=n_validpatterns)
    n_trainpatterns, n_validpatterns, n_testpatterns = train_pset.n, valid_pset.n, test_pset.n

    net_t = Stopwatch()
    net = RbmNetwork(np.prod(train_pset.shape), nhidden, lrate, wcost, momentum, n_temperatures, n_sampling_steps, v_shape=train_pset.shape, plot=plot)
    print 'Init net in %.1fs' % net_t.finish(milli=False), net
    print 'train:', train_pset, '\tvalid:', valid_pset, '\ttest:', test_pset

    train_errors = []
    valid_errors = []
    test_errors = []

    valid_patterns = np.array(valid_pset.patterns).reshape((n_validpatterns,-1))
    test_patterns = np.array(test_pset.patterns).reshape((n_testpatterns, -1))

    for epochnum in range(n_train_epochs):
        mb_t = Stopwatch()
        minibatch_pset = Minibatch(train_pset, n_in_minibatch)
        # print '\tsetup minibatch in %.1f' % mb_t.finish(milli=False)

        learn_trial_t = Stopwatch()
        net.learn_trial(minibatch_pset.patterns)
        # print '\tlearn_trial in %.1f' % learn_trial_t.finish(milli=False)
        
        printme = should_print(epochnum)
        plotme = should_plot(epochnum)
        if printme or plotme:
            train_error = np.mean(net.test_trial(minibatch_pset.patterns)[0])
            train_errors.append(train_error)
            valid_test_t = Stopwatch()
            valid_error = np.mean(net.test_trial(valid_patterns)[0])
            valid_errors.append(valid_error)
            test_error = np.mean(net.test_trial(test_patterns)[0])
            test_errors.append(test_error)
            msg = 'At E#%i, T#%i, train_minibatch_error = %.1f, valid_error = %.1f, test_error = %.1f, n_temperatures = %i' % \
                (epochnum, net.trial_num, train_error, valid_error, test_error, net.n_temperatures(net) or 0)

            if printme:
                print msg
        
            if plotme:
                pattern0 = minibatch_pset.patterns[0].reshape(1, net.n_v)
                net.plot_layers(pattern0, ttl=msg)
                if net.fignum_weights: net.plot_weights(net.w, net.fignum_weights, 'Weights to hidden at E#%i' % epochnum)
                if net.fignum_dweights: net.plot_weights(net.d_w, net.fignum_dweights, 'D weights to hidden at E#%i' % epochnum)
                net.plot_errors(train_errors, valid_errors, test_errors)
                if net.fignum_biases: net.plot_biases(net.a, net.b, net.fignum_biases, 'Biases at E#%i' % epochnum)
                if net.fignum_dbiases: net.plot_biases(net.d_a, net.d_b, net.fignum_dbiases, 'D biases at E#%i' % epochnum)
    
    pause()

