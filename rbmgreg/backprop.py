from ipdb import set_trace as pause
import numpy as np
from random import shuffle

from base import Minibatch2, Network, Patternset
from datasets import load_mnist
from utils.utils import deriv_sigmoid, imagesc, sigmoid, sumsq, vec_to_arr


class BackpropNetwork(Network):
    def __init__(self, layersizes, lrate=0.01, momentum=0.9):
        assert len(layersizes) >= 3 # incl inpout & output
        self.layersizes = layersizes
        self.w = self.init_weights(scale=1)
        self.d_w = [np.zeros_like(cur_w) for cur_w in self.w]
        self.b = [np.zeros((n,)) for n in self.layersizes]
        self.d_b = [np.zeros_like(cur_b) for cur_b in self.b]
        self.n_l = len(self.layersizes)
        self.momentum = momentum
        self.lrate = lrate

    def init_weights(self, scale=0.1):
        return [np.random.normal(size=(n1,n2), scale=scale)
                for n1,n2 in zip(self.layersizes, self.layersizes[1:])]

    def test_trial(self, act0, target):
        inps, acts = self.propagate_fwd_all(act0)
        out_act = acts[-1]
        error = self.report_error(out_act, target)
        return error, out_act

    def propagate_fwd_all(self, act0):
        acts = [act0]
        inps = [act0]
        for l_idx in range(self.n_l-1):
            inp, act = self.propagate_fwd(l_idx, acts[-1])
            inps.append(inp)
            acts.append(act)
        # pause()
        return inps, acts

    def propagate_fwd(self, lowlayer_idx, act):
        w, b = self.w[lowlayer_idx], self.b[lowlayer_idx+1]
        return super(BackpropNetwork, self).propagate_fwd(act, w, b)

    def report_error(self, actual, desired):
        return np.sum((actual - desired)**2, axis=1)

    def learn_trial(self, act0, target):
        d_w, d_b = self.delta_weights(act0, target)
        # self.w += d_w + self.momentum*self.d_w
        for cur_w, new_d_w, old_d_w in zip(self.w, d_w, self.d_w):
            cur_w += new_d_w + self.momentum*old_d_w
        for cur_b, new_d_b, old_d_b in zip(self.b, d_b, self.d_b):
            cur_b += new_d_b + self.momentum*old_d_b
        self.d_w, self.d_b = d_w, d_b
        return self.d_w

    def delta_weights(self, act0, tgt_k):
        d_w = [np.zeros_like(cur_d_w) for cur_d_w in self.d_w]
        d_b = [np.zeros_like(cur_d_b) for cur_d_b in self.d_b]
        inps, acts = self.propagate_fwd_all(act0)
        act_k = acts[-1]
        err_k = tgt_k - act_k
        inp_k = inps[-1]
        act_j = acts[-2]
        l_idx_uppermost = self.n_l - 1
        l_idx_penult = l_idx_uppermost - 1
        d_w_jk, d_b_k, sensitivity_k = self.delta_weights_uppermost(err_k, inp_k, act_j)
        d_w[l_idx_penult] = d_w_jk
        d_b[l_idx_uppermost] = d_b_k
        for l_idx_hidden in reversed(range(l_idx_penult)):
            w_jk = self.w[l_idx_hidden+1]
            inp_j = inps[l_idx_hidden+1]
            act_i = acts[l_idx_hidden]
            d_w_ij, d_b_j, sensitivity_j = self.delta_weights_middle(inp_j, w_jk, sensitivity_k, act_i)
            d_w[l_idx_hidden] = d_w_ij
            d_b[l_idx_hidden+1] = d_b_j
            sensitivity_k = sensitivity_j # for use in the next iteration
        return d_w, d_b

    def delta_weights_uppermost(self, err_k, inp_k, act_j):
        """
        Changing W_JK, i.e. the weights from penultimate
        (hidden) layer J to uppermost (output) layer K.
        """
        sensitivity_k = err_k * self.deriv_act_fn(inp_k)
        d_w_jk = self.lrate * np.dot(act_j.T, sensitivity_k)
        npatterns = act_j.shape[0]
        d_b_k = np.mean(self.lrate * sensitivity_k, axis=0)
        return d_w_jk, d_b_k, sensitivity_k

    def delta_weights_middle(self, inp_j, w_jk, sensitivity_k, act_i):
        """
        Changing W_IJ, i.e. the weights from (input or
        hidden) layer I to next (hidden) layer J.
        """
        # assert err_i.shape == inp_i.shape == act_i.shape
        sensitivity_j = self.deriv_act_fn(inp_j) * np.dot(w_jk, sensitivity_k.T).T
        d_w_ij = self.lrate * np.dot(act_i.T, sensitivity_j)
        # assert d_w_ij.shape == w_ij
        d_b_j = np.mean(self.lrate * sensitivity_j, axis=0)
        return d_w_ij, d_b_j, sensitivity_j

    def act_fn(self, x): return sigmoid(x)
    def deriv_act_fn(self, x): return deriv_sigmoid(x)
    
    def err_fn(self, actual, desired): return (actual - desired)**2


def arr_str(arr): return np.array_str(arr, precision=2)

def test_epoch(net, acts0, targets, verbose=True):
    errors = []
    for act0, target in zip(acts0, targets):
        error, out = net.test_trial(act0, target)
        errors.append(error)
        if verbose:
            print '%s, %s -> %s, err = %.2f' % (arr_str(act0), arr_str(target), arr_str(out), error)
    mean_error = np.mean(errors)
    print '%i) err = %.2f' % (e, mean_error)
    return errors, mean_error

def xor(*args, **kwargs):
    data = [[[0,0], [0]],
            [[0,1], [1]],
            [[1,0], [1]],
            [[1,1], [0]]
            ]
    iset = Patternset([vec_to_arr(np.array(a)) for a,t in data])
    oset = Patternset([vec_to_arr(np.array(t)) for a,t in data])
    net = BackpropNetwork([2, 2, 1], *args, **kwargs)
    return net, iset, oset

def autoencoder(pset, nhiddens, *args, **kwargs):
    n_inp_out = np.prod(pset.shape)
    layersizes = [n_inp_out] + nhiddens + [n_inp_out]
    net = BackpropNetwork(layersizes, *args, **kwargs)
    return net, pset, pset

def rand_autoencoder():
    n_inp_out = 4
    npatterns = 10
    nhiddens = [12]
    pset = Patternset([np.random.uniform(size=(1,n_inp_out)) for d in range(npatterns)])
    return autoencoder(pset, nhiddens=nhiddens)


if __name__ == "__main__":
    np.random.seed()
    # net, iset, oset = xor(lrate=0.1)
    net, iset, oset = rand_autoencoder()
    nEpochs = 2000 # 100000
    n_in_minibatch = min(len(iset), 20)
    report_every = 1000
    for e in range(nEpochs):
        act0, target = Minibatch2(iset, oset, n_in_minibatch).patterns
        net.learn_trial(act0, target)
        if not e % report_every:
            error, mean_error = test_epoch(net, iset.patterns, oset.patterns, False)
        if mean_error < 0.01: break
    errors, mean_error = test_epoch(net, iset.patterns, oset.patterns, True)
    pause()
