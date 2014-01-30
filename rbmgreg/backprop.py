from ipdb import set_trace as pause
import numpy as np
from random import shuffle

from base import Network
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
        d_w_jk = self.lrate * np.outer(act_j, sensitivity_k)
        d_b_k = np.squeeze((self.lrate * sensitivity_k), axis=0)
        return d_w_jk, d_b_k, sensitivity_k

    def delta_weights_middle(self, inp_j, w_jk, sensitivity_k, act_i):
        """
        Changing W_IJ, i.e. the weights from (input or
        hidden) layer I to next (hidden) layer J.
        """
        # assert err_i.shape == inp_i.shape == act_i.shape
        sensitivity_j = self.deriv_act_fn(inp_j) * np.dot(w_jk, sensitivity_k.T).T
        d_w_ij = self.lrate * np.outer(act_i, sensitivity_j)
        # assert d_w_ij.shape == w_ij
        d_b_j = np.squeeze((self.lrate * sensitivity_j), axis=0)
        return d_w_ij, d_b_j, sensitivity_j

    def act_fn(self, x): return sigmoid(x)
    def deriv_act_fn(self, x): return deriv_sigmoid(x)
    
    def err_fn(self, actual, desired): return (actual - desired)**2


def arr_str(arr): return np.array_str(arr, precision=2)

def test_epoch(net, data, verbose=True):
    errors = []
    for act0, target in data:
        error, out = net.test_trial(act0, target)
        errors.append(error)
        if verbose:
            print '%s, %s -> %s, err = %.2f' % (arr_str(act0), arr_str(target), arr_str(out), error)
    mean_error = np.mean(errors)
    print '%i) err = %.2f' % (e, mean_error)
    return errors, mean_error

def data_xor():
    data = [[[0,0], [0]],
            [[0,1], [1]],
            [[1,0], [1]],
            [[1,1], [0]]
            ]
    for d,[a,t] in enumerate(data):
        data[d] = (vec_to_arr(np.array(a)), vec_to_arr(np.array(t)))        
    net = BackpropNetwork([2, 2, 1], lrate=0.01)
    # net = BackpropNetwork([2, 5, 1], lrate=0.01)
    return net, data

def data_rand_autoencoder():
    data = []
    n_inp_out = 4
    nhidden = 12
    npatterns = 10
    for d in range(npatterns):
        cur = np.random.uniform(size=(1,n_inp_out))
        data.append((cur,cur))        
    # net = BackpropNetwork([n_inp_out, nhidden, n_inp_out])
    net = BackpropNetwork([n_inp_out, nhidden*2, nhidden, n_inp_out])
    return net, data


if __name__ == "__main__":
    np.random.seed()
    # net, data = data_xor()
    net, data = data_rand_autoencoder()
    nEpochs = 100000
    report_every = 1000
    for e in range(nEpochs):
        cur_data = data[:]
        shuffle(cur_data)
        while cur_data:
            act0, target = cur_data.pop()
            net.learn_trial(act0, target)
        if not e % report_every:
            error, mean_error = test_epoch(net, data, False)
        if mean_error < 0.001: break
    errors, mean_error = test_epoch(net, data, True)
    pause()
