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
        self.b = [np.zeros(n) for n in layersizes]
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
        # d_w, d_a, d_b = self.delta_weights(act0, target)
        d_w = self.delta_weights(act0, target)
        # self.w += d_w + self.momentum*self.d_w
        for cur_w, cur_d_w in zip(self.w, d_w):
            cur_w += cur_d_w + self.momentum*cur_d_w
        # self.a += d_a + self.momentum*self.d_a
        # self.b += d_b + self.momentum*self.d_b
        # self.d_w, self.d_a, self.d_b = d_w, d_a, d_b
        self.d_w = d_w
        return self.d_w

    def delta_weights(self, act0, tgt_k):
        d_w = [np.zeros_like(cur_d_w) for cur_d_w in self.d_w]
        inps, acts = self.propagate_fwd_all(act0)
        act_k = acts[-1]
        err_k = tgt_k - act_k
        errors = self.propagate_err_back_all(err_k)
        inp_k = inps[-1]
        act_j = acts[-2]
        l_idx_upper = self.n_l - 1
        l_idx_penult = l_idx_upper - 1
        d_w_jk, sensitivity_k = self.delta_weights_uppermost(err_k, inp_k, act_j)
        d_w[l_idx_penult] = d_w_jk
        w_jk = self.w[-1]
        for l_idx_hidden in reversed(range(l_idx_penult)):
            w_ij = self.w[l_idx_hidden]
            err_i = errors[l_idx_hidden]
            inp_j = inps[l_idx_hidden+1]
            inp_i = inps[l_idx_hidden]
            act_i = acts[l_idx_hidden]
            d_w_ij, sensitivity_j = self.delta_weights_middle(inp_j, w_jk, sensitivity_k, act_i)
            d_w[l_idx_hidden] = d_w_ij
        return d_w

#         errors.append(error)
#         d_ws = []
#         for l_idx in reversed(range(self.n_l-1)):
#             l_act = acts[l_idx]
#             u_act = acts[l_idx+1]
#             # like PROPAGATE_BACK, but without bias or activation function
#             w, b = self.w[l_idx], self.b[l_idx]
#             d_w = np.zeros_like(w)
#             # dn_error = np.dot(w, up_error) + b # xxx incl B???
#             u_error = errors[-1]
#             l_error = np.dot(w, u_error)
#             errors.append(l_error)
#             for i in range(self.layersizes[l_idx]): # NUM_LOWER
#                 for j in range(self.layersizes[l_idx+1]): # NUM_UPPER
#                     d_w[i,j] = self.lrate * u_error[j] * l_act[i]
#             d_ws.append(d_w)
#         # d_a, d_b =
#         # we want to return D_WS bottom up
#         return reversed(d_ws)

    def propagate_err_back(self, err_j, w_ij): return np.dot(w_ij, err_j.T).T

    def propagate_err_back_all(self, err_k):
        errors = [err_k]
        for l_idx in reversed(range(self.n_l-1)):
            w_ij = self.w[l_idx]
            err_j = errors[0]
            assert w_ij.shape[1] == err_j.shape[1]
            err_i = self.propagate_err_back(err_j, w_ij)
            assert w_ij.shape[0] == err_i.shape[1]
            errors.insert(0, err_i) # prepend, so ERRORS stores bottom-up
        return errors

    def delta_weights_uppermost(self, err_k, inp_k, act_j):
        """
        Changing W_JK, i.e. the weights from penultimate
        (hidden) layer J to uppermost (output) layer K.
        """
        sensitivity_k = err_k * self.deriv_act_fn(inp_k)
        d_w_jk = self.lrate * np.outer(act_j, sensitivity_k)
        return d_w_jk, sensitivity_k

    def delta_weights_middle(self, inp_j, w_jk, sensitivity_k, act_i):
        """
        Changing W_IJ, i.e. the weights from (input or
        hidden) layer I to next (hidden) layer J.
        """
        # assert err_i.shape == inp_i.shape == act_i.shape
        sensitivity_j = self.deriv_act_fn(inp_j) * np.dot(w_jk, sensitivity_k.T).T
        d_w_ij = self.lrate * np.outer(act_i, sensitivity_j)
        # assert d_w_ij.shape == w_ij
        return d_w_ij, sensitivity_j

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
    # net = BackpropNetwork([2, 2, 1], lrate=0.01)
    net = BackpropNetwork([2, 3, 1], lrate=0.01)
    return net, data

def data_rand_autoencoder():
    data = []
    n_inp_out = 4
    nhidden = 12
    npatterns = 10
    for d in range(npatterns):
        # cur = np.random.normal(size=(1,n_inp_out))
        cur = np.random.uniform(size=(1,n_inp_out))
        data.append((cur,cur))        
    net = BackpropNetwork([n_inp_out, nhidden, n_inp_out])
    return net, data


if __name__ == "__main__":
    np.random.seed()
    net, data = data_xor()
    # net, data = data_rand_autoencoder()
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
