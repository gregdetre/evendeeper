from ipdb import set_trace as pause
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import time

from base import Network, Patternset
from datasets import load_mnist
from utils import imagesc, sigmoid, sumsq, vec_to_arr


# TODO
#


class RbmNetwork(Network):
    def __init__(self, n_v, n_h, lrate):
        self.n_v, self.n_h = n_v, n_h
        self.lrate = lrate
        self.w = self.init_weights(n_v, n_h)
        self.a = 0 # bias to visible
        self.b = 0 # bias to hidden

    def init_weights(self, n_v, n_h, high=0.01):
        # return np.random.uniform(size=(n_v, n_h), high=high)
        return np.random.normal(size=(n_v, n_h)) * high

    def act_fn(self, x): return sigmoid(x)

    def test_trial(self, v_plus):
        h_plus_inp, h_plus_act,  = self.propagate_fwd(v_plus)
        v_minus_inp, v_minus_act = self.propagate_back(h_plus_act)
        error = self.calculate_error(v_minus_act, v_plus)
        return error, v_minus_act

    def learn_trial(self, v_plus): return self.update_weights(v_plus)

    def calculate_error(self, actual, desired):
        return np.mean(np.abs(actual - desired))

    def gibbs_step(self, v_plus, verbose=False):
        h_plus_inp, h_plus_act = self.propagate_fwd(v_plus)
        h_plus_prob = self.probsample(h_plus_act)
        # h_plus_prob = np.round(h_plus_act)
        v_minus_inp, v_minus_act = self.propagate_back(h_plus_prob)
        v_minus_prob = self.probsample(v_minus_act)
        h_minus_inp, h_minus_act = self.propagate_fwd(v_minus_prob)
        h_minus_prob = self.probsample(h_minus_act)
        if verbose: return \
                h_plus_inp, h_plus_act, h_plus_prob, \
                v_minus_inp, v_minus_act, v_minus_prob, \
                h_minus_inp, h_minus_act, h_minus_prob
        else: return h_plus_prob, v_minus_prob, h_minus_prob

    def probsample(self, x): return np.random.uniform(size=x.shape) > x

    def update_weights(self, v_plus):
        h_plus, v_minus, h_minus = self.gibbs_step(v_plus)
        dw = np.zeros(self.w.shape)
        # d_a = self.a * 0.
        # d_b = self.b * 0.
        for i in range(self.n_v):
            for j in range(self.n_h):
                dw[i,j] = self.lrate * (v_plus[i]*h_plus[j] - v_minus[i]*h_minus[j])
        self.w += dw
        return dw

    def plot_trial(self, v_plus, v_shape=None, ttl=None):
        if v_shape is None: v_shape = (self.n_v, 1)
        h_plus_inp, h_plus_act, h_plus_prob, \
            v_minus_inp, v_minus_act, v_minus_prob, \
            h_minus_inp, h_minus_act, h_minus_prob = self.gibbs_step(v_plus, verbose=True)
        v_plus = v_plus.reshape(v_shape)
        h_plus_inp = vec_to_arr(h_plus_inp)*1.
        h_plus_act = vec_to_arr(h_plus_act)*1.
        h_plus_prob = vec_to_arr(h_plus_prob)*1.
        v_minus_inp = v_minus_inp.reshape(v_shape)
        v_minus_act = v_minus_act.reshape(v_shape)
        v_minus_prob = v_minus_prob.reshape(v_shape)
        h_minus_inp = vec_to_arr(h_minus_inp)*1.
        h_minus_act = vec_to_arr(h_minus_act)*1.
        h_minus_prob = vec_to_arr(h_minus_prob)*1.

        vmax = max(map(lambda x: max(x.ravel()), [h_plus_inp, h_plus_act, h_plus_prob,
                                                  v_minus_inp, v_minus_act, v_minus_prob,
                                                  h_minus_inp, h_minus_act, h_minus_prob]))
        vmin = min(map(lambda x: max(x.ravel()), [h_plus_inp, h_plus_act, h_plus_prob,
                                                  v_minus_inp, v_minus_act, v_minus_prob,
                                                  h_minus_inp, h_minus_act, h_minus_prob]))
        
        # fig = plt.figure(figsize=(6,9))
        plt.clf()
        if ttl: fig.suptitle(ttl)
        gs = gridspec.GridSpec(15,2)

        # top left downwards
        ax = fig.add_subplot(gs[  0, 0]); imagesc(h_plus_prob, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_prob')
        ax = fig.add_subplot(gs[  1, 0]); imagesc(h_plus_act, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_act')
        ax = fig.add_subplot(gs[  2, 0]); imagesc(h_plus_inp, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_inp')
        ax = fig.add_subplot(gs[4:7, 0]); imagesc(v_plus, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_plus')

        # top right downwards
        ax = fig.add_subplot(gs[   0, 1]); imagesc(h_minus_prob, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_prob')
        ax = fig.add_subplot(gs[   1, 1]); imagesc(h_minus_act, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_act')
        ax = fig.add_subplot(gs[   2, 1]); imagesc(h_minus_inp, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_inp')
        ax = fig.add_subplot(gs[4: 7, 1]); imagesc(v_minus_prob*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_prob')
        ax = fig.add_subplot(gs[8: 11,1]); imagesc(v_minus_act*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_act')
        ax = fig.add_subplot(gs[12:15,1]); imagesc(v_minus_inp*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_inp')

#         # top left downwards
#         ax = fig.add_subplot(6,2,1); imagesc(h_plus_prob, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_prob')
#         ax = fig.add_subplot(6,2,3); imagesc(h_plus_act, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_act')
#         ax = fig.add_subplot(6,2,5); imagesc(h_plus_inp, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_plus_inp')
#         ax = fig.add_subplot(6,2,7); imagesc(v_plus, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_plus')

#         # top right downwards
#         ax = fig.add_subplot(6,2,2); imagesc(h_minus_prob, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_prob')
#         ax = fig.add_subplot(6,2,4); imagesc(h_minus_act, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_act')
#         ax = fig.add_subplot(6,2,6); imagesc(h_minus_inp, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('h_minus_inp')
#         ax = fig.add_subplot(6,2,8); imagesc(v_minus_prob*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_prob')
#         ax = fig.add_subplot(6,2,10); imagesc(v_minus_act*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_act')
#         ax = fig.add_subplot(6,2,12); imagesc(v_minus_inp*1, dest=ax, vmin=vmin, vmax=vmax); ax.set_title('v_minus_inp')

        plt.draw()
        time.sleep(.5)
    
    def plot_errors(self, errors):
        plt.plot(errors)
        plt.ylim(ymin=0, ymax=max(errors)*1.1)
        plt.show()


def create_random_patternset(shape=(8,2), npatterns=5):
    iset = [np.random.rand(*shape) for _ in range(npatterns)]
    return Patternset(iset)

# def create_stripe_patternset():
#     input0 = np.array([[.5,1,.2,0,0,0],
#                        [.5,1,.2,0,0,0]])
#     input1 = np.array([[0,.5,1,.2,0,0],
#                        [0,.5,1,.2,0,0]])
#     input2 = np.array([[0,0,.5,1,.2,0],
#                        [0,0,.5,1,.2,0]])
#     input3 = np.array([[0,0,0,.5,1,.2],
#                        [0,0,0,.5,1,.2]])
#     pset = Patternset([input0, input1, input2, input3])
#     return pset

def create_stripe_patternset():
    input0 = np.array([[1,1,0,0,0,0,0,0],
                       [1,1,0,0,0,0,0,0]])
    input1 = np.array([[0,0,1,1,0,0,0,0],
                       [0,0,1,1,0,0,0,0]])
    input2 = np.array([[0,0,0,0,1,1,0,0],
                       [0,0,0,0,1,1,0,0]])
    input3 = np.array([[0,0,0,0,0,0,1,1],
                       [0,0,0,0,0,0,1,1]])
    pset = Patternset([input0, input1, input2, input3])
    return pset

def create_mnist_patternset(npatterns=5):
    print 'Loading mnist...',
    mnist_ds = load_mnist(filen='../rbm/minst_train.csv', nrows=npatterns)
    assert mnist_ds.X.shape[0] == npatterns
    pset = Patternset(mnist_ds.X, shape=(28,28))
    print 'done'
    return pset


if __name__ == "__main__":
    nhidden = 30
    npatterns = 5
    train_errors = []
    train_v_minuses = []
    n_train_epochs = 100

    # pset = create_random_patternset(npatterns=npatterns)
    # pset = create_stripe_patternset()
    pset = create_mnist_patternset(npatterns=npatterns)

    net = RbmNetwork(np.prod(pset.shape), nhidden, 0.01)
    # pset = create_random_patternset()
    print net
    print pset

    fig = plt.figure(figsize=(6,9))
    for epochnum in range(n_train_epochs):
        # patnum = random.randint(0,npatterns-1)
        patnum = 0
        pattern = pset.get(patnum)
        dw = net.learn_trial(pattern)
        error, v_minus = net.test_trial(pattern)
        msg = 'After %i epochs, error on P%i = %.2f' % (epochnum, patnum, error)
        print msg
        # pset.imshow(v_minus)
        
        if patnum == 0:
            # plt.close('all')
            # fig = plt.gcf()
            net.plot_trial(pattern, v_shape=pset.shape, ttl=msg)
            # plt.close(fig)
            # pause()

        train_errors.append(error)
        train_v_minuses.append(v_minus)

    for patnum in range(npatterns):
        pattern = pset.get(patnum)
        error, v_minus = net.test_trial(pattern)
        print 'End of training, error on P%i = %.2f' % (patnum, error)
        pset.imshow(v_minus)
        pause()

    print '  '.join(['%.2f' % error for error in train_errors])
    plt.figure()
    net.plot_errors(train_errors)
    pause()

