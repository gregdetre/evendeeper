import numpy as np
import scipy.io as spio

"""
Implements a Conditional Restricted Boltzmann Machine, following the article of Graham(2011)
Data should be sorted so that the oldest datapoint is at index = 0
"""


class CRBM():
    def __init__(self, size_hidden, size_visible, lag_vis_vis, lag_vis_hid, sampling_steps):
        # if this is not assert the current implementation wont work, since for example the dynamic bias functions
        # rely on this
        assert (lag_vis_hid > 0)
        assert (lag_vis_vis > 0)

        # initialize the random state
        np.random.seed()

        self.size_hidden = size_hidden
        self.size_visible = size_visible
        self.lag_vis_vis = lag_vis_vis
        self.lag_vis_hid = lag_vis_hid
        self.max_lag = np.maximum(lag_vis_hid, lag_vis_vis)

        # block_size is smallest size of a data vector for trainig
        self.block_size = self.max_lag + 1
        self.sampling_steps = sampling_steps

        # weights between the current visible nodes and past visible nodes
        self.A = 0.01 * np.random.standard_normal(size=(size_visible, lag_vis_vis * size_visible))

        # weights between past visible nodes and the current hidden units
        self.B = 0.01 * np.random.standard_normal(size=(size_hidden, lag_vis_hid * size_visible))

        # weights between current visible and hidden nodes
        self.W = 0.01 * np.random.standard_normal(size=(size_visible, size_hidden))

        # static bias of the hidden nodes
        self.bias_hidden = 0.01 * np.random.uniform(size=(size_hidden, 1))

        # static bias of the visible nodes
        self.bias_visible = 0.01 * np.random.uniform(size=(size_visible, 1))

        self.learning_rate = self.momentum = self.weight_decay = self.num_epoch = self.errorsum = 0

    def train(self, train_data, learning_rate=0.0001, momentum=0.9, weight_decay=0.0002, num_epochs=5):
        """
         Trains the conditional restricted boltzmann machine. The settings for the several parameters were copied from
         Taylors implementation. Data must be a matrix containing one observation per row wich leads to a size of
         T x size_visible.

         Furthermore the data is shuffled to reduce autocorrelation between samples.
        """
        # assert that data has the correct size
        assert (len(train_data[0]) == self.size_visible)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epoch = num_epochs

        # init deltas
        delta_W = np.zeros(shape=self.W.shape)
        delta_A = np.zeros(shape=self.A.shape)
        delta_B = np.zeros(shape=self.B.shape)
        delta_bias_visible = np.zeros(shape=self.bias_visible.shape)
        delta_bias_hidden = np.zeros(shape=self.bias_hidden.shape)

        # init temp values
        delta_W_old = np.zeros(shape=(len(self.W), len(self.W[0])))
        delta_A_old = np.zeros(shape=(len(self.A), len(self.A[0])))
        delta_B_old = np.zeros(shape=(len(self.B), len(self.B[0])))
        delta_bias_visible_old = np.zeros(shape=len(self.bias_visible))
        delta_bias_hidden_old = np.zeros(shape=len(self.bias_hidden))

        length_data = len(train_data)
        num_shuffles = int(np.ceil(length_data / float(self.block_size)))

        for epoch in range(num_epochs):
            for batch_index in np.random.permutation(num_shuffles):
                t = batch_index * self.block_size
                t_minus_m = t + self.block_size

                # if there are not enough samples in the last batch, take from the one before
                if t_minus_m > (length_data - 1):
                    t -= (t_minus_m - length_data + 1)

                # create variable for batch data
                v_t = np.train_data[t,:]
                v_tm = train_data[t+1:t_minus_m,:]

                # calculate dynamic biases
                b_hat = self.dynamic_hidden_bias(v_tm[0:self.lag_vis_hid, :])
                a_hat = self.dynamic_visible_bias(v_tm[0:self.lag_vis_vis, :])

                hid_0_probs = self.gibbs_hidden(v_t, b_hat)
                hid_0_states = hid_0_probs > np.random.uniform(size=self.size_hidden)

                # Gibbs sampling for k steps
                for step in range(self.sampling_steps):
                    vis_k_probs = self.gibbs_visible(hid_0_states, a_hat)
                    hid_k_probs = self.gibbs_hidden(vis_k_probs, b_hat)
                    hid_k_states = hid_k_probs > np.random.uniform(size=self.size_hidden)

                # store old parameters for momentum
                delta_W_old = delta_W
                delta_A_old = delta_A
                delta_B_old = delta_B
                delta_bias_hidden_old = delta_bias_hidden
                delta_bias_visible_old = delta_bias_visible

                # calculate deltas
                # The Kronecker product is not commutative, so changes in the dimension of any have to be reflected
                # by changing to order of the Kronecker
                delta_W = np.reshape(np.kron(v_t, hid_0_probs), delta_W.shape) - \
                          np.reshape(np.kron(vis_k_probs, hid_k_probs), delta_W.shape)

                delta_A = np.reshape(np.kron(v_t, v_tm), delta_A.shape) - \
                          np.reshape(np.kron(vis_k_probs, v_tm), delta_A.shape)

                delta_B = np.reshape(np.kron(hid_0_states, v_tm), delta_B.shape) - \
                          np.reshape(np.kron(hid_k_states, v_tm), delta_A.shape)

                delta_bias_visible = v_t - vis_k_probs
                delta_bias_hidden = hid_0_probs - hid_k_probs

                delta_W = self.compute_update(delta_W, delta_W_old)
                delta_A = self.compute_update(delta_A, delta_A_old)
                delta_B = self.compute_update(delta_B, delta_B_old)
                delta_bias_visible = self.compute_update(delta_bias_visible, delta_bias_visible_old)
                delta_bias_hidden = self.compute_update(delta_bias_hidden, delta_bias_hidden_old)

                delta_W_old = delta_W
                delta_A_old = delta_A
                delta_B_old = delta_B
                delta_bias_visible_old = delta_bias_visible
                delta_bias_hidden_old = delta_bias_hidden

                # calculate updates
                self.W += delta_W
                self.A += delta_A
                self.B += delta_B
                self.bias_hidden += delta_bias_hidden
                self.bias_visible += delta_bias_visible

                # compute error
                error = np.square(v_t - vis_k_probs)
                self.errorsum += error

                # print error
                print("epoch: %{epoch}, error: %{error}" % {'epoch':epoch, 'error':error})

            # print errorsum
            print("epoch: %{epoch}, errorum: %{error}" % {'epoch':epoch, 'error':self.errorsum})

    def compute_update(self, delta, delta_old, epoch):
        momentum = 0 if epoch > 5 else self.momentum
        return self.learning_rate * delta - self.weight_decay * delta + momentum * delta_old

    def dynamic_hidden_bias(self, v_tm):
        dim = (np.prod(v_tm.shape), 1)
        return self.bias_hidden + np.dot(self.B, np.reshape(v_tm, dim))

    def dynamic_visible_bias(self, v_tm):
        dim = (np.prod(v_tm.shape), 1)
        return self.bias_visible + np.dot(self.A, np.reshape(v_tm, dim))

    def gibbs_hidden(self, visible, b_hat):
        return np.array(1.0 / 1.0 + np.exp(-1 * b_hat - np.dot(self.W.T, visible)))

    def gibbs_visible(self, hidden, a_hat):
        # to do this, data must be standardized to unit variance
        return np.random.normal(a_hat + np.dot(self.W, hidden.T), 1, size=self.size_visible)

if __name__ == "__main__":
    # load data
    mat = spio.loadmat('data/data.mat')

    # obtain data
    data = mat['batchdata']

    size_visible = 49
    size_hidden = 150
    lag_vis_vis = 3
    lag_vis_hid = 3
    sampling_steps = 1

    crbm = CRBM(size_hidden, size_visible, lag_vis_vis, lag_vis_hid, sampling_steps)
    crbm.train(data)