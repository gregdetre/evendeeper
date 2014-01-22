import numpy as np
import time

class BinaryRBM():
    """
    This class implements a binary Restricted Boltzman Machine (binary RBM). 
    It follows the paper of Fischer A., Igel C.: "An introduction to Restricted 
    Boltzman Machines", CIARP, pp 14 - 36, 2012
    """
    def __init__(self, size_visible_layer, size_hidden_layer, batch_size=10):
        self.size_hidden = size_hidden_layer
        self.size_visible = size_visible_layer
        self.training_size = 0
        self.batch_size = batch_size

        self.weights = np.random.normal(0,0.01,size=(size_hidden_layer, size_visible_layer))
        self.hidden_bias = np.zeros(shape=size_hidden_layer)
        self.visible_bias = np.random.uniform(0,0.5,size=size_visible_layer)

        # deltas that will be update during the process of calling update_contrasitve_divergence
        self.delta_weights = np.zeros(shape=self.weights.shape)
        self.delta_hidden_bias = np.zeros(shape=self.hidden_bias.shape)
        self.delta_visible_bias = np.zeros(shape=self.visible_bias.shape)


    def initialize_weight_matrix(self, weight_matrix):
        """
        Initializes the weight matrix for this RBM. This is useful if this RBM 
        is part of a larger network or should be initialized in a certain way.

        @ param 1 weight_matrix is the weight matrix for initialization
        """
        if not (weight_matrix.shape == self.weights.shape):
            raise Exception("Incorrect input size.")

        self.weights = np.copy(weight_matrix)

    def mini_batch_contrastive_divergence(self, data_matrix, learning_rate=0.1, momentum=0, weight_decay=0):
        """
        Performs k-step contrastive divergence on given batch
        """
        if not (data_matrix.shape == (self.batch_size,self.size_visible)):
            raise Exception("Incorrect input size.")

        # saving the old values for momentum calculations
        old_delta_hidden_bias = self.delta_hidden_bias / float(self.batch_size)
        old_delta_visible_bias = self.delta_visible_bias / float(self.batch_size)
        old_delta_weights = self.delta_weights / float(self.batch_size)

        # resetting the deltas
        self.delta_weights = np.zeros(shape=self.weights.shape)
        self.delta_hidden_bias = np.zeros(shape=self.hidden_bias.shape)
        self.delta_visible_bias = np.zeros(shape=self.visible_bias.shape)

        error = 0
        for batch_index in xrange(self.batch_size):
            start_t = time.clock()
            error += self.update_contrastive_divergence(data_matrix[batch_index])
            stop_t = time.clock()
            print "CD_1 step (sec): %f" % (stop_t - start_t)

        start_t = time.clock()
        # update hidden bias
        self.hidden_bias += (learning_rate / float(self.batch_size)) * self.delta_hidden_bias - weight_decay \
                            * self.hidden_bias + momentum * old_delta_hidden_bias
        # update visible bias
        self.visible_bias += (learning_rate / float(self.batch_size)) * self.delta_visible_bias - weight_decay \
                            * self.visible_bias + momentum * old_delta_visible_bias
        # update weights
        self.weights += (learning_rate / float(self.batch_size)) * self.delta_weights \
                            - weight_decay * self.weights + momentum * old_delta_weights
        stop_t = time.clock()
        print "Updating parameters (sec): %f" % (stop_t - start_t)

        return error / float(self.training_size)


    def update_contrastive_divergence(self, v_0):
        """
        Computes deltas

        @ param data_array: a dataset of class Dataset from datasets.py used for training
        """
        self.training_size += 1

        # Gibbs sampling step
        v_k = self.gibbs_sample(v_start=v_0)

        # Obtain P(H_i = 1 | v)
        probs_0 = self.get_hidden_probs(v_0)
        probs_k = self.get_hidden_probs(v_k)

        #hidden bias gradient update
        self.delta_hidden_bias += probs_0 - probs_k

        # visible bias gradient update
        self.delta_visible_bias += v_0 - v_k

        # weight gradient update
        for i in xrange(self.size_hidden):
            self.delta_weights[i] += probs_0[i] * v_0 - probs_k[i] * v_k

        return np.sum(np.absolute(v_0 - v_k))

    def gibbs_sample(self, v_start, k=1):
        """
        Performs block Gibbs sampling k-times
        """
        # init the sampler
        h_t = self.get_hidden_state(v_in=v_start)

        # block sampling (remind conditional independence must hold)
        for i in xrange(k):
            v_t = self.get_visible_state(h_in=h_t)
            h_t = self.get_hidden_state(v_in=v_t)

        return v_t

    def get_hidden_state(self, v_in):
        """
        Samples hidden states (remember: states are binary at this point)

        Returns values from a sigmoid activation function inline and for an 
        entire vector.
        Activation function: P(H_i = 1 | v) = sigm(sum_j=1_m(w_i_j * v_j + c_i))

        @ param v_in: visible unit vector to obtain hidden probabilits
        @ param h_out: vector holding the hidden state
        """
        return self.get_hidden_probs(v_in) > np.random.uniform(low=0.0,high=1.0,size=self.size_hidden)

    def get_hidden_probs(self, v_in):
        """
        Calculates P(H_i = 1 | v) = sigm(sum_j=1_m(w_i_j * v_j + c_i))
        """
        probs = np.zeros(self.size_hidden)
        for i in xrange(self.size_hidden):
            inner_sum = 0
            for j in xrange(self.size_visible):
                inner_sum += self.weights[i][j] * v_in[j] + self.hidden_bias[i]
            probs[i] = self.sigmoid_activation(inner_sum)

        return probs

    def get_visible_state(self, h_in):
        """
        Samples visible states (remember: states are binary at this point)
        """
        return self.get_visible_probs(h_in) > np.random.uniform(low=0.0,high=1.0,size=self.size_visible)

    def get_visible_probs(self, h_in):
        """
        Calculates P(V_j = 1 | h) = sigm(sum_i=1_n(w_i_j * h_i + b_j))
        """
        probs = np.zeros(self.size_visible)
        for j in xrange(self.size_visible):
            inner_sum = 0
            for i in xrange(self.size_hidden):
                inner_sum += self.weights[i][j] * h_in[i] + self.visible_bias[j]
            probs[j] = self.sigmoid_activation(inner_sum)

        return probs

    def sigmoid_activation(self, x):
        """
        Computes sigmoid function for given x
        """
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def get_free_energy(self):
        pass
