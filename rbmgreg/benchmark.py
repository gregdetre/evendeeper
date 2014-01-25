import numpy as np

from base import Minibatch, Patternset
from rbm import create_mnist_patternset, RbmNetwork
from utils.dt import dt_str
from utils.stopwatch import Stopwatch
from utils.utils import HashableDict

# TODO
#
# - test on withheld data


if __name__ == "__main__":
    npatterns = 1000
    pset = create_mnist_patternset(npatterns=npatterns)
    # xxx - this isn't a proper test since we're using the training data, rather than withheld...
    n_in_test_minibatch = 1000
    test_minibatch = Minibatch(pset, n_in_test_minibatch)

    attempts = [
        {'lrate': 0.001,
         'wcost': 0.0002,
         'nhidden': 400,
         'npatterns': npatterns,
         'n_in_train_minibatch': 20,
         'n_train_epochs': 10000,
         'should_plot': False,
         },
        {'lrate': 0.005,
         'wcost': 0.0002,
         'nhidden': 400,
         'npatterns': npatterns,
         'n_in_train_minibatch': 20,
         'n_train_epochs': 10000,
         'should_plot': False,
         },
        {'lrate': 0.02,
         'wcost': 0.0002,
         'nhidden': 400,
         'npatterns': npatterns,
         'n_in_train_minibatch': 20,
         'n_train_epochs': 10000,
         'should_plot': False,
         },
        ]
    
    nattempts = len(attempts)
    for attempt_idx, attempt in enumerate(attempts):
        np.random.seed(1)
        t = Stopwatch()
        net = RbmNetwork(np.prod(pset.shape),
                         attempt['nhidden'],
                         attempt['lrate'],
                         attempt['wcost'],
                         v_shape=pset.shape,
                         plot=False)
        for epochnum in range(attempt['n_train_epochs']):
            # tenpercent =  / attempt['n_train_epochs']
            # if epochnum and not epochnum % 
            train_minibatch = Minibatch(pset, attempt['n_in_train_minibatch'])
            [d_w, d_a, d_b] = net.learn_trial(train_minibatch.patterns)
        train_elapsed = t.finish(milli=False)
        test_errors, _ = net.test_trial(test_minibatch.patterns)
        test_error = np.mean(test_errors)
        print 'Finished %i of %i: error %.2f in %.1f seconds' % (attempt_idx+1, nattempts, test_error, train_elapsed)
        print attempt
        print '--------------------'
        print


    
