from ipdb import set_trace as pause
import numpy as np
import os

from base import Minibatch, Patternset
from rbm import create_mnist_patternset, RbmNetwork
from utils.dt import dt_str
from utils.stopwatch import Stopwatch
from utils.utils import HashableDict

# TODO
#
# - test on withheld data


def print_best(attempts, keys):
    attempts = sorted_attempts(attempts)
    print 'rank\terror\t' + '\t'.join(keys)
    for rank, attempt in enumerate(attempts):
        print ('%i\t%.2f\t' % (rank+1, attempt['test_error'])) +'\t'.join([str(attempt[k]) for k in keys])


def sorted_attempts(attempts):
    attempts = sorted(attempts, key=lambda x: x['test_error'])
    return attempts
        

if __name__ == "__main__":
    npatterns = 1000
    pset = create_mnist_patternset(npatterns=npatterns)
    # xxx - this isn't a proper test since we're using the training data, rather than withheld...
    n_in_test_minibatch = 1000
    test_minibatch = Minibatch(pset, n_in_test_minibatch)
    max_time_secs = 100 # how long to give each ATTEMPT

    params = ['lrate', 'wcost', 'nhidden', 'n_in_train_minibatch']
    attempts = []
    for lrate in [0.005, 0.02]:
        for wcost in [0.0002, 0.002]:
            for nhidden in [200, 400, 800]:
                for n_in_train_minibatch in [10, 250]:
                    attempts.append({'lrate': lrate,
                                     'wcost': wcost,
                                     'nhidden': nhidden,
                                     'npatterns': npatterns,
                                     'n_in_train_minibatch': n_in_train_minibatch,
                                     'should_plot': False,
                                     })
    
    nattempts = len(attempts)
    pid = os.getpid()
    print 'Beginning %i attempts (PID=%s, DT=%s)' % (nattempts, pid, dt_str())
    all_t = Stopwatch()
    for attempt_idx, attempt in enumerate(attempts):
        np.random.seed(1)
        t = Stopwatch()
        net = RbmNetwork(np.prod(pset.shape),
                         attempt['nhidden'],
                         attempt['lrate'],
                         attempt['wcost'],
                         v_shape=pset.shape,
                         plot=False)
        epochnum = 0
        while True:
            train_minibatch = Minibatch(pset, attempt['n_in_train_minibatch'])
            [d_w, d_a, d_b] = net.learn_trial(train_minibatch.patterns)
            if t.finish(milli=False) > max_time_secs: break
            epochnum += 1
        train_elapsed = t.finish(milli=False)
        test_errors, _ = net.test_trial(test_minibatch.patterns)
        test_error = np.mean(test_errors)
        # add ATTEMPT keys that will vary each time below
        # this line. the HASH should be the same for this
        # set of parameters every time you run the benchmark
        attempt['hash'] = hash(HashableDict(attempt))
        attempt['attempt_idx'] = attempt_idx
        attempt['pid'] = pid
        attempt['n_train_epochs'] = epochnum
        attempt['train_elapsed'] = train_elapsed
        attempt['dt'] = dt_str
        attempt['test_error'] = test_error
        print 'Finished %i of %i: error %.2f in %.1f seconds' % (attempt_idx+1, nattempts, test_error, train_elapsed)
        print attempt
        print '--------------------'
        print
    

    print_best(attempts, params)
    if all_t.finish(milli=False) > 1000: pause()

    
