from argparse import ArgumentParser
from ipdb import set_trace as pause
import numpy as np
import os

from base import Minibatch, Patternset, create_mnist_patternsets
from rbm import RbmNetwork
from utils.dt import dt_str, eta_str
from utils.stopwatch import Stopwatch
from utils.utils import HashableDict

"""
USAGE: e.g.

  python benchmark.py -h

  python benchmark.py -s 3 --lrate 0.001 --lrate 0.01 --momentum 0.1 --momentum 0.2 --nhidden 1000
  ->
    {'nhidden': 1000, 'wcost': 0.0002, 'momentum': 0.1, 'lrate': 0.001}
    {'nhidden': 1000, 'wcost': 0.0002, 'momentum': 0.2, 'lrate': 0.001}
    {'nhidden': 1000, 'wcost': 0.0002, 'momentum': 0.1, 'lrate': 0.01}
    {'nhidden': 1000, 'wcost': 0.0002, 'momentum': 0.2, 'lrate': 0.01}
"""

# TODO
#
# - test on withheld data
# - write out to sqlite, pickle
# - plot on graph
# - run on c3.large

def print_best(attempts, keys):
    attempts = sorted_attempts(attempts)
    print 'rank\terror\t' + '\t'.join(keys)
    for rank, attempt in enumerate(attempts):
        print ('%i\t%.2f\t' % (rank+1, attempt['test_error'])) +'\t'.join([str(attempt[k]) for k in keys])

def sorted_attempts(attempts):
    attempts = sorted(attempts, key=lambda x: x['test_error'])
    return attempts

def gridsearch(nhidden, lrate, wcost, momentum, n_in_train_minibatch, sampling_steps, n_temperatures, max_time_secs):
    params = ['nhidden', 'lrate', 'wcost', 'momentum', 'n_in_train_minibatch'] # add 'sampling_steps', 'n_temperatures'
    attempts = []
    for nh in nhidden: # [200, 400, 800]:
        for lr in lrate: # [0.005, 0.02]:
            for wc in wcost: #, 0.002]:
                for mo in momentum: # [0, 0.4, 0.9]:
                    for nitm in n_in_train_minibatch: # [10, 250]:
                        attempts.append({'nhidden': nh,
                                         'lrate': lr,
                                         'wcost': wc,
                                         'momentum': mo,
                                         'n_in_train_minibatch': nitm,
                                         'should_plot': False,
                                         })
    
    nattempts = len(attempts)
    pid = os.getpid()
    total_time_secs = max_time_secs * nattempts
    print 'Beginning %i attempts (PID=%s, DT=%s), each for %i secs, ETA = %s' % \
        (nattempts, pid, dt_str(), max_time_secs, eta_str(total_time_secs))
    for attempt in attempts: print attempt

    # Generate data set
    n_trainpatterns = 5000
    n_validpatterns = 1000
    train_pset, valid_pset, test_pset = create_mnist_patternsets(n_trainpatterns=n_trainpatterns, n_validpatterns=n_validpatterns)
    n_trainpatterns, n_validpatterns, n_testpatterns = train_pset.n, valid_pset.n, test_pset.n
    valid_patterns = np.array(valid_pset.patterns).reshape((n_validpatterns,-1))
    test_patterns = np.array(test_pset.patterns).reshape((n_testpatterns, -1))

    all_t = Stopwatch()
    for attempt_idx, attempt in enumerate(attempts):
        np.random.seed(1)
        t = Stopwatch()
        net = RbmNetwork(np.prod(train_pset.shape),
                         attempt['nhidden'],
                         attempt['lrate'],
                         attempt['wcost'],
                         attempt['momentum'],
                         v_shape=train_pset.shape,
                         plot=True)
        train_errors = []
        valid_errors = []
        test_errors = []
        epochnum = 0
        while True: # train as long as time limit is not exceeded
            minibatch_pset = Minibatch(train_pset, nitm)
            net.learn_trial(minibatch_pset.patterns)

            # calculate errors
            train_error = np.mean(net.test_trial(minibatch_pset.patterns)[0])
            train_errors.append(train_error)
            valid_error = np.mean(net.test_trial(valid_patterns)[0])
            valid_errors.append(valid_error)
            test_error = np.mean(net.test_trial(test_patterns)[0])
            test_errors.append(test_error)

            if t.finish(milli=False) > max_time_secs: break
            epochnum += 1

        train_elapsed = t.finish(milli=False)

        # add ATTEMPT keys that will vary each time below
        # this line. the HASH should be the same for this
        # set of parameters every time you run the benchmark
        attempt['hash'] = hash(HashableDict(attempt))
        attempt['attempt_idx'] = attempt_idx
        attempt['pid'] = pid
        attempt['n_train_epochs'] = epochnum
        attempt['train_elapsed'] = train_elapsed
        attempt['dt'] = dt_str()
        attempt['test_error'] = train_errors[-1] # error from last iteration
        attempt['npatterns'] = n_trainpatterns

        filename = 'error%i.png' % attempt_idx
        net.save_error_plots(train_errors, valid_errors, test_errors, filename)

        print 'Finished %i of %i: error %.2f in %.1f secs' % (attempt_idx+1, nattempts, test_error, train_elapsed)
        print attempt
        print '--------------------'
        print
    
    print_best(attempts, params)
    
if __name__ == "__main__":
    # Parameters for testing
    nhidden = [100]
    lrate = [0.001, 0.0001]
    wcost = [0.0002]
    momentum = [0.9]
    n_in_train_minibatch = [10]
    sampling_steps = [] # CD-k
    n_temperatures = [] # For single tempering, insert 1
    max_time_secs = 300

    gridsearch(nhidden, lrate, wcost, momentum, n_in_train_minibatch, \
                                sampling_steps, n_temperatures, max_time_secs)
