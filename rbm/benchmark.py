from argparse import ArgumentParser
from ipdb import set_trace as pause
import numpy as np
import os

from base import Minibatch, Patternset
from rbm import create_mnist_patternset, RbmNetwork
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

def gridsearch(nhidden, lrate, wcost, momentum, n_in_train_minibatch, max_time_secs):
    params = ['nhidden', 'lrate', 'wcost', 'momentum', 'n_in_train_minibatch']
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

    npatterns = None
    pset = create_mnist_patternset(npatterns=npatterns)
    # xxx - this isn't a proper test since we're using the training data, rather than withheld...
    n_in_test_minibatch = 1000
    test_minibatch = Minibatch(pset, n_in_test_minibatch)

    all_t = Stopwatch()
    for attempt_idx, attempt in enumerate(attempts):
        np.random.seed(1)
        t = Stopwatch()
        net = RbmNetwork(np.prod(pset.shape),
                         attempt['nhidden'],
                         attempt['lrate'],
                         attempt['wcost'],
                         attempt['momentum'],
                         v_shape=pset.shape,
                         plot=False)
        epochnum = 0
        while True:
            train_minibatch = Minibatch(pset, nitm)
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
        attempt['dt'] = dt_str()
        attempt['test_error'] = test_error
        attempt['npatterns'] = npatterns

        print 'Finished %i of %i: error %.2f in %.1f secs' % (attempt_idx+1, nattempts, test_error, train_elapsed)
        print attempt
        print '--------------------'
        print
    

    print_best(attempts, params)
    # if all_t.finish(milli=False) > 1000: pause()

    
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--nhidden', type=int, default=[], action='append', help='Number of hidden units')
    parser.add_argument('--lrate', type=float, default=[], action='append', help='Learning rate')
    parser.add_argument('--wcost', type=float, default=[], action='append', help='Weight cost')
    parser.add_argument('--momentum', type=float, default=[], action='append', help='Momentum')
    parser.add_argument('--n_in_train_minibatch', type=int, default=[], action='append', help='Minibatch size (during training)')
    parser.add_argument('-s', '--max_time_secs', type=int, default=100, help='How much time to run each attempt for')
#     parser.add_argument('--shouldplot', type=int, default=0, help='How often to plot graphs - default is 0, i.e. never')
    kwargs = vars(parser.parse_args())
    # i haven't figured out a better way to include defaults
    # in ArgParse (without them still getting included even
    # when you try and override them)
    defaults = {'nhidden': 100,
                'lrate': 0.005,
                'wcost': 0.0002,
                'momentum': 0.9,
                'n_in_train_minibatch': 10,}
    for k,v in defaults.items():
        if kwargs.get(k) == []: kwargs[k].append(v)
    gridsearch(**kwargs)
