import multiprocessing
from math import ceil
from time import sleep
from Queue import Queue
import numpy as np
from ipdb import set_trace as pause
from utils.stopwatch import Stopwatch
import os

def multiply_matrices(a):
    """ 
    Some function that does a lot of cpu processing
    """
    return np.dot(a, a)

def execute_parallel(args, nprocs):
    def worker(proc_args, out_q, i_proc):
        """ 
        The worker function: this function is assigned to different processes
        """
        print "Process %i has been assigned %i subtask(s)" % (os.getpid(), len(proc_args))
        for arg_idx, arg in enumerate(proc_args):
            t = Stopwatch()
            out_q.put(multiply_matrices(arg))
            elapsed = t.finish(milli=False)
            print "Process %i needed %d seconds for task %i/%i" % (os.getpid(), elapsed, arg_idx+1, len(proc_args))

    print "Creating queue for %i tasks..." % len(args)
    # Each process will get 'chunksize' nums and a queue to put his results into
    m = multiprocessing.Manager()
    out_q = m.Queue()
    chunksize = int(ceil(len(args) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        p = multiprocessing.Process(
                target=worker, args=(args[chunksize * i:chunksize * (i + 1)], out_q, i))
        procs.append(p)
        p.start()
        print "Starting process with pid %d" % p.pid

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    for args in args:     
        out_q.get()
        print "Collected result (Queue size: %d)" % (out_q.qsize())


if __name__ == '__main__':

    # numpy import mess with core affinity on import (http://bugs.python.org/issue17038)
    # reset task affinity otherwise code will only run on one core
    os.system("taskset -p 0xff %d" % os.getpid())

    num_cores = 2
    task_list = [np.random.randn(1000, 1000), np.random.randn(1000, 1000), np.random.randn(1000, 1000), np.random.randn(1000, 1000)]

    execute_parallel(task_list, num_cores)

