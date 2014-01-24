from matplotlib import pyplot as plt
import numpy as np


def imagesc(data, dest=None, grayscale=True, vmin=None, vmax=None):
    plt.ion()
    cmap = plt.cm.gray if grayscale else None
    if dest is None:
        fig = plt.figure(figsize=(7,4))
        plt.matshow(data, cmap=cmap, fignum=fig.number, vmin=vmin, vmax=vmax)
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
    else:
        show = dest.matshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        dest.axes.get_xaxis().set_visible(False)
        dest.axes.get_yaxis().set_visible(False)
    plt.show()
    return show

def isunique(lst): return len(set(lst))==1

def sigmoid(x):
    # from peter's rbm
    return 1.0 / (1.0 + np.exp(-x))

def sumsq(x): return sum(x**2)

def vec_to_arr(x): return x.reshape(1, len(x))
