# from matplotlib import cm, pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image


def sumsq(x): return sum(x**2)

def imagesc(data, grayscale=True):
    plt.ion()
    # NEAREST appears to do no interpolation too. not sure how they differ
    # plt.imshow(data, cmap="Greys_r" if grayscale else None, interpolation='nearest')
    fig = plt.figure(figsize=(7,4), num=1)
    plt.matshow(data, cmap=plt.cm.gray if grayscale else None, fignum=1)
    plt.show()
    return plt

def tiff(data, filen):
    image = PIL.Image.fromarray(data)
    image.save(filen + '.tiff')
    return image

def downsample(myarr,factor,estimator=np.mean):
    "see https://code.google.com/p/agpy/source/browse/trunk/AG_image_tools/downsample.py"
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor] 
        for i in range(factor)] 
        for j in range(factor)]), axis=0)
    return dsarr

