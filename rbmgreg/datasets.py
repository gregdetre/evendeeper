import csv, numpy


class Dataset(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_mnist(test=True, nrows=None):
    """
    Reads in the MNIST dataset and returns a Dataset object which holds the data 
    along with metadata from the files specified
    """
    filename = 'mnist_test.csv' if test else 'mnist_train.csv'
    data = []
    cols = 0
    rows = 0
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, dialect='excel', delimiter=',');
        for line in csvreader:
            data.extend(line)
            if cols == 0:
                cols = len(line)
            rows += 1
            if nrows is not None and rows >= nrows: break

    data = numpy.reshape(numpy.array(data,dtype=int), newshape=(rows,cols))

    return Dataset(X = data, name = 'mnist_test', num_obs = rows, inputs = cols)
