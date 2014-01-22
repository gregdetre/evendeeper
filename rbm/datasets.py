import csv, numpy
from pyparsing import delimitedList


class Dataset(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_minst(test=True):
    """
    Reads in the MNIST dataset and returns a Dataset object which holds the data 
    along with metadata from the files specified
    """
    filename = 'minst_test.csv' if test else 'minst_train.csv'
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

    data = numpy.reshape(numpy.array(data,dtype=int), newshape=(rows,cols))

    return Dataset(X = data, name = 'minst_test', num_obs = rows, inputs = cols)
