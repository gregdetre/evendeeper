from ipdb import set_trace as pause
import numpy as np

from base import Network, Patternset
from utils import isunique, sigmoid


class RbmNetwork(Network):
    def act_fn(self, x): return sigmoid(x)

    def update_weights(self, target):
        pause()
        dw = np.empty(self.w.shape)
        for i in range(len(self.v)):
            for j in range(len(self.h)):
                dw[i,j] = self.lrate * self.v[i] * (target[j] - self.h[j])
        self.w += dw
        return dw


def generate_pset(shape=(3,3), npatterns=5):
    iset = [np.random.rand(shape) for _ in range(npatterns)]
    return Patternset(iset)


if __name__ == "__main__":
    net = RbmNetwork()
    pset = generate_pset()
    
