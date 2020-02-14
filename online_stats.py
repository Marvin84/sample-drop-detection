'''
@author: Sunit Sivasankaran, Inria-Nancy
Computes the mean and variance in an online fashion
Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
'''
import numpy as np


class OnlineStatCompute(object):
    '''
        A class to compute online stats
    '''

    def __init__(self, dim):
        self._mean = np.zeros((dim))
        self._std = np.zeros((dim))
        self._var = np.zeros((dim))
        self.feat_count = 0
        self.updated = False

    def update_stats(self, full_data):
        '''
            full_data is assumed to be of shape
                dim x frames
        '''
        for data in full_data.T:
            self.feat_count += 1
            delta = data - self._mean
            self._mean += delta / self.feat_count
            self._var += delta * (data - self._mean)
        self.updated = True

    def get_stats(self):
        return self._mean, np.sqrt(self._var / self.feat_count)

    def get_mean(self):
        return self._mean

    def get_std(self):
        return np.sqrt(self._var / self.feat_count)

    @property
    def std(self):
        return self.get_std()

    @property
    def mean(self):
        return self.get_mean()

    @property
    def var(self):
        return self._var


# if __name__ == '__main__':
#     # Demonstrate usage
#     import ipdb
#
#     ipdb.set_trace()
#     dim = 512
#     data1 = np.random.rand(dim, 1000)
#     data2 = np.random.rand(dim, 1000)
#     data3 = np.random.rand(dim, 1000)
#     data = np.hstack((data1, data2, data3))
#     real_mean = data.mean(1)
#     real_std = data.std(1)
#
#     interface = OnlineStatCompute(dim)
#     interface.update_stats(data1)
#     interface.update_stats(data2)
#     interface.update_stats(data3)
#     print(np.allclose(real_mean, interface.mean, atol=1e-1))