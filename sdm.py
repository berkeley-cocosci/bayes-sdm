import numpy as np

class SDM(object):

    def __init__(self, n, m, D, seed=0):
        """Initialize a SDM.

        Parameters
        ----------
        n    : length of inputs/addresses
        m    : number of addresses/storage locations
        D    : hamming radius
        seed : random number generator seed

        """

        # save parameters
        self.n = n
        self.m = m
        self.D = D

        # random number generator
        rso = np.random.RandomState(seed)
        
        # address matrix: matrix of random +/- 1s, where the kth row
        # is the address of the kth storage location
        self.A = (rso.randint(0, 2, (m, n)) * 2) - 1
        # counter matrix: stores contents of the addressed locations
        self.C = np.zeros((n, m))

    def _select(self, address):
        """Select addresses with the Hamming radius(self.D) of the
        given address

        Parameters
        ----------
        address : vector of size n

        Returns
        -------
        theta : binary vector of size m

        """
        x = np.dot(self.A, address)
        theta = ((0.5*(self.n-x)) <= self.D).astype('f8')
        return theta

    def read(self, address):
        """Read the data at the location indicated by the given
        address.

        Parameters
        ----------
        address : vector of size n

        Returns
        -------
        data : vector of size n

        """
        s = self._select(address)
        h = np.dot(self.C, s)
        data = (h > 0).astype('f8') - (h < 0).astype('f8')
        return data

    def write(self, address, data):
        """Write the given data at the location indicated by the given
        address.

        Parameters
        ----------
        address : vector of size n
        data    : vector of size n

        """
        s = self._select(address)
        self.C += np.dot(data[:, None], s[None, :])

    def writeM(self, addresses, data):
        """Write M vectors at data at the locations indicated by the
        given M addresses.

        Parameters
        ----------
        addresses : matrix of size (n, M)
        data      : matrix of size (n, M)

        """
        s = self._select(addresses)
        c = np.sum(data[:, None] * s[None, :], axis=2)
        self.C += c

    def clear(self, address):
        data = -self.read(address)
        s = self._select(address)
        self.C += np.dot(data[:, None], s[None, :])

    def clearM(self, addresses):
        data = -self.read(addresses)
        s = self._select(addresses)
        c = np.sum(data[:, None] * s[None, :], axis=2)
        self.C += c
