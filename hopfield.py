"""hopfield.py -- Hopfield Network

Authors:
  Jessica Hamrick (jhamrick@berkeley.edu)
  Josh Abbott (joshua.abbott@berkeley.edu)

"""

import numpy as np

class hopnet(object):

    def __init__(self, input):
        """Initialize a Hopfield network

        Parameters
        ----------
        input	: an hstack of the input		

        """

        # convert input to -1/+1
        ninput = (input * 2) - 1

        # save parameters
        self.numNeurons,self.numPatterns = ninput.shape

        # create connectivity matrix
        self.T = np.zeros((self.numNeurons, self.numNeurons))
		
        # initialize T using hopfield init rule
        for i in xrange(self.numPatterns):
            self.T += np.outer(ninput[:,i], ninput[:,i])

        self.T /= self.numPatterns


    def read(self, address, iters):
        """Simulates a Hopfield network

        Parameters
        ----------
        address : array or None
            initial input for the network (numNeuronsx1) to test
        iters : int 
            number of iterations to run

        Returns
        -------
        data : array
            the result of running `iters` iterations of the hopfield network 
            defined by the matrix `T`, starting with input `address`
        """

        # convert to -1/+1
        data = ((address * 2) - 1).astype('i4')

        idx = np.random.randint(self.numNeurons, size=iters)
        #data[idx]= 2*(np.dot(self.T[idx,:],data)>0) -
        data[idx]= np.dot(self.T[idx,:],data) > 0

        return data


    def readM(self, addresses, iters):
        M = addresses.shape[1]
        data = np.empty(addresses.shape, dtype='i4')
        for i in xrange(M):
            address = addresses[:, i]
            data[:, i] = self.read(address, iters)
        return data
