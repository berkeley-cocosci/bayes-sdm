"""hopfield.py -- Hopfield Network

Authors:
  Jessica Hamrick (jhamrick@berkeley.edu)
  Josh Abbott (joshua.abbott@berkeley.edu)

"""

import numpy as np

class hopnet(object):

    def __init__(self,input=None):
        """Initialize a Hopfield network

        Parameters
        ----------
        input	: an hstack of the input		

        """

        # save parameters
        self.numNeurons,self.numPatterns = input.shape

        # create connectivity matrix
        self.T = np.zeros((self.numNeurons,self.numNeurons))
		
        # initialize T using hopfield init rule
        for i in xrange(self.numPatterns):
            self.T += np.outer(input[:,i],input[:,i])

        self.T /= self.numPatterns


    def read(self, iters=None, address=None):
        """Simulates a Hopfield network

        Parameters
        ----------
        iters : int 
            number of iterations to run
        address : array or None
            initial input for the network (numNeuronsx1) to test

        Returns
        -------
        data : array
            the result of running `iters` iterations of the hopfield network 
            defined by the matrix `T`, starting with input `address`
        """

        data = address.copy()

        idx = np.random.randint(self.numNeurons, size=iters)
        data[idx]= 2*(np.dot(self.T[idx,:],data)>0) -1


        return data

