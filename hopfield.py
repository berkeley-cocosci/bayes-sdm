import numpy as np
import matplotlib.pyplot as plt

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

		sz=np.sqrt(self.numNeurons);

		data = address

#		idx = np.random.randint(self.numNeurons, size=iters)
#		data[idx]= 2*(np.dot(self.T[idx,:],data)>0) -1
		

#		h=plt.imshow(data.reshape((sz,sz)).T,vmin=-1, vmax=1, interpolation='nearest');
#		fig, ax = plt.gcf(),plt.gca()
#		plt.draw()

		for i in np.random.randint(self.numNeurons, size=iters): # pick the neuron to flip
			# compute net input and flip state accordingly
			data[i]= 2*(np.dot(self.T[i,:],data)>0) -1

			# refresh display
#			h.set_data(data.reshape((sz,sz)).T)
#			plt.draw()


		return data

