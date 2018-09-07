import numpy as np 
import pandas as pd 
import h5py
#import random

class weight(object):
	"""The weights vector connecting neurons. It has a magnitude and direction"""
	def __init__(self, inp_node=None):
		super(weight, self).__init__()
		self.magnitude = np.random.uniform(0,1)
		self.prev = inp_node
		if isinstance(inp_node,neuron):
			self.ishead = False
		else:
			self.ishead = True

		#self.error = 0
		
	def backprop(self,eta,error):
		if self.ishead:
			backgradW = max(0,self.prev)
		else:
			backgradW = max(0,self.prev.state)

		backgradS = max(0,self.magnitude)
		self.magnitude = self.magnitude - eta*(error)*backgradW
		#if isinstance(self.prev,neuron):
		if not self.ishead:
			self.prev.backprop(error*backgradS)
			self.prev.update_neuron()
		
		return



class neuron(object):
	"""The individual neuron responsible for the learning"""
	def __init__(self, inp):
		super(neuron, self).__init__()
		self.bias = np.random.uniform(0,1)

		#self.inputs = inp 
		#self.state = np.array(random.uniform(0,1))
		#self.output = np.zeros(self.state.shape)
		self.weights = np.random.uniform(0,1,inp.shape)
		for i in range(inp.shape):
			self.weights[i] = weight(inp[i])

		self.update_neuron()

	def update_neuron(self):
		#stat = np.sum(inp*self.weights)+self.bias
		wTx = 0
		for i in range(self.weights.shape):
			if isinstance(self.weights[i].prev,neuron):
				wTx+= self.weights[i].magnitude*self.weights[i].prev.state
			else:
				wTx+= self.weights[i].magnitude*self.weights[i].prev
		
		wTx+= self.bias
		self.state = max(0, wTx)
		return self.state
	
	def backprop(self,error):



############################

def read_in_data():
	"""
	This will read in MNIST, normalize and spit out vector ready to be put into
	neural net

	"""
	raw = h5py.File('MNISTdata.hdf5', 'r')
	keys = list(raw.keys())

	df_test = pd.DataFrame(np.array(raw[keys[0]]))
	df_train = pd.DataFrame(np.array(raw[keys[1]]))
	
	df_test['Y_test'] = np.array(raw[keys[2]][:,0])
	df_train['Y_train'] = np.array(raw[keys[3]][:,0])

	return (df_test,df_train)

def setup_network(input_df, hiddenU=30, outputU=10):
	vector_length= len(input_df['X_train'][0])
	Hlayer = []
	Olayer = []
	for i in range(hiddenU):
		Nhid = neuron(vector_length)
		Hlayer.append(Nhid)

	for i in range(outputU):
		Nout = neuron(hiddenU)
		Olayer.append(Nout)

	return Hlayer, Olayer
	
def frontprop(inpX,Hlayer,Olayer):
	for neur in Hlayer:
		neur.update_neuron(inpX)
	return

def backprop(Hlayer,Olayer):
	
	return

def main():
	df_test,df_train = read_in_data()
	x = neuron(np.array([0,1,0]))
	print(x.weights)
	print(x.state)
	#print(x.bias)
	x.update_neuron(np.array([0,1,0]))
	print(x.state)


if __name__ == '__main__':
	main()