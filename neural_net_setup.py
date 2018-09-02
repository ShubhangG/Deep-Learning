import numpy as np 
import pandas as pd 
import h5py
import random



class neuron(object):
	"""The individual neuron responsible for the learning"""
	def __init__(self, x):
		super(neuron, self).__init__()
		self.bias = random.uniform(0,1)
		self.state = x
		self.output = np.zeros(self.state.shape)
		



		


############################

def read_in_data():
	"""
	This will read in MNIST, normalize and spit out vector ready to be put into
	neural net

	"""
	raw = h5py.File('MNISTdata.hdf5', 'r')
	keys = list(raw.keys())

	df_test = pd.DataFrame(np.array(raw[keys[0]]),columns=['X_test'])
	df_train = pd.DataFrame(np.array(raw[keys[1]]),columns=['X_train'])
	
	df_test['Y_test'] = np.array(raw[keys[2]])
	df_train['Y_train'] = np.array(raw[keys[3]])


	return (df_test,df_train)

def main():
	df_test,df_train = read_in_data()
	


if __name__ == '__main__':
	main()