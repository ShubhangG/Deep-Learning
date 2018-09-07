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
		
	def backprop(self,eta,error):
		if self.ishead:
			backgradW = max(0,self.prev)
		else:
			backgradW = max(0,self.prev.state)
		
		backgradS = max(0,self.magnitude)
		self.magnitude = self.magnitude - eta*(error)*backgradW
		#if isinstance(self.prev,neuron):
		if not self.ishead:
			self.prev.backprop(eta, error*backgradS)
			#self.prev.update_neuron()
		return



class neuron(object):
	"""The individual neuron responsible for the learning"""
	def __init__(self, inp):
		super(neuron, self).__init__()
		self.bias = np.random.uniform(0,1)
		#self.inputs = inp 
		#self.state = np.array(random.uniform(0,1))
		#self.output = np.zeros(self.state.shape)
		#if isinstance(inp,neuron):
		self.weights = np.empty(inp.shape,dtype=object)
		for i in range(inp.size):
			self.weights[i] = weight(inp[i])
		self.update_neuron()
	
	def update_neuron(self):
		#stat = np.sum(inp*self.weights)+self.bias
		wTx = 0
		for i in range(self.weights.size):
			if isinstance(self.weights[i].prev,neuron):
				wTx+= self.weights[i].magnitude*self.weights[i].prev.state
			else:
				wTx+= self.weights[i].magnitude*self.weights[i].prev
		wTx+= self.bias
		self.state = max(0, wTx)
		return self.state
	
	def backprop(self,eta,error):
		for i in range(self.weights.size):
			self.weights[i].backprop(eta,error)
		self.bias -= eta*error
		return



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

def setup_network(input_vector, hiddenU=30, outputU=10):

	Hlayer = np.empty(hiddenU, dtype=object)
	Olayer = np.empty(outputU,dtype=object)

	#network = np.vectorize(neuron)
	#Hlayer[:] = network(input_vector)
	for i in range(hiddenU):
	 	Hlayer[i] = neuron(input_vector)

	for i in range(outputU):
		Olayer[i] = neuron(Hlayer)


	return Hlayer, Olayer
	
def frontprop(Hlayer,Olayer):
	for neur in Hlayer:
		neur.update_neuron()
	for neu in Olayer:
		neu.update_neuron()
	return

def backprop(Hlayer,Olayer):
	return

def objective_function(Olayer):
	#SOFTMAX
	U = np.zeros(Olayer.size)
	Z = sum(np.exp(outpt.state) for outpt in Olayer)
	for i in range(Olayer.size):
		U[i] = np.exp(Olayer[i].state)/Z

	return U

def train_net(df):

	y_tr = df['Y_train']
	del df['Y_train']
	step_size = 0.01
	#SETUP NETWORK
	rndm_loc = np.random.randint(0,len(df)) 
	inp_vec = df.iloc[rndm_loc]
	Hlayer, Olayer = setup_network(inp_vec)

	#FRONT PROPAGATE
	frontprop(Hlayer,Olayer)
	
	#TAKE OUTPUT AND PASS THROUGH ACTIVATION
	Uprob = objective_function(Olayer)

	net_outpt = max(Uprob)
	max_idx = np.argmax(Uprob)

	#CALCULATE ERROR
	true_idx = y_tr[rndm_loc]
	#E = -np.log(net_outpt)

	print("Olayer_states", [net.state for net in Olayer])
	print("True Num {}".format(true_idx))
	print("Max index {} with prob {}".format(max_idx,net_outpt))

	#Run BackProp
	for i in range(len(Olayer)):
		if i!=true_idx:
			Olayer[i].backprop(step_size,-Uprob[i])
		else:
			Olayer[i].backprop(step_size,1-Uprob[i])

	frontprop(Hlayer,Olayer)

	return Olayer



def main():
	df_test,df_train = read_in_data()
	#y_tr = df_train['Y_train']
	# del df_train['Y_train']
	# print(max(df_train.values.flatten()))
	# assert 1==0
	Ol = train_net(df_train)
	print([nu.state for nu in Ol])



if __name__ == '__main__':
	main()


#############
def testing():
	x = neuron(np.array([0,1,0]))
	print(x.weights)
	print(x.state)
	#print(x.bias)
	x.update_neuron(np.array([0,1,0]))
	print(x.state)