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

	def update_inputs(self,inp):
		self.weights = np.empty(inp.shape,dtype=object)
		for i in range(inp.size):
			self.weights[i] = weight(inp[i])
		self.update_neuron()



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
	mx_st = max(out.state for out in Olayer)
	Z = sum(np.exp(outpt.state-mx_st) for outpt in Olayer)
	for i in range(Olayer.size):
		U[i] = np.exp(Olayer[i].state-mx_st)/Z
		if(U[i] > 1):
			U[i]=1
	return U

def next_inp(inp,Hlayer):
	for node in Hlayer:
		node.update_inputs(inp)

def train_net(df):

	y_tr = df['Y_train']
	del df['Y_train']
	step_size = 0.01

	#SETUP NETWORK
	rndm_loc = np.random.randint(0,len(df)) 
	inp_vec = df.iloc[rndm_loc]
	Hlayer, Olayer = setup_network(inp_vec)

	num_epochs = 1
	for ep in range(num_epochs):
		if (ep > 5):
			step_size  = 0.01
		if (ep > 10):
			step_size  = 0.001
		if (ep > 15):
			step_size  = 0.0001
		num_correct = 0
		for n in range(len(df)):
			n_rnd = np.random.randint(0,len(df))
			x_tr = df.iloc[n_rnd]
			true_idx = y_tr[n_rnd]
			next_inp(x_tr,Hlayer)
			frontprop(Hlayer,Olayer)
			Uprob = objective_function(Olayer)
			net_outpt = max(Uprob)
			max_idx = np.argmax(Uprob)
			if(max_idx == true_idx):
				num_correct+=1
			# print("Olayer_states", [net.state for net in Olayer])
			# print("True Num {}".format(true_idx))
			# print("Max index {} with prob {}".format(max_idx,net_outpt))

			#Run BackProp
			for i in range(len(Olayer)):
				if i!=true_idx:
					Olayer[i].backprop(step_size,-Uprob[i])
				else:
					Olayer[i].backprop(step_size,1-Uprob[i])

			frontprop(Hlayer,Olayer)
		print("For epoch {} the number correct are {}".format(ep,num_correct))


def main():
	df_test,df_train = read_in_data()
	#y_tr = df_train['Y_train']
	# del df_train['Y_train']
	# print(max(df_train.values.flatten()))
	print len(df)
	# import cProfile
	# cProfile.run(train_net(df_train),sort='cumtime')
	train_net(df_train)
	#print([nu.state for nu in Ol])



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