'''
	question: what is the use of batch size?
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle



f=open('input.txt','r')
inp=f.read()
inp2=list(inp)
vocab=list(set(inp))
inp_sz=len(vocab)

char_ind = {ch:i for i,ch in enumerate(vocab)}
ind_char = {i:ch for i,ch in enumerate(vocab)}
int_inp = [char_ind[i] for i in inp2]


# hyperparameters

num_epochs = 10000
batch_size = 1
truncated_backprop_length = 25
num_batches = len(inp)//batch_size//truncated_backprop_length
state_size = 100
#train = tf.placeholder(tf.bool)


# get one-hot encoding

def one_hot(ind,sz):

	xenc = np.zeros((len(ind),sz))
	c=0
	for i in ind:
		xenc[c][i]=1
		c+=1
        
	return xenc


# get input

def enc_input(inp2):
	
	inp_enc = [one_hot(c) for c in inp2]


#cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0, state_is_tuple=False)
cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
W2 = tf.Variable(np.random.rand(state_size, inp_sz),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,inp_sz)), dtype=tf.float32)
	
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])


batchX = tf.one_hot(batchX_placeholder, inp_sz)
#batchY = tf.one_hot(batchY_placeholder, inp_sz)


#inputs_series = tf.split(batchX, truncated_backprop_length, 1)
#inputs_series = [tf.squeeze(i,1) for i in inputs_series]
inputs_series = tf.unstack(batchX, axis=1)

print(tf.shape(inputs_series))
labels_series = tf.unstack(batchY_placeholder, axis=1)


states_series, current_state = tf.nn.static_rnn(cell, inputs_series, init_state)



logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


# sample character from predictions
	
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(1e-1).minimize(total_loss)


# sample char according to probabilities of output vector

def get_char(inp):
	
	op_ind = int(np.random.choice(range(inp_sz), p=inp))
	
	return op_ind
	

# sample a paragraph of 200 characters

def sample(curr, ip):

	#c=get_char(tf.nn.softmax(ip[0]))
	ipt, curr = cell(ip,curr)
	#print(tf.shape(ipt))
	ip = tf.matmul(ipt,W2) + b2
			
	return tf.nn.softmax(ip), curr


def softmax(inp):
	return (np.exp(inp) / np.sum(np.exp(inp)))



curr_state = tf.placeholder(tf.float32, [batch_size, state_size])
ip = tf.placeholder(tf.float32, [batch_size, inp_sz])	
op, fin_state = sample(curr_state, ip)



# run session

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    print("numbatches : ",num_batches)
    data_len = batch_size*truncated_backprop_length*num_batches
    print("total : ",data_len)
    _current_state = np.zeros((batch_size, state_size))


    for epoch_idx in range(num_epochs):
	
		
	x=np.array(int_inp[:data_len])	
	y=np.array(int_inp[1:data_len+1])	
	x = x.reshape((batch_size, -1))  
	y = y.reshape((batch_size, -1))

	#print(b2.eval())

	print("New data, epoch", epoch_idx)

	for batch_idx in range(num_batches):
	    start_idx = batch_idx * truncated_backprop_length
	    end_idx = start_idx + truncated_backprop_length

	    batchX = x[:,start_idx:end_idx]
	    batchY = y[:,start_idx:end_idx]
	    #print(batchX)
	    #print(batchY)

	    _total_loss, _train_step,_current_state, _predictions_series = sess.run(
		[total_loss, train_step, current_state, predictions_series],
		feed_dict={
		    batchX_placeholder:batchX,
		    batchY_placeholder:batchY,
		    init_state:_current_state
		})

	    loss_list.append(_total_loss)

	    if batch_idx%100 == 0:
		print("Step",batch_idx, "Loss", _total_loss)
		print("")
		print("")


	#generate sample of length 200

	# get random first char

	ip_1 = np.random.randint(0,inp_sz)
	#ip_2 = np.random.randint(0,inp_sz)
	_ip = one_hot([ip_1],inp_sz)
	#print(_ip.shape)
	string=""
        _current_state2 = np.zeros((batch_size, state_size))
	#_current_state2 = np.copy(_current_state)
	for i in xrange(200):

		#print(_current_state2)
		#print(_current_state.shape)
		#print("_ip")
		#print(_ip[0])
		ind = get_char(_ip[0])
		c = ind_char[ind]
		string+=c
		_ip = one_hot([ind],inp_sz)
		_ip, _current_state2 = sess.run([op, fin_state], feed_dict={curr_state:_current_state2, ip: _ip})

	print(string)
	print("")
	print("")



	#print(type(_current_state))
	#print(_current_state[0])
	#print(current_state[0])
	#print(tf.constant(_current_state[0]))
	#string=sample(tf.constant(_current_state))
	#print(string)

#pickle.dump(cell,open("cell.p","wb"))	
#pickle.dump(_current_state,open("curr.p","wb"))	
