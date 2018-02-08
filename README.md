## Character Level Language Model using an RNN/ GRU

#### An implementation of a Character Level Language Model in Python using Tensorflow. 

#### The architecture includes an input layer, hidden layer and output layer.

#### The input is encoded using 1 hot encoding and is passed onto a hidden layer with 100 neurons. The output layer is a probability distribution over the vocabulary and an output character is sampled from this distribution. Backpropagation is carried out every 25 timesteps.

#### Running the code:

#### (Input is stored in input.txt)

#### >> python rnntf.py

#### Output is a 200 character long sample of text after every epoch.
