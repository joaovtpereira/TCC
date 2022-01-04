from pandas import DataFrame
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


# binary encode an input pattern, return a list of binary vectors
def encode(pattern, n_unique):
	encoded = list()
	for value in pattern:
		row = [0.0 for x in range(n_unique)]
		row[value] = 1.0
		encoded.append(row)
	return encoded

# create input/output pairs of encoded vectors, returns X, y
def to_xy_pairs(encoded):
	X,y = list(),list()
	for i in range(1, len(encoded)):
		X.append(encoded[i-1])
		y.append(encoded[i])
	return X, y

# convert sequence to x/y pairs ready for use with an LSTM
def to_lstm_dataset(sequence, n_unique):
	# one hot encode
	encoded = encode(sequence, n_unique)
	# convert to in/out patterns
	X,y = to_xy_pairs(encoded)
	# convert to LSTM friendly format
	dfX, dfy = DataFrame(X), DataFrame(y)
	lstmX = dfX.values
	lstmX = lstmX.reshape(lstmX.shape[0], 1, lstmX.shape[1])
	lstmY = dfy.values
	return lstmX, lstmY

# define sequences
seq1 = [3, 0, 1, 2, 3]
seq2 = [4, 0, 1, 2, 4]
seq3 = [6, 2, 3, 4, 6]
seq4 = [7, 3, 4, 5, 7]
seq5 = [8, 4, 5, 6, 8]
seq6 = [5, 1, 2, 3, 5]
seq8 = [10, 6, 7, 8, 10]
seq9 = [11, 7, 8, 9, 11]
seq10 = [12, 8, 9, 10, 12]
seq11 = [13, 9, 10, 11, 13]
seq7 = [9, 5, 6, 7, 9]
# convert sequences into required data format
n_unique = len(set(seq1 + seq2 + seq3 + seq4 + seq5 + seq6 + seq8+ seq9 + seq10 +seq11+ seq7))

seq1X, seq1Y = to_lstm_dataset(seq1, n_unique)
seq2X, seq2Y = to_lstm_dataset(seq2, n_unique)
seq3X, seq3Y = to_lstm_dataset(seq3, n_unique)
seq4X, seq4Y = to_lstm_dataset(seq4, n_unique)
seq5X, seq5Y = to_lstm_dataset(seq5, n_unique)
seq6X, seq6Y = to_lstm_dataset(seq6, n_unique)
seq7X, seq7Y = to_lstm_dataset(seq7, n_unique)
seq8X, seq8Y = to_lstm_dataset(seq8, n_unique)
seq9X, seq9Y = to_lstm_dataset(seq9, n_unique)
seq10X, seq10Y = to_lstm_dataset(seq10, n_unique)
seq11X, seq11Y = to_lstm_dataset(seq11, n_unique)

# define LSTM configuration
n_neurons = 20
n_batch = 1
n_epoch = 10
n_features = n_unique


# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_features), stateful=True))
model.add(Dense(n_unique, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# train LSTM
for i in range(n_epoch):
	model.fit(seq1X, seq1Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq2X, seq2Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()
    
# train LSTM
for i in range(n_epoch):
	model.fit(seq3X, seq3Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq4X, seq4Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()
    
# train LSTM
for i in range(n_epoch):
	model.fit(seq5X, seq5Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq6X, seq6Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()
    
# train LSTM
for i in range(n_epoch):
	model.fit(seq8X, seq8Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq9X, seq9Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()
    
# train LSTM
for i in range(n_epoch):
	model.fit(seq10X, seq10Y, epochs=1, batch_size=n_batch, verbose=1, shuffle=False)
	model.reset_states()
	model.fit(seq11X, seq11Y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
	model.reset_states()

# test LSTM on sequence3
print('Sequence 7')
result = model.predict_classes(seq7X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq7[i], seq7[i+1], result[i]))
    
"""

# test LSTM on sequence 1
print('Sequence 1')
result = model.predict_classes(seq1X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq1[i], seq1[i+1], result[i]))

# test LSTM on sequence 2
print('Sequence 2')
result = model.predict_classes(seq2X, batch_size=n_batch, verbose=0)
model.reset_states()
for i in range(len(result)):
	print('X=%.1f y=%.1f, yhat=%.1f' % (seq2[i], seq2[i+1], result[i]))

"""