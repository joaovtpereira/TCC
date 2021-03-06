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

# abrindo arquivo contendo os dados da moeda
f = open('data.json')
 
# guardando dados na variavel data
data = json.load(f)

# guardando atributo de dados na variavel base_de_dados
base_de_dados = data['Data'][:]


i = 0

while i < len(base_de_dados):
  tempList = []
  # ordenando as keys do dicinario em ordem alfabetica
  base_de_dados[i] = dict( sorted(base_de_dados[i].items(), key=lambda x: x[0].lower()) )
  
  for key, value in base_de_dados[i].items():
    # pegando somente valor da key
    # removendo colunas que não sao float
    if isinstance(value, (float, int)):
        tempList.append(value)

  # setando a lista no lugar do dicionario
  base_de_dados[i] = tempList
  i = i + 1
  

i = 0

w, h = 5,int(len(base_de_dados)/24)
vetorCaracteristica = [[0 for x in range(w)] for y in range(h)] 
y = 0


while i < len(base_de_dados) - 24:
    j = 0
    while j < 24:
        if(j == 0):
            vetorCaracteristica[y][0]=base_de_dados[i+j][3]
            vetorCaracteristica[y][3]=base_de_dados[i+j][2]
            vetorCaracteristica[y][4]=base_de_dados[i+j][1]
            
        if(j > 0):
            if(vetorCaracteristica[y][3] > base_de_dados[i+j][2]):
                 vetorCaracteristica[y][3]=base_de_dados[i+j][2]
                 
            if(vetorCaracteristica[y][4] > base_de_dados[i+j][1]):
                 vetorCaracteristica[y][4]=base_de_dados[i+j][1]
            
        if(j == 23):
           vetorCaracteristica[y][1]=base_de_dados[i+j][0]
           vetorCaracteristica[y][2]=base_de_dados[i+j][6]
           
        j = j+1
    
    y = y+1
    i = i+j
