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
    if isinstance(value, float):
        tempList.append(value)

  # setando a lista no lugar do dicionario
  base_de_dados[i] = tempList
  i = i + 1
  
# pegando indice do meio da base de dados
middle_index = len(base_de_dados)//2

# pegando primeira metade para base de treinamento
base_de_treinamento = base_de_dados[:middle_index]

# pegando segunda metade para base de teste
base_de_testes = base_de_dados[middle_index:]


# base de treinamento menor
base_de_treinamento_menor = base_de_treinamento[0:50]

# base de teste menor
base_de_teste_menor = base_de_testes[0:50]

# Começo da LSTM

# convert sequences into required data format
n_unique = len(base_de_treinamento_menor)

# percorrerendo base de treinamento

# setando i para 0 para usa-lo como index novamente
i = 0

# criando array para armazenas seqX e seqY de cada posição da base de treinamento

seqs_base_treinamento = []

for x in base_de_treinamento_menor:
  print(x)
  seqX, seqY = to_lstm_dataset(x, n_unique)
  seqs_base_treinamento[i][0] = seqX
  seqs_base_treinamento[i][1] = seqY
  i = i + 1