import json
import pandas as pd
from collections import deque
import numpy as np
import random
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#tensorflow imports

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


import matplotlib.pyplot as plt

EPOCHS = 10
BATCH_SIZE = 64
SEQ_LEN= 1
TEST =  "LSTM-0.20"
NAME = f"{SEQ_LEN}-SEQ-{TEST}-PRED-{int(time.time())}"

# abrindo arquivo contendo os dados da moeda
f = open('data3.json')
# guardando dados na variavel data
data = json.load(f)

base_de_dados = data['Data'][:]
i = 0

nome_das_classes = ['compra', 'venda']

# função para criar o target baseado no preço presente e futuro
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

# função para realizar pré processamento da base
def preprocess_df(df):
    for col in df.columns:
        if col != "target": # não precisamos normalizar o target
            df[col] = df[col].pct_change() # uma forma de normalizar tendo como base o movimento dos valores para cada característica
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) # nornaliza os dados entre -1 e 1
           
            
    df.dropna(inplace=True)    
    
    sequential_data = []
    prev_days = deque(maxlen=(SEQ_LEN))    
    
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # pegando todos as colunas menos o target
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
        
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:

        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)
    
    
    lower = min(len(buys), len(sells)) # pegando qual classe tem menos dados
    
    buys = buys[:lower] # nivelando os targets
    sells = sells[:lower] # nivelando os targets
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = [] ## entrada da rede
    Y = [] ## targets da rede
    
    for seq, target in sequential_data:
        X.append(seq)
        Y.append(target)
        
    return np.array(X), Y
while i < len(base_de_dados):
  tempList = []
  # ordenando as keys do dicinario em ordem alfabetica
  base_de_dados[i] = dict( sorted(base_de_dados[i].items(), key=lambda x: x[0].lower()) )
  
  for key, value in base_de_dados[i].items():
    # pegando somente valor da key
    # removendo colunas que não sao float
    # if isinstance(value, (float, int)) and ( key != 'time'):
    if isinstance(value, (float, int)) and ( key != 'time' and key != 'volumeto'):
       #  if(key == 'volumeto' or key == 'volumefrom'):
         #    value = scaler.fit_transform(value)
        tempList.append(value)

  # setando a lista no lugar do dicionario
  base_de_dados[i] = tempList
  i = i + 1
  
  
#pegando target
targets = [0 for y in range(1968)] 
i=0
  
while i < len(base_de_dados)  - 24:
    j = 0
    value_initial = 0
    
    while j < 24:
        if(value_initial == 0):
            # if(j == 0) :
                # print('valor: ', base_de_dados[i + j][3])
            value_initial = base_de_dados[i + j][3]
        
        if(j == 23):
            # if(j == 0) :
                # print('valor close: ', base_de_dados[i + j][0])
                # print('valor abertura: ', value_initial)
            if(base_de_dados[i + j][0] >= value_initial):
                for x in range(i, i+24):
                    targets[x] = 1
            else :
                for x in range(i, i+24):
                    targets[x] = 0
                
        j = j+1
        
    i = i+24
  
main_df = pd.DataFrame(base_de_dados)
main_df.columns = ['close', 'high', 'low', 'open', 'volumefrom']
main_df['target'] = targets

times = sorted(main_df.index.values)
last_5pct = times[-int(0.30*len(times))]


validation_main_df = main_df[(main_df.index >= last_5pct)] # aqui ele ta separando um array com todas as amostras com timestamp maior que a last_5pct, ou seja, pegando os 5% restante da base
main_df = main_df[(main_df.index < last_5pct)] # aqui ele ta pegando os 95% restante da base, já que separou os 5% para teste (Esse é o treinamento)

# pré processamento dos dados
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)} Buys: {train_y.count(1)}")
print(f"Validation Dont buys: {validation_y.count(0)} validation buys: {validation_y.count(1)}")

"""
model = Sequential()
model.add(Input(shape=(train_x.shape[1:]))) # 128 nós
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Input(shape=(train_x.shape[1:]))) # 128 nós
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Input(shape=(train_x.shape[1:]))) # 128 nós
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))
"""

dimX1, dimX2, dimX3 = np.array(train_x).shape
train_x = np.reshape(np.array(train_x), (dimX1*dimX2, dimX3))

dimX1, dimX2, dimX3 = np.array(validation_x).shape
validation_x = np.reshape(np.array(validation_x), (dimX1*dimX2, dimX3))


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
print(f"Acurácia de treinamento: {knn.score(train_x, train_y)}")


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

y_pred = knn.predict(validation_x) # retorna a classe diretamente
y_pred_prob = knn.predict_proba(validation_x) # retorna a probabilidade de cada classe
acc_teste = knn.score(validation_x, validation_y)
print(f"Acurácia de teste: {acc_teste}")

    
relatorio = classification_report(validation_y, y_pred, target_names=nome_das_classes)
print("Relatório de classificação:")
print(relatorio)

mat_conf = confusion_matrix(validation_y, y_pred)
print("Matriz de confusão:")
print(mat_conf)



