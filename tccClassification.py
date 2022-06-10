import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import numpy as np


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
targets = [0 for y in range(h)] 

while i < len(base_de_dados)  - 24:
    j = 0
    value_initial = 0

    while j < 24:
        if(j == 0):
            vetorCaracteristica[y][0]=base_de_dados[i+j][3]
            vetorCaracteristica[y][3]=base_de_dados[i+j][2]
            vetorCaracteristica[y][4]=base_de_dados[i+j][1]
            value_initial = base_de_dados[i+j][0]
            
        if(j > 0):
            if(vetorCaracteristica[y][3] > base_de_dados[i+j][2]):
                 vetorCaracteristica[y][3]=base_de_dados[i+j][2]
                 
            if(vetorCaracteristica[y][4] > base_de_dados[i+j][1]):
                 vetorCaracteristica[y][4]=base_de_dados[i+j][1]
            
        if(j == 23):
           vetorCaracteristica[y][1]=base_de_dados[i+j][0]
           vetorCaracteristica[y][2]=base_de_dados[i+j][6]
           if(value_initial < base_de_dados[i+j][0]):
              targets[y] = 0
           else :
              targets[y] = 1
           
        j = j+1
    
    y = y+1
    i = i+j


# rede

X_train, X_test, y_train, y_test = train_test_split(np.array(vetorCaracteristica), np.array(targets), test_size=0.33)
N, D = X_train.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print("Train score: ", model.evaluate(X_train, y_train))
print("Test score: ", model.evaluate(X_test, y_test))

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
