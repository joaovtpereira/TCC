import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

print(np.__version__)
print(tf.__version__)

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
    # if isinstance(value, (float, int)) and ( key != 'time'):
    if isinstance(value, (float, int)) and ( key != 'time' and key != 'volumeto' and key != 'volumefrom'):
        tempList.append(value)

  # setando a lista no lugar do dicionario
  base_de_dados[i] = tempList
  i = i + 1
  

i = 0

targets = [0 for y in range(2001)] 

while i < len(base_de_dados):
    if(base_de_dados[i][3] < base_de_dados[i][0]):
        targets[i] = 1
    else:
        targets[i] = 0
    
    i= i+1

# Model configuration
additional_metrics = ['accuracy']
batch_size = 128
embedding_output_dims = 15
loss_function = BinaryCrossentropy()
max_sequence_length = 5
num_distinct_words = 100000
number_of_epochs = 50
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load dataset
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
x_train, x_test, y_train, y_test = train_test_split(np.array(base_de_dados), np.array(targets), test_size=0.33)
print(x_train.shape)
print(x_test.shape)


# Pad all sequences
padded_inputs = pad_sequences(x_train, maxlen=max_sequence_length, value = 0.0) # 0.0 because it corresponds with <PAD>
padded_inputs_test = pad_sequences(x_test, maxlen=max_sequence_length, value = 0.0) # 0.0 because it corresponds with <PAD>

# Define the Keras model
model = Sequential()
model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

# Give a summary
model.summary()

# Train the model
history = model.fit(padded_inputs, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# Test the model after training
test_results = model.evaluate(padded_inputs_test, y_test, verbose=False)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]}%')

# normalizar volumeTo
# K fouuld
# testar a base no modelo padrao