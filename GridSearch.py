import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import itertools
import os
import warnings

#avoid unnecessary warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_json("data.json")

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each text.
MAX_SEQUENCE_LENGTH = 500
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, char_level=False, split=' ')
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

Y = pd.get_dummies(df['label']).values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)


#returns compiled model with given parameters
def makeModel(dropout=0.2, lstmOutputSize=100, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(dropout))
    model.add(LSTM(lstmOutputSize, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#list of all potential hyperparameters
epochsList = [1,2,3,4,5,6]
batchSizes = [64, 32, 16]
optimizers = ['adagrad', 'adam']
dropouts = [0.5, 0.4, 0.2]
lstmOutputSizes = [50, 75, 100, 150]

combinations = [epochsList, batchSizes, optimizers, dropouts, lstmOutputSizes]
accuracyDict = {} #keeps track of accuracies from various combinations of hyperparameters

for epochs,batchSize,optimizer,dropout,lstmOutputSize in itertools.product(*combinations): #make, fit, and evaluate model for each combination of hyperparameters
    print("Running with Epochs: {}; BatchSize: {}; Optimzer: {}; Dropout: {}; LstmOutputSize: {}".format(epochs,batchSize,optimizer,dropout,lstmOutputSize))
    model = makeModel(dropout=dropout, lstmOutputSize=lstmOutputSize, optimizer=optimizer)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batchSize,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)], verbose=0)
    accr = model.evaluate(X_test,Y_test,verbose=0)
    print('\tLoss: {:0.4f}\n \tAccuracy: {:0.4f}\n'.format(accr[0],accr[1]))
    accuracyDict[accr[1]] = (epochs,batchSize,optimizer,dropout,lstmOutputSize)

print("results:", accuracyDict) #output raw results

#output best hyperparameter combination
bestAccuracy = max(accuracyDict)
epochs,batchSize,optimizer,dropout,lstmOutputSize = accuracyDict[bestAccuracy]
print("\nBest Accuracy: {:0.5f}".format(bestAccuracy))
print("From hyperparameters: Epochs: {}; BatchSize: {}; Optimzer: {}; Dropout: {}; LstmOutputSize: {}".format(epochs,batchSize,optimizer,dropout,lstmOutputSize))
