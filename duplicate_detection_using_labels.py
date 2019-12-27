
# Authors:
# Konstantinos Nikopoulos
# Ioannis Maliouris
# Source:
# Abhishek Thakur (https://github.com/abhishekkrthakur)

# Changes from original version (https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/deepnet.py):
# Significal smaller model and use a subset of dataset for training in CPU, text processing without glove for space efficiency. 


import keras
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence, text
from keras.utils import np_utils
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing import sequence, text

######################################## Preproccess dataset #########################################

data = pd.read_csv("train_original.csv")
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

data = data.fillna('null')

y = data.is_duplicate.values

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
# Reshape dataset and keep 100.000 samples
x_train= x_train[0:80000]
y_train= y_train[0:80000]
x_test= x_test[0:20000]
y_test= y_test[0:20000]
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)

# An index with all words of x_train
tk_train = text.Tokenizer(num_words=80000)

max_len = 40

tk_train.fit_on_texts(list(x_train.question1.values) + list(x_train.question2.values.astype(str)))
x1_train = tk_train.texts_to_sequences(x_train.question1.values)
x1_train = sequence.pad_sequences(x1_train, maxlen=max_len)
x2_train = tk_train.texts_to_sequences(x_train.question2.values.astype(str))
x2_train = sequence.pad_sequences(x2_train, maxlen=max_len)

tk_train.fit_on_texts(list(x_test.question1.values) + list(x_test.question2.values.astype(str)))
x1_test = tk_train.texts_to_sequences(x_test.question1.values)
x1_test = sequence.pad_sequences(x1_test, maxlen=max_len)
x2_test = tk_train.texts_to_sequences(x_test.question2.values.astype(str))
x2_test = sequence.pad_sequences(x2_test, maxlen=max_len)

word_index = tk_train.word_index

# Labels to one-hot 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

######################################### Neural Network ##############################################

inputs1 = Input(shape=(40,))
model1 = Embedding(len(word_index) + 1, 300, input_length=40)(inputs1)
model1 = Dropout(0.2)(model1)
model1 = LSTM(100, recurrent_dropout=0.2, dropout=0.2)(model1)

inputs2 = Input(shape=(40,))
model2 = Embedding(len(word_index) + 1, 300, input_length=40)(inputs2)
model2 = Dropout(0.2)(model2)
model2 = LSTM(100, recurrent_dropout=0.2, dropout=0.2)(model2)

merged = concatenate([model1, model2])

merged = BatchNormalization()(merged)

merged = Dense(100, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

merged = Dense(100, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)

merged = Dense(2, activation='softmax')(merged)

model = Model(inputs=[inputs1,inputs2], outputs=merged)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

results = model.fit([x1_train, x2_train], y=y_train, batch_size=256, epochs=50,
                 verbose=1, validation_split=0.2, shuffle=True, callbacks=[checkpoint])
                 
########################################## Results ##############################################
    
print("Train accuracy: ", model.evaluate([x1_train, x2_train], y_train, batch_size=256))
print("Test accuracy: ", model.evaluate([x1_test, x2_test], y_test, batch_size=256))






