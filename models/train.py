from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras.backend as K
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,  Flatten,Embedding, LSTM, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
import pickle
import numpy as np

test = pickle.load(open('../data/X_transfer.pkl', 'rb'))
X_transfer_vecs = pickle.load(open('../data/X_transfer_vecs.pkl', 'rb'))
X_transfer = test[len(test)/3:]
X_test = test[0:len(test)/3]

corpus = pickle.load(open('../data/X_main_vecs.pkl', 'rb'))
X_main_vecs = pickle.load(open('../data/X_main.pkl', 'rb'))
# Train subset size (0 < size < len(tokenized_corpus))
train_size = int(corpus.shape[0] * 0.9)

# Test subset size (0 < size < len(tokenized_corpus) - train_size)
test_size = int(X_test.shape[0])

# Compute average and max tweet length
avg_length = 0.0
# Tweet max length (number of tokens)
max_tweet_length = 100

#
print("tranining...")
# Keras convolutional model
batch_size = 32
nb_epochs = 100

model = Sequential()

model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length, vector_size)))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=2, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=nb_epochs,
          validation_data=(X_test, Y_test),
callbacks=[EarlyStopping(min_delta=0.00025, patience=2)])
