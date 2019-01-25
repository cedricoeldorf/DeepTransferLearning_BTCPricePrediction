import pickle
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Dropout, Reshape
from keras.layers import Dense, Input,  Flatten,Embedding, LSTM, SpatialDropout1D,TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, AveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, rmsprop
from keras.models import load_model
from keras import callbacks
import matplotlib.pyplot as plt
from sklearn import svm
from keras.preprocessing import sequence

try:
    with open('../data/train_vecs_w2v.pkl', 'rb') as fp:
        train_vecs_w2v = pickle.load(fp)
        #train_vecs_w2v = train_vecs_w2v.reshape(train_vecs_w2v.shape[0],1,train_vecs_w2v.shape[1])
    with open('../data/test_vecs_w2v.pkl', 'rb') as fp:
        test_vecs_w2v = pickle.load(fp)
        #test_vecs_w2v = test_vecs_w2v.reshape(test_vecs_w2v.shape[0],1,test_vecs_w2v.shape[1])
    with open('../data/y_train.pkl', 'rb') as fp:
        y_train = pickle.load(fp)
    with open('../data/y_test.pkl', 'rb') as fp:
        y_test = pickle.load(fp)
    with open('../data/TRANSFER_test_vecs_w2v.pkl', 'rb') as fp:
        TRANSFER_X_test = pickle.load(fp)
        #TRANSFER_X_test = TRANSFER_X_test.reshape(TRANSFER_X_test.shape[0],1,TRANSFER_X_test.shape[1])
    with open('../data/TRANSFER_y_test.pkl', 'rb') as fp:
        TRANSFER_y_test = pickle.load(fp)
except (OSError, IOError) as e:
    print("PLEASE RUN data_prep.py")
model_type = input("model? (ffnn/cnn)")
if model_type == 'ffnn':
    load = input("load existing model? (y/n) ")
    if load == 'y':
        model = load_model('../model_saved/current.h5')
    else:

        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim = 300))
        #model.add(TimeDistributed(Dense(32),input_shape=(300,)))
        model.add(Dropout(0.7))

        #model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        #model.add(TimeDistributed(Dense(1), activation='sigmoid')) # output shape: (nb_samples, timesteps, 5)
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        class_weight = {0: 1,
                        1: 5}

        history = model.fit(train_vecs_w2v, y_train, epochs=22, batch_size=512, validation_data = (TRANSFER_X_test, TRANSFER_y_test))
        model.save('../model_saved/current.h5')


    print("_________________")
    print("ORIGINAL SCORE")
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=512)
    print(score[1])
    score = model.evaluate(TRANSFER_X_test, TRANSFER_y_test, batch_size=512)
    print(score[1])
    print("_________________")

    with open('../data/TRANSFER_train_vecs_w2v.pkl', 'rb') as fp:
        TRANSFER_X_train = pickle.load(fp)
    with open('../data/TRANSFER_y_train.pkl', 'rb') as fp:
        TRANSFER_y_train = pickle.load(fp)

    class_weight = {0: 1,
                    1: 1.28}
    history = model.fit(TRANSFER_X_train, TRANSFER_y_train, epochs=30, batch_size=512 ,validation_data = (TRANSFER_X_test, TRANSFER_y_test))

    print("_________________")
    print("TRANSFER SCORE")
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=512)
    print(score[1])
    score = model.evaluate(TRANSFER_X_test, TRANSFER_y_test, batch_size=512)
    print(score[1])
    model.save('../model_saved/current_transfer.h5')
    plt.figure(figsize=(12, 14))
    plt.ion()
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
if model_type == 'cnn':
    load = input("load existing model? (y/n) ")
    if load == 'y':
        model = load_model('../model_saved/cnn.h5')
    else:

        train_vecs_w2v = sequence.pad_sequences(train_vecs_w2v, maxlen=300)
        TRANSFER_X_test = sequence.pad_sequences(TRANSFER_X_test, maxlen=300)
        model = Sequential()
        model.add(Embedding(15000, 300, input_length=300))
        model.add(Convolution1D(64, 3, border_mode='same'))
        model.add(Convolution1D(32, 3, border_mode='same'))
        model.add(Convolution1D(16, 3, border_mode='same'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(180,activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        history = model.fit(train_vecs_w2v, y_train, epochs=22, batch_size=1024, validation_data = (test_vecs_w2v, y_test),callbacks=[earlyStopping])
        model.save('../model_saved/cnn.h5')

    print("_________________")
    print("ORIGINAL SCORE")
    test_vecs_w2v = sequence.pad_sequences(test_vecs_w2v, maxlen=300)
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=512)
    print(score[1])
    score = model.evaluate(TRANSFER_X_test, TRANSFER_y_test, batch_size=512)
    print(score[1])
    print("_________________")

    with open('../data/TRANSFER_train_vecs_w2v.pkl', 'rb') as fp:
        TRANSFER_X_train = pickle.load(fp)
    with open('../data/TRANSFER_y_train.pkl', 'rb') as fp:
        TRANSFER_y_train = pickle.load(fp)
    TRANSFER_X_train = sequence.pad_sequences(TRANSFER_X_train, maxlen=300)

    for layer in model.layers[:-3]:
        layer.trainable = False


    history = model.fit(TRANSFER_X_train, TRANSFER_y_train, epochs=30, batch_size=512 ,validation_data = (TRANSFER_X_test, TRANSFER_y_test))

    print("_________________")
    print("TRANSFER SCORE")
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=512)
    print(score[1])
    score = model.evaluate(TRANSFER_X_test, TRANSFER_y_test, batch_size=512)
    print(score[1])
    model.save('../model_saved/current_transfer.h5')
    plt.figure(figsize=(12, 14))
    plt.ion()
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
