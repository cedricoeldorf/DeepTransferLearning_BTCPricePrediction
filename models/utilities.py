import pickle

def load_final_data(avg_length,max_tweet_length):

    ## Create testing and transfer data
    X_transfer_vecs = pickle.load(open('../data/X_transfer_vecs.pkl', 'rb'))
    X_transfer = X_transfer_vecs[len(X_transfer_vecs)/3:]
    X_test = X_transfer_vecs[0:len(X_transfer_vecs)/3]
    del X_transfer_vecs

    ## Turn into shape for NN
     Create train and test sets
    # Generate random indexes
    indexes = set(np.random.choice(len(corpus), train_size + test_size, replace=False))

    X_test = np.zeros((test_size, max_tweet_length, vector_size), dtype=K.floatx())
    Y_test = np.zeros((test_size, 2), dtype=np.int32)

    print("Creating data ....")
    for i, index in enumerate(indexes):
        for t, token in enumerate(corpus[index]):
            if t >= max_tweet_length:
                break

            if token not in X_vecs:
                continue

            if i < train_size:
                X_train[i, t, :] = X_vecs[token]
            else:
                X_test[i - train_size, t, :] = X_vecs[token]

        if i < train_size:
            Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        else:
            Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]



    X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())
    Y_train = np.zeros((train_size, 2), dtype=np.int32)

    print("Creating data ....")
    for i, index in enumerate(indexes):
        for t, token in enumerate(corpus[index]):
            if t >= max_tweet_length:
                break

            if token not in X_vecs:
                continue

            if i < train_size:
                X_train[i, t, :] = X_vecs[token]
            else:
                X_test[i - train_size, t, :] = X_vecs[token]

        if i < train_size:
            Y_train[i, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]
        else:
            Y_test[i - train_size, :] = [1.0, 0.0] if labels[index] == 0 else [0.0, 1.0]







    corpus = pickle.load(open('../data/X_main_vecs.pkl', 'rb'))
    X_main_vecs = pickle.load(open('../data/X_main.pkl', 'rb'))
    # Train subset size (0 < size < len(tokenized_corpus))
    train_size = int(corpus.shape[0] * 0.9)
