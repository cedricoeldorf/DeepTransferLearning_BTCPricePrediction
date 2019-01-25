####################################
## visualize tweets
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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



#
"""
TRANSFER_X_test_embedded = TSNE(n_components=3,verbose=2).fit_transform(TRANSFER_X_test)
colors = ['b','r']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,2):
    points = np.array([TRANSFER_X_test_embedded[j] for j in range(len(TRANSFER_X_test_embedded)) if str(TRANSFER_y_test[j]) == str(i)], dtype = 'float')
    print(points.shape)
    ax.scatter(points[:, 0], points[:, 1], c=colors[i])

plt.title("TSNE of Word Vectors - Tranfer Data")
plt.ion()
plt.show()
"""
#train_vecs_embedded = TSNE(n_components=3,verbose=2, learning_rate = 80).fit_transform(train_vecs_w2v[0:15000])
colors = ['b','r']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0,2):
    points = np.array([train_vecs_embedded[j] for j in range(len(train_vecs_embedded)) if str(y_train[j]) == str(i)], dtype = 'float')
    print(points.shape)
    ax.scatter(points[:, 0], points[:, 1], c=colors[i])

plt.title("TSNE of Word Vectors - Main Data")
plt.ion()
plt.show()
