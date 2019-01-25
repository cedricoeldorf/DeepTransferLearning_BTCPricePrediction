import csv
import pandas as pd
import numpy as np
import pickle

train_path = ['../data/amazon_train.csv',
'../data/sanders.csv',
'../data/twitter_stock.csv',
'../data/tweets_corpus_general.csv']



X_main = []
Y_main = []
X_transfer = []
Y_transfer = []
X_combined = []
Y_combined = []

###################################################
## Create main
print("Creating Main Set ...")
df = pd.read_csv(train_path[1])
for i in range(0,len(df)):
    if df.loc[i].Sentiment == 'positive':
        X_main.append(df.loc[i].TweetText)
        Y_main.append(1)
        X_combined.append(df.loc[i].TweetText)
        Y_combined.append(1)
    if df.loc[i].Sentiment == 'negative':
        X_main.append(df.loc[i].TweetText)
        Y_main.append(0)
        X_combined.append(df.loc[i].TweetText)
        Y_combined.append(0)
df = pd.read_csv(train_path[3], error_bad_lines = False)
for i in range(0,len(df)):
    X_main.append(df.loc[i].SentimentText)
    Y_main.append(df.loc[i].Sentiment)
    X_combined.append(df.loc[i].SentimentText)
    Y_combined.append(df.loc[i].Sentiment)

## Save and free memory
print("Saving . . .")
X_main = np.asarray(X_main)
Y_main = np.asarray(Y_main)
with open('../data/X_main.pkl', 'wb') as fp:
		pickle.dump(X_main, fp)
with open('../data/Y_main.pkl', 'wb') as fp:
		pickle.dump(Y_main, fp)

del X_main
del Y_main
#####################################################

## Create Secondary
print("Creating Secondary Set ...")
df = pd.read_csv(train_path[2])
for i in range(0,len(df)):
    X_transfer.append(df.loc[i].text)
    if df.loc[i].sentiment == 'positive':
        Y_transfer.append(1)
    else:
        Y_transfer.append(0)
    X_combined.append(df.loc[i].text)
    if df.loc[i].sentiment == 'positive':
        Y_combined.append(1)
    else:
        Y_combined.append(0)

print("Saving . . .")
X_transfer = np.asarray(X_transfer)
Y_transfer = np.asarray(Y_transfer)
with open('../data/X_transfer.pkl', 'wb') as fp:
		pickle.dump(X_transfer, fp)
with open('../data/Y_transfer.pkl', 'wb') as fp:
		pickle.dump(Y_transfer, fp)

print("Finishing . . .")
# save combined set
X_combined = np.asarray(X_combined)
Y_combined = np.asarray(Y_combined)
with open('../data/X_combined.pkl', 'wb') as fp:
		pickle.dump(X_combined, fp)
with open('../data/Y_combined.pkl', 'wb') as fp:
		pickle.dump(Y_combined, fp)
