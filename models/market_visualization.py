###########################################
#Market Visualization
###########################################

## get all tweet file names
## for each filename, predict sentiment
## get average sentiment
## populate df with filename (date) and sentiment
from os import listdir
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from keras.models import load_model
import matplotlib.pyplot as plt
def ingest(type = 'main'):
    if type == "main":
        data = pd.read_csv('../data/trainingandtestdata/tweets.csv',encoding='latin-1',
        header = None, names=['Sentiment','ItemID','Date','Blank','SentimentSource','SentimentText'])
        data.drop(['ItemID', 'SentimentSource'], axis=1, inplace=True)
        data = data[data.Sentiment.isnull() == False]
        data['Sentiment'] = data['Sentiment'].map(int)
        data['Sentiment'] = data['Sentiment'].map( {4:1, 0:0} )
        data = data[data['SentimentText'].isnull() == False]
        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        print('dataset loaded with shape', data.shape)
    if type == 'transfer':
        data = pd.read_csv('../data/twitter_stock_1.csv')
        data = data.sample(frac=1).reset_index(drop=True)
        data = data[data.sentiment.isnull() == False]
        data['Sentiment'] = data.sentiment.map( {'negative':0, 'positive':1} )
        data['SentimentText'] = data.text
        data.drop(['created_at','text','sentiment'], axis=1, inplace=True)

        data.reset_index(inplace=True)
        data.drop('index', axis=1, inplace=True)
        print('dataset loaded with shape', data.shape)
    if type == 'new':
        data = pd.read_csv('../data/tweets_corpus_general.csv', error_bad_lines=False)
        data['Sentiment'] = data['target""'].map( {'negative""':0, 'positive""':1} )
        data = data[data['Sentiment'].isnull() == False]

    return data

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D|;D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def tokenize(tweet):
    try:
        #tweet = unicode(tweet.decode('utf-8').lower())
        tweet = handle_emojis(tweet)
        tweet = tweet.lower()
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
        # Replace @handle with the word USER_MENTION
        tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
        # Replaces #hashtag with hashtag
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
        tokens = tokenizer.tokenize(tweet)
        return tokens
    except:
        return 'NC'

def postprocess(data, n=1000000):
    #data = data.head(n)
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

def buildWordVector(tokens, size, tweet_w2v, tfidf):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


averages = pd.read_csv('../visualizations/average_sentiment.csv')
    ## Data
onlyfiles = [f for f in listdir('../twitter_stream/clean')]
tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('../model_saved/GoogleNews-vectors-negative300.bin', binary = True)
for i in range(len(onlyfiles)):
    data = pd.read_csv('../twitter_stream/clean/' + str(onlyfiles[i]))
    data['SentimentText'] = data.tweet_text
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    ## model
    data = np.array(data.tokens)
    data = labelizeTweets(data, 'TEST')
    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in data])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    import gensim

    n_dim = 300
    n=1000000
    pred_data = np.concatenate([buildWordVector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, data))])
    pred_data = scale(pred_data)


    ##################
    ## MODEL
    ##
    model = load_model('../model_saved/current_transfer.h5')
    sentiment = model.predict(pred_data)
    averages.loc[i] = [onlyfiles[i][:-4], sentiment.mean(),]

averages.to_csv('../twitter_stream/plot.csv', index = False)

prices = [331795931599,328698372110,326101899973,301223306912,324610490333]
#prices = prices / np.linalg.norm(prices)
plt.ion()

#plt.plot(averages.date,prices)
#plt.plot(averages.date,averages.average_sentiment)

plt.figure(figsize=(12, 14))
plt.title("Twitter Sentiment and Total Market Capitalization")
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ind = averages.date
plt.ylim(0.5, 0.8)
plt.bar(ind, averages.average_sentiment, color='#3F5D7D', label='Average Sentiment')
plt.ylabel('Sentiment')
plt.legend()
x = ind
y = prices
axes2 = plt.twinx()
axes2.plot(x, y, color='#e7a12a', label='Total Market Capitalization',linewidth=4)
#axes2.set_ylim(-1, 1)
axes2.set_ylabel('USD')

plt.show()
