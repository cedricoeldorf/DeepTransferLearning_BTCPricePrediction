#############################
## market visualize seperate sources
############################

import pandas as pd
import matplotlib as plt
import numpy as np
from os import listdir
from cryptocmd import CmcScraper
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

tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('../model_saved/GoogleNews-vectors-negative300.bin', binary = True)

onlyfiles = [f for f in listdir('../twitter_stream/seperate')]

def get_market(coin = 'BTC'):
    scraper = CmcScraper('BTC', '06-03-2018', '30-05-2018')
    headers, data = scraper.get_data()
    scraper.export_csv(csv_path='/home/cedric/Documents/UM/Info_mining/twitter_stream/market')

def clean_source(file):

    data = pd.read_csv('../twitter_stream/seperate/' + str(file))
    data['SentimentText'] = data.text
    data['tokens'] = data['SentimentText'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    data.drop('id', inplace=True, axis=1)
    dates = data.created_at
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
    df = pd.DataFrame({'date':dates.tolist(),'sentiment':sentiment.ravel().tolist()})
    df.date = pd.to_datetime(df.date)
    df.index = df.date
    df.drop('date',inplace = True, axis = 1)
    df = df.sentiment.rolling(25, center = True)
    df = pd.DataFrame({'sentiment':df.mean(),'std':df.std()})

    #df = df.groupby('sentiment').rolling('12H').mean()
    df = df.groupby(pd.TimeGrouper(freq='1D')).mean()
    df = df.loc['2018-03-06':]
    #df.dropna()
    return df

plt.ion()
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)



plot_type =1
if plot_type == 0:
    plt.figure(figsize=(12, 14))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    for i in range(len(onlyfiles)):
        df = clean_source(onlyfiles[i])
        df = df.fillna(method='ffill')
        plt.plot(df.sentiment,color=tableau20[i])
        plt.text('2018-05-31', df.sentiment[-1], onlyfiles[i][:-4], fontsize=10, color=tableau20[i])
    plt.show()
else:
    merged = pd.DataFrame()
    for i in range(len(onlyfiles)):
        df = clean_source(onlyfiles[i])
        df = df.fillna(method='ffill')
        merged = pd.concat([merged,df])
    merged = merged.groupby(pd.TimeGrouper(freq='1D')).mean()
    plt.figure(figsize=(12, 14))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0.54, 0.85)
    plt.fill_between(merged.index, merged.sentiment - merged['std'],
                 merged.sentiment + merged['std'], color="#3F5D7D")
    plt.plot(merged.index, merged.sentiment, color="white", lw=2)

    prices = pd.read_csv('../twitter_stream/market/this.csv')
    prices.Date = pd.to_datetime(prices.Date)
    prices.index = prices.Date
    prices.drop(['Date','Open*', 'High', 'Low', 'Volume', 'Market Cap'],inplace = True, axis = 1)
    prices = prices.Close.rolling(3, center = True)
    prices = pd.DataFrame({'Close':prices.mean()})
    prices = prices.fillna(method='ffill')
    axes2 = plt.twinx()
    axes2.bar(prices.index, prices['Close'], color=tableau20[3], label='Bitcoin Closing Price',linewidth=4)
    #axes2.set_ylim(-1, 1)
    axes2.set_ylabel('USD')
    plt.ylim(6000,11000)
    plt.legend()
    plt.show()

### interpolate nans
### get all sources, average them, get error
### plot each source
### plot all sources and market data
