#########################
## https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html
#########################
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

type = input("create main or transfer data? (main/transfer)")
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


if type == 'main':
    data = ingest(type)
    data = postprocess(data)
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                        np.array(data.Sentiment), test_size=0.2)
    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')
    ########################
    ## Use pretrained w2v
    ## create tf idf matrix of our data
    ########################
    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in x_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    import gensim
    tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('../model_saved/GoogleNews-vectors-negative300.bin', binary = True)
    n_dim = 300
    n=1000000
    train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, x_train))])
    train_vecs_w2v = scale(train_vecs_w2v)
    test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, x_test))])
    test_vecs_w2v = scale(test_vecs_w2v)
    with open('../data/train_vecs_w2v.pkl', 'wb') as fp:
    		pickle.dump(train_vecs_w2v, fp)
    with open('../data/test_vecs_w2v.pkl', 'wb') as fp:
    		pickle.dump(test_vecs_w2v, fp)
    with open('../data/y_train.pkl', 'wb') as fp:
    		pickle.dump(y_train, fp)
    with open('../data/y_test.pkl', 'wb') as fp:
    		pickle.dump(y_test, fp)

if type == 'transfer':
    data = ingest(type)
    data = postprocess(data)
    x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(int(len(data)*0.5)).tokens),
                                                        np.array(data.head(int(len(data)*0.5)).Sentiment), test_size=0.2)
    x_train = labelizeTweets(x_train, 'TRAIN')
    x_test = labelizeTweets(x_test, 'TEST')
    ########################
    ## Use pretrained w2v
    ## create tf idf matrix of our data
    ########################
    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in x_train])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    import gensim
    tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('../model_saved/GoogleNews-vectors-negative300.bin', binary = True)
    n_dim = 300
    n=1000000
    train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, x_train))])
    train_vecs_w2v = scale(train_vecs_w2v)
    test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, x_test))])
    test_vecs_w2v = scale(test_vecs_w2v)
    with open('../data/TRANSFER_train_vecs_w2v.pkl', 'wb') as fp:
    		pickle.dump(train_vecs_w2v, fp)
    with open('../data/TRANSFER_test_vecs_w2v.pkl', 'wb') as fp:
    		pickle.dump(test_vecs_w2v, fp)
    with open('../data/TRANSFER_y_train.pkl', 'wb') as fp:
    		pickle.dump(y_train, fp)
    with open('../data/TRANSFER_y_test.pkl', 'wb') as fp:
    		pickle.dump(y_test, fp)
