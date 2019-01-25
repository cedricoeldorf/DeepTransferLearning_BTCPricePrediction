import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
import string
import multiprocessing
from gensim.models.word2vec import Word2Vec
def preprocess_word(text):
    # Remove punctuation
    text = text.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Remove - & '
    text = re.sub(r'(-|\')', '', text)
    text = re.sub(r'thx', 'thanks', text)
    text = re.sub(r'thank you', 'thanks', text)
    text = re.sub(r'thank you', 'thanks', text)
    text = re.sub(r'bout', 'about', text)
    text = re.sub(r'cannot', 'cant', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet

def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()
    tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
    stemmer = LancasterStemmer()
    for word in words:
        #print(word)
        word = preprocess_word(word)
        if is_valid_word(word):
            if word not in stopwords.words('english'):
                word = stemmer.stem(word)
                processed_tweet.append(word)

            #print(word)

    return ' '.join(processed_tweet)

#################
## Word2Vec

comb = input("Has combined been cleaned? (y/n) ")
if comb == 'n':
    print("Loading Combined . . .")
    X = pickle.load(open('../data/X_combined.pkl', 'rb'))
    X_p = []
    for i in range(0,len(X)):
        print(str(i) + '/' +str(len(X)))
        if len(X[i].split()) > 2:
            X_p.append(preprocess_tweet(X[i]))
    with open('../data/X_combined_clean.pkl', 'wb') as fp:
            pickle.dump(X_p, fp)
else:
    X = pickle.load(open('../data/X_combined_clean.pkl', 'rb'))

print("Creating W2V . . .")
vector_size = 512
window_size = 10
word2vec = Word2Vec(sentences=X,
                    size=vector_size,
                    window=window_size,
                    negative=20,
                    iter=50,
                    seed=1000,
                    workers=multiprocessing.cpu_count())

print("saving model")
word2vec.save('../model_saved/word2vec_full.model')

#######################
## Vectorize main and transfer sets
#######################
"""
X = pickle.load(open('../data/X_main.pkl', 'rb'))
X_p = []
for i in range(0,len(X)):
    print(str(i) + '/' +str(len(X)))
    if len(X[i].split()) > 2:
        X_p.append(preprocess_tweet(X[i]))
with open('../data/X_main_clean.pkl', 'wb') as fp:
        pickle.dump(X_p, fp)
print("Vectorizing main . . .")
"""
X_p = pickle.load(open('../data/X_main_clean.pkl', 'rb'))
X_main_vecs = word2vec.transform(X_p)
with open('../data/X_main_vecs.pkl', 'wb') as fp:
        pickle.dump(X_main_vecs, fp)
del X_main_vecs

X = pickle.load(open('../data/X_transfer.pkl', 'rb'))
X_p = []
for i in range(0,len(X)):
    print(str(i) + '/' +str(len(X)))
    if len(X[i].split()) > 2:
        X_p.append(preprocess_tweet(X[i]))
with open('../data/X_transfer_clean.pkl', 'wb') as fp:
        pickle.dump(X_p, fp)
print("Vectorizing transfer . . .")

X_transfer_vecs = word2vec.transform(X_p)
with open('../data/X_transfer_vecs.pkl', 'wb') as fp:
        pickle.dump(X_transfer_vecs, fp)


del X_transfer_vecs
del X
