import tweepy
import pandas as pd
import json
import numbers
import re
import os.path
import datetime

pd.options.display.max_colwidth = 400
pd.options.display.max_rows = 25
pd.options.display.max_columns = None

def load_credentials():
    consumer_key, consumer_key_secret, access_token, access_token_secret = (None,)*4
    if not os.path.isfile('credentials.ini'):
        return consumer_key, consumer_key_secret, access_token, access_token_secret
    lines = [line.rstrip('\n') for line in open('credentials.ini')]
    chars_to_strip = " \'\""
    for line in lines:
        if "consumer_key" in line and 'fill_in' not in line:
            consumer_key = re.findall(r'[\"\']([^\"\']*)[\"\']', line)[0]
        if "consumer_secret" in line and 'fill_in' not in line:
            consumer_key_secret = re.findall(r'[\"\']([^\"\']*)[\"\']', line)[0]
        if "access_token" in line and 'fill_in' not in line:
            access_token = re.findall(r'[\"\']([^\"\']*)[\"\']', line)[0]
        if "access_secret" in line and 'fill_in' not in line:
            access_token_secret = re.findall(r'[\"\']([^\"\']*)[\"\']', line)[0]
    return consumer_key, consumer_key_secret, access_token, access_token_secret

#consumer_key, consumer_key_secret, access_token, access_token_secret = load_credentials()
consumer_key = "UGDh2ccTuuXAKKavSGorMI0X3"
consumer_secret = "CzZBPgdQePFwht1uHkB2yONCOAS9veVcH9TW3az54e7UBEoMIn"
access_token = "1151369185-2jJHksydQDCILzmsByo7oFnudTrVozkDCSpjFWg"
access_secret = "be9QSGphg48lNJeo3du9rvlH6uZVgA3fQazbLycHYu3kU"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

class MyStreamListener(tweepy.StreamListener):
    """ Options you can set by passing to the MyStreamListener object:
        limit: int, how many tweets to capture
        print_output: bool, whether to print the tweet to screen
        save_output: bool, whether to save the tweet data to a csv file
        filename: str, the filename to name the saved output, by default it's file.csv
        include_rts: bool, whether to capture retweets
        strict_text_search: bool, ocasionally, stream will capture a tweet that doesn't actually include the search query
            set to True to filter out these "accidental" tweets
        search_terms: str or array, pass in the search query or an array of terms you want to use for filtering
            if strict_text_search = True. Script checks and turns any string into array of strings
    """
    def __init__(self,limit=20,print_output=True,save_output=True,
                 filename='file.csv',include_rts=True,strict_text_search=False,
                 search_terms=None):
        self.df = pd.DataFrame()
        self.limit = limit
        self.counter = 0
        self.print_output = print_output
        self.header=False
        self.save_output=save_output
        self.filename=filename
        self.include_rts=include_rts
        self.strict_text_search = strict_text_search
        self.search_terms = search_terms

    def on_data(self, data):
        d = {}
        decoded = json.loads(data)
        # full list of fields you can collect: https://dev.twitter.com/overview/api/tweets
        tweet_fields_to_collect = ['created_at','id','text','favorite_count','lang','retweet_count','retweeted','truncated']
        user_fields_to_collect = ['screen_name','id_str','statuses_count','followers_count','friends_count','favourites_count']
        if self.strict_text_search:
            if not isinstance(self.search_terms, list):
                self.search_terms = re.findall(r"[\w']+", self.search_terms)
            if not any(term.lower() in decoded['text'].lower() for term in self.search_terms):
                print("skipped")
                print(decoded['text'])
                return True
        for k,v in iter(decoded.items()):
            if k in tweet_fields_to_collect:
                if isinstance(v, numbers.Number):
                    v = str(v)
                try:
                    d['tweet_' + k.strip()] = v
                except:
                    print("Failure collecting tweet field", v.encode('ascii', 'ignore'))
            if k=='user':
                for user_k,user_v in iter(v.items()):
                    if user_k in user_fields_to_collect:
                        if isinstance(user_v, numbers.Number):
                            user_v = str(user_v)
                        try:
                            d[user_k.strip()]=user_v
                        except:
                            print("Failure collecting user field",user_v.encode('ascii', 'ignore'))
            if k=='retweeted_status':
                for retweet_k,retweet_v in iter(v.items()):
                    if retweet_k in tweet_fields_to_collect:
                        if isinstance(retweet_v, numbers.Number):
                            retweet_v = str(retweet_v)
                        try:
                            d['retweet_'+retweet_k.strip()]=retweet_v
                        except:
                            print("Failure collecting retweet field",user_v.encode('ascii', 'ignore'))
        if not self.include_rts:
            if ('retweet_text' in d and len(d['retweet_text'])>0) or d['tweet_text'].startswith('RT @'):
                return True
        tweet_df = pd.DataFrame(d, index=[0])
        frames = [self.df, tweet_df]
        self.df = pd.concat(frames)
        self.counter+=1
        if self.print_output:
            try:
                print(decoded['text'])
            except:
                print("Failure outputting tweet text",decoded['text'].encode('ascii', 'ignore'))
        if self.counter>=self.limit:
            print("finished collecting %s tweets, ending" % self.limit)
            if self.include_rts and 'retweet_text' in self.df.columns:
                self.df = self.df[['tweet_' + x for x in tweet_fields_to_collect] + user_fields_to_collect + ['retweet_' + x for x in tweet_fields_to_collect]]
            else:
                self.df = self.df[['tweet_' + x for x in tweet_fields_to_collect] + user_fields_to_collect]
            self.df.rename(columns={'id_str':'user_id'},inplace=True)
            self.df.to_csv(self.filename, index=False, encoding='utf-8')
            return False
        else:
            return True

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def on_disconnect(self, notice):
        print("disconnecting due to " + str(notice))



search_query = 'bitcoin,btc,ethereum,eth,ripple,xrp,bitcoin cash,bcc,eos,litecoin,ltc,cardano,ada,stellar,xlm,iota,neo,neo_blockchain,monero,xmr,dash, crypto, cryptocurrency, cryptocurrencies, blockchain, coinmarketcap,tron,tether,vechain,coindesk,cointelegraph'
filename = '%s.csv' % (datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))

"""
myStreamListener = MyStreamListener(limit=10000,print_output=False,
                                    filename=filename,
                                    search_terms=search_query,
                                    include_rts=False)
myStream = tweepy.Stream(auth, listener=myStreamListener)
myStream.filter(track=[search_query],languages=["en"], )

#df = pd.read_csv(filename)
df = myStreamListener.df
df['followers_count'] = df['followers_count'].apply(pd.to_numeric)
df = df[df['followers_count'] >= 100]
df['tweet_text'] = df['tweet_text'].str.lower()
df = df[df['tweet_text'].str.contains(('|'.join(['launch','payment','payments','reward','whitepaper','startup','airdrop','opportunity','sale','goal','invest','ico','promising','team','recommended','recommended','claim','min','referral','nazi','%','discount','free', 'solution', 'subscribe','apple','giveaway','affiliate','website','business','click here','check here','register','registration','join','earn','rich']))) == False]
df.drop(['favourites_count', 'followers_count', 'friends_count', 'id_str',
       'screen_name', 'statuses_count', 'tweet_created_at',
       'tweet_favorite_count', 'tweet_id', 'tweet_lang', 'tweet_retweet_count',
       'tweet_retweeted','tweet_truncated'], axis=1, inplace=True)
df.to_csv(filename, index = False)
"""

myStreamListener = MyStreamListener(limit=1000,print_output=True,
                                    filename=filename,
                                    include_rts=False)
myStream = tweepy.Stream(auth, listener=myStreamListener)
myStream.filter(follow=['110759328','951587699269472258','361289499','220283722','3187848089',
'2263963231','294376526','877728873340956672','3179873194','2207129125','1333467482','873651533807845376'] )

#df = pd.read_csv(filename)
df = myStreamListener.df

df['tweet_text'] = df['tweet_text'].str.lower()
df = df[df['tweet_text'].str.contains(('|'.join(['launch','payment','payments','reward','whitepaper','startup','airdrop','opportunity','sale','goal','invest','ico','promising','team','recommended','recommended','claim','min','referral','nazi','%','discount','free', 'solution', 'subscribe','apple','giveaway','affiliate','website','business','click here','check here','register','registration','join','earn','rich']))) == False]
df.drop(['favourites_count', 'followers_count', 'friends_count', 'id_str',
       'screen_name', 'statuses_count',
       'tweet_favorite_count', 'tweet_id', 'tweet_retweet_count',
       'tweet_retweeted','tweet_truncated'], axis=1, inplace=True)
df.to_csv(filename, index = False)
