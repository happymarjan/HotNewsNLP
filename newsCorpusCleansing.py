#!/usr/bin/env python3.4

#Author: Marjan Alavi

import pandas as pd
import sqlite3
import re
from functools import partial
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

import sys

#----------------------------------------------------------------------------
tweetcleanedFile='tweetcleaned.txt'

extendedStop = ['would', 'will', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'with', 'into', 'to', 'from', 'in', 'again', 'further', 
'then', 'once', 'here', 'there', 'other','such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
's', 't', 'can', 'will', 'just', 'don', 'should', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 
'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

#----------------------------------------------------------------------------

DBName = 'newsDB.db'
dbase = sqlite3.connect(DBName)
curr = dbase.cursor()
df = pd.read_sql("select id_str, tweet, ctime, rtCnt from newsTbl",dbase)

#----------------------------------------------------------------------------

def tweetToWordList(tweet, removeStop=False, stemmingEnabled=False,lemmatizationEnabled=False):
    tweet=tweet.lower()
    tweet=" ".join(filter(lambda x:x[0]!='@' and x[0]!='#' and not x.startswith("http") and x!="rt", tweet.split()))    
    tweet = re.sub("[^a-zA-Z]"," ", tweet)
    #re.sub(r'\b\w{1,3}\b', '', tweet)
    tweetWords = tweet.split()
    if removeStop:
        stops = set(stopwords.words("english"))
        tweetWords = [w for w in tweetWords if not w in stops and not w in extendedStop and len(w)>=3] 
        if stemmingEnabled:
            tweetWords = [SnowballStemmer('english').stem(w) for w in tweetWords] 
            tweetWords = [word for word in tweetWords if len(word)>=3]
        if lemmatizationEnabled:
            lemma = WordNetLemmatizer()
            tweetWords = [lemma.lemmatize(word) for word in tweetWords]           
            tweetWords = [lemma.lemmatize(word,pos='v') for word in tweetWords]
            tweetWords = [word for word in tweetWords if len(word)>=3]
            
    return(' '.join(tweetWords))
    
#----------------------------------------------------------------------------
        
if __name__ == "__main__":
    inplen=len(df.index)
    print(inplen)
    
    mapFunc = partial(tweetToWordList, removeStop=True, stemmingEnabled=False,lemmatizationEnabled=True)
    inputCorpus = df['tweet'].map(mapFunc)
    
    tweetIds=df['id_str']

    print("Size of the collected tweet corpus")
    print(len(inputCorpus))
    
    i=0
    with open(tweetcleanedFile, 'w') as f:
        for inp in inputCorpus:
            f.write(tweetIds[i]+","+ inp+"\n")
            i+=1
            
#----------------------------------------------------------------------------
            