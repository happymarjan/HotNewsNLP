#!/usr/bin/env python3.4

#Author: Marjan Alavi

import numpy as np
from nltk import pos_tag
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.utils import to_unicode
from gensim.models import Phrases
import lda
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.plotting import show
import sqlite3
import logging
import random
import collections
import sys

#--------------------------------------------------------------

colormap = np.array([
"#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", 
"#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5", 
"#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", 
"#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
"#e7cb94", "#f2d979", "#79a7f2", "#c0f279", "#e279f2"
])

tweetcleaned='tweetcleaned.txt'
kmeansClustered = 'kmeansClustered.txt'
clusterFeatures = 'kmeansclusterFeatures.txt'
tweetsExtracted = 'tweetsExtracted.txt'
top30TweetsPerCluster ='top30TweetsPerCluster.txt'
finalTweets = 'finalTweets.txt'
vecSize = 50
w2vMdlName ='w2vModel'
DBName = 'newsDB.db'
inputCorpus=[]
numKmeanClusters = 25
numLDATopics = 25


log = logging.getLogger("HotNews")
logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("lda").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("bokeh").setLevel(logging.ERROR)

#--------------------------------------------------------------

def computeTfIdfScores(inputCorpus): 
    #featureSize determined by the tfidf algorithm itself  #max_features = 500,sublinear_tf=True,ngram_range=(1, 2)       
    tfidfvectorizer = TfidfVectorizer(min_df=5, max_df=0.5, norm='l2', max_features = None,use_idf=True,smooth_idf=True,stop_words = 'english')
    tfidfMatrix = tfidfvectorizer.fit_transform(inputCorpus) #(inputSize, featureSize) matrix
    #Learned idf vector per feature association in the tweet; idf_: global term weights
    tfidfDict = dict(zip(tfidfvectorizer.get_feature_names(), tfidfvectorizer.idf_)) 
    return tfidfvectorizer,tfidfMatrix,tfidfDict

#--------------------------------------------------------------

    '''build WordtoVec model'''

def buildW2VModel(inputCorpus): 
    input=[[to_unicode(str.encode(x)) for x in s.split()] for i,s in enumerate(inputCorpus)]    
    sentsStream = [doc.split(" ") for doc in inputCorpus]
    bigrams = Phrases(sentsStream, min_count=4, threshold=2)
    bigramEnrichedInput=[]
    
    cnt=0
    for inp in input:
        bigramEnrichedInput.append(bigrams[inp])
        cnt+=len(bigrams[inp])
            
        
    print("words size")
    print(cnt)
    
    numSentences=len(bigramEnrichedInput)
    model = Word2Vec(min_count=4, window=7, size=vecSize, sample=1e-4, workers=4, negative=5)  
    model.build_vocab(bigramEnrichedInput)
    for epoch in range(50):
        shuffled=sorted(bigramEnrichedInput, key=lambda k: random.random())
        model.train(shuffled,total_examples=numSentences) 
    model.save('./tweets.w2v') 
    return model

#-------------------------------------------------------------- 

    '''build DoctoVec model'''
    
def buildD2VModel(wordsList):
    model = Doc2Vec(min_count=4, window=7, size=vecSize, sample=1e-4, workers=4, negative=5)
    model.build_vocab(wordsList)
    numSentences=len(wordsList)
    print("numSentences")
    print(numSentences)
    for epoch in range(50):
        shuffled=sorted(wordsList, key=lambda k: random.random())
        model.train(shuffled,total_examples=numSentences) 
    model.save('./tweets.d2v') 
    return model
#-------------------------------------------------------------- 


    '''performing linear dimensionality reduction by truncated singular value decomposition (LSA)'''

def reduceDimentionalityLSA(inputMatrix):    
    svdModel = TruncatedSVD(n_components=100)  #random_state=0
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svdModel, normalizer)
    svdOutMatrix = lsa.fit_transform(inputMatrix)
    #explained_variance = svd.explained_variance_ratio_.sum()
    return svdOutMatrix
    
#--------------------------------------------------------------

    '''performing dimensionality reduction by applying PCA (for dense matrices)'''

def reduceDimentionalityPCA(inputMatrix):    
    pcaModel = PCA()
    normalizer = Normalizer(copy=False)
    pca = make_pipeline(pcaModel, normalizer)
    pcaOutMatrix = pca.fit_transform(inputMatrix)
    return pcaOutMatrix
    
#--------------------------------------------------------------

    '''TSNE dimensionality reduction'''
    
def reduceDimentionalityTSNE(inputMatrix):
    tsneModel = TSNE(n_components=2, verbose=1, random_state=0)
    svdOutMatrix = tsneModel.fit_transform(inputMatrix) #(inputSize, 2)
    return svdOutMatrix

#--------------------------------------------------------------

    ''' cluster tweets based on Kmeans'''

def kmeansClustering(tfidfMatrix,useDefaultNumClusters=False):
    global numKmeanClusters
    if useDefaultNumClusters:
        kmeansModel = MiniBatchKMeans(init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False) 
    else:
        kmeansModel = MiniBatchKMeans(n_clusters=numKmeanClusters,init='k-means++', n_init=1, 
                         init_size=1000, batch_size=1000, verbose=False) 
    #Computes k-means clustering
    kmeans = kmeansModel.fit(tfidfMatrix) 
    #for future predictions kind of semi-supervised, Predict the closest cluster each sample in matrix belongs to; shape:(inputSize,)
    kmeansClusters = kmeans.predict(tfidfMatrix) 
    #Transform matrix to a cluster-distance space; shape:(inputSize, numKmeanClusters)
    kmeansDistances = kmeans.transform(tfidfMatrix) 
    numKmeanClusters=kmeansModel.get_params().get('n_clusters')
    return kmeansModel,kmeans,kmeansClusters,kmeansDistances
#--------------------------------------

    '''get a dictionary for the distance of each tweet to the closest cluster it belongs to and print each cluster and its corresponding tweet texts with the associated distance'''

def getKmeansClustersTextDists(): 
    clusterToTextsDict = {}
    closestTweetsToCentroid=collections.OrderedDict()
    for i, q in enumerate(inputCorpus):
        dist = format(kmeansDistances[i][kmeansClusters[i]],'.3f')
        tweetid=tweetIds[q]
        clusterToTextsDict.setdefault(kmeansClusters[i],[]).append((str(tweetid),q.strip(),dist))
    with open(kmeansClustered, 'w') as f:
        tmpstr ="Tweet categories clustered using Kmeans \n"
        print(tmpstr)
        f.write(tmpstr)
        for k in clusterToTextsDict:
            tmpstr ="\n Tweets in category "+str(k)+':\n'
            print(tmpstr)
            f.write(tmpstr)
            closestTweetsToCentroid[k]=sorted(clusterToTextsDict[k],key=lambda x:x[2], reverse=True)
            for item in closestTweetsToCentroid[k]:
                print(item) 
                f.write(str(item))
                f.write('\n')
    return closestTweetsToCentroid
    
#--------------------------------------------------------------

    '''selects top events(clusters) based on sum of all retweets in that cluster as a feature'''

def selectTopTwoEvents(closestTweetsToCentroid):
    eventPopularityDict={}
    for k in closestTweetsToCentroid:
        sumRetweetsInCluster=0
        for i,t,d in closestTweetsToCentroid[k][:]:
            record=getTweetInfoFromDB(str(i))
            sumRetweetsInCluster+=record[2]
        eventPopularityDict[k]=(k,sumRetweetsInCluster)
        
    topTwoEvents = sorted(eventPopularityDict.values(), key=lambda x:x[1],reverse=True)[:2]
    print("topTwoEvents")
    print(type(topTwoEvents))
    print(len(topTwoEvents))
    print(topTwoEvents)
    return topTwoEvents
    
#--------------------------------------------------------------           
               
    '''extracts top news in a cluster based on its closeness to the cluster centroid.
       Then for each candidate news uses TF-IDF weights and part of speech tags (POS) 
       to weight the tweets and rank them.'''
    
def extractTopNews(closestTweetsToCentroid,tfidfDict,topTwoEvents):
    extractedTweetsDict={}
    extractedTweetsDictOrdered=collections.OrderedDict()
    tags=['NN','NNS','NNP','NNPS','RB','VB']
    tags=set(tags)
    for k in closestTweetsToCentroid:
        #clusterToTextsDict[k]=sorted(clusterToTextsDict[k],key=lambda x:x[1])
        
        for i,t,d in closestTweetsToCentroid[k][:30]:
            words=t.split()
            pasTaggedTweet=pos_tag(words)
            avg=0.0
            tlen=len(words)
            for word,tag in pasTaggedTweet:
                if word in tfidfDict.keys() and tag in tags:
                    avg+=tfidfDict[word]
                else: 
                    continue
            avg/=tlen
            extractedTweetsDict.setdefault(k,[]).append((str(i),t.strip(),avg))
            
            
    with open(tweetsExtracted, 'w') as f:
        for k in extractedTweetsDict:
            tmpstr ="\n Extracted tweets in category "+str(k)+':\n'
            print(tmpstr)
            f.write(tmpstr)
            extractedTweetsDictOrdered[k]=sorted(extractedTweetsDict[k],key=lambda x:x[2], reverse=True)
            for item in extractedTweetsDictOrdered[k]:
                print(item) 
                f.write(str(item))
                f.write('\n')            
    
    getToppestPerCluster(extractedTweetsDictOrdered,topTwoEvents)
#--------------------------------------------------------------   

    '''gets the selected tweets per cluster and prints them to a file
       and finally prints the two selected news in finalTweets.txt file.'''
    
def getToppestPerCluster(extractedTweetsDictOrdered,topTwoEvents):
    with open(top30TweetsPerCluster, 'wb') as f:
        f.write(b"Tweet , Time Created , Number of Retweets")
        f.write(b'\n\n')
        for k in extractedTweetsDictOrdered:
            i,CleansedTweet,avg=extractedTweetsDictOrdered[k][0]
            record=getTweetInfoFromDB(str(i))
            print(record)
            txt=record[0].encode('ascii', 'ignore')+b", "
            txt+=record[1].encode('ascii','ignore')
            txt+=b", "
            txt+=str(record[2]).encode('ascii','ignore')
            txt+=b"\n\n"
            f.write(txt)
                     
    with open(finalTweets, 'wb') as f:
        f.write(b"Tweet , Time Created , Number of Retweets")
        f.write(b'\n\n')
        for event in topTwoEvents:
            clusterId,numofRetweets = event
            i,CleansedTweet,avg=extractedTweetsDictOrdered[clusterId][0]
            record=getTweetInfoFromDB(str(i))
            print(record)
            txt=record[0].encode('ascii', 'ignore')+b", "
            txt+=record[1].encode('ascii','ignore')
            txt+=b", "
            txt+=str(record[2]).encode('ascii','ignore')
            txt+=b"\n\n"
            f.write(txt)

#--------------------------------------------------------------   
    
    '''gets tweet attributes from database'''
            
def getTweetInfoFromDB(id_str):
    dbase = sqlite3.connect(DBName)
    cur = dbase.cursor()
    try:
        cur.execute('SELECT tweet,ctime, rtCnt FROM newsTbl WHERE id_str = ?',(id_str,))
        record = cur.fetchone()
        if not record is None:
            return record
    except sqlite3.Error as err:
        log.error('Error selecting the tweet associated with tweet id {0}: {1}'.format(id_str, err))
    return

#--------------------------------------------------------------

    '''prints the top 10 words per each cluster''' 

def getClusterFeatures(vctzerObj,kmeansModel):
    global numKmeanClusters
    sortedCentroids = kmeansModel.cluster_centers_.argsort()[:, ::-1]
    featureNames = vctzerObj.get_feature_names()
    with open(clusterFeatures, 'w') as f:
        tmpstr ="Top words per category clustered using Kmeans \n"
        print(tmpstr)
        f.write(tmpstr)
        for i in range(numKmeanClusters):
            tmpstr ="\n Cluster "+str(i)+':'
            print(tmpstr)
            f.write(tmpstr)
            for j in sortedCentroids[i, :10]:
                tmpstr =featureNames[j] + ' '
                print(tmpstr,end='')
                f.write(tmpstr)
        print('\n')
        
#--------------------------------------------------------------

def LDATopicModeling():
    
    '''Topic modeling using LDA and using the topic distributions.
    for each sentence as a measure for grouping similar sentences together.'''
    
    global numLDATopics
    #Vectorizing data by representing each tweet as a 500 dimensional vector
    cvectorizer = CountVectorizer(min_df=4, max_features=500) 
    #feature matrix
    countMatrix = cvectorizer.fit_transform(inputCorpus) 

    #remove all 0 rows (tweets that has no words left after preprocessing) both here and from the corpus as they are irrelevant for our purpose
    num_nonzeros = np.diff(countMatrix.indptr)
    countMatrix = countMatrix[num_nonzeros != 0]
    zeros=np.where(num_nonzeros==0)[0]
    for i in range(len(zeros)):
        del inputCorpus[i]
        
    #LDA tries to detect n_topics latent topics in the data
    ldaModel = lda.LDA(n_topics=numLDATopics, n_iter=2000) 
    #matrix of topic distributions
    topicMatrix = ldaModel.fit_transform(countMatrix)     
    #print(ldaModel.topic_word_ )
    return cvectorizer,ldaModel,topicMatrix

#--------------------------------------------------------------
    
    '''get and prints the most relevant words to a certain word using LDA'''

def getReleventWordstoTopics(cvectorizer,ldaModel,topicMatrix,numRelWords=10):
    topicRelevWords = []
    topicWords = ldaModel.topic_word_  #(numLDATopics, featureSize) , featureSIze determined by the model itself
    featureList = cvectorizer.get_feature_names()
    tmpstr ="Most related words per topic using LDA \n"
    print(tmpstr)
    for i, topicDist in enumerate(topicWords):
        relevantWords = np.array(featureList)[np.argsort(topicDist)][:-(numRelWords+1):-1]
        topicRelevWords.append(' '.join(relevantWords))
        print('Topic {}: {}'.format(i, ' '.join(relevantWords)))
    print('\n')

#--------------------------------------------------------------
 
def plotTSNE(modelName,matrix,entry,topicKeys=None):        
    title="Tweets visualized using " + modelName
    plot = bp.figure(plot_width=900, plot_height=700, title=title,
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)
    
    plot.scatter(x=matrix[:,0], y=matrix[:,1], 
        source=bp.ColumnDataSource({
        "entry": entry, 
        }))
    hover = plot.select(dict(type=HoverTool))
    hover.tooltips={"entry": "@entry"}
    show(plot)
    
#--------------------------------------------------------------

def plotTSNELDA(modelName,matrix,corpus,topicKeys,isKmeans=False):
    title="Tweets visualized using " + modelName
    plot = bp.figure(plot_width=900, plot_height=700, title=title,
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)
    if not isKmeans:
        plot.scatter(x=matrix[:,0], y=matrix[:,1], 
            color=colormap[topicKeys][:], 
            source=bp.ColumnDataSource({
            "corpus": corpus[:], 
            "topicKey": topicKeys[:]
            }))
        hover = plot.select(dict(type=HoverTool))
        hover.tooltips={"Tweet": "@corpus - topic: @topicKey"}
    else:
        plot.scatter(x=tsneKmeans[:,0], y=tsneKmeans[:,1],
            color=colormap[kmeansClusters][:], 
            source=bp.ColumnDataSource({
            "corpus": corpus[:], 
            "cluster": topicKeys[:]
             }))  
        hover = plot.select(dict(type=HoverTool))
        hover.tooltips={"Tweet": "@corpus - cluster: @cluster"}        
    show(plot)
      
#--------------------------------------------------------------  
def item(x):
    return item

if __name__=="__main__":
    tweetIds={}
    with open(tweetcleaned, 'r', encoding='utf-8') as f:
        #inputCorpus = f.readlines()
        for line in f:
            id,tweet=line.strip().split(',')
            if not tweet in tweetIds.keys():
                tweetIds[tweet]=id
                inputCorpus.append(tweet)

            
    print("corpus size: "+str(len(inputCorpus)))
    
    #--------------
    
    '''Tf-idf for getting word frequencies with respect to entire document'''
    
    tfidfvectorizer,tfidfMatrix,tfidfDict = computeTfIdfScores(inputCorpus)
    svdTfidf = reduceDimentionalityLSA(tfidfMatrix)
    tsneTfidf = reduceDimentionalityTSNE(svdTfidf) 
    plotTSNE('Tf-idf followed by LSA TSNE',tsneTfidf,inputCorpus[:],topicKeys=None)
    
    print("svdTFidf matrix shape")
    print(svdTfidf.shape)
               
    #--------------
    
    '''Kmeans clustering'''
    
    kmeansModel,kmeans,kmeansClusters,kmeansDistances = kmeansClustering(svdTfidf)#tfidfMatrix  
    tsneKmeans = reduceDimentionalityTSNE(kmeansDistances)
    
    kMeansLabelsUnique = np.unique(kmeansModel.labels_)
    clusterToTextsDict = getKmeansClustersTextDists()
    getClusterFeatures(tfidfvectorizer,kmeansModel)
    topTwoEvents=selectTopTwoEvents(clusterToTextsDict)
    extractTopNews(clusterToTextsDict,tfidfDict,topTwoEvents)
    plotTSNELDA('Kmeans TSNE',tsneKmeans,inputCorpus,kmeansClusters,True)
    #--------------
    
    '''LDA topic modeling'''

    cvectorizer,ldaModel,topicMatrix = LDATopicModeling() 
    #docTopic holds the probability association of each sentence and a certain topic
    docTopic = ldaModel.doc_topic_  
    topicKeys = []
    for i, q in enumerate(inputCorpus):
        topicKeys += [docTopic[i].argmax()]
    tsneLDA= reduceDimentionalityTSNE(topicMatrix)
    plotTSNELDA('LDA TSNE',tsneLDA,inputCorpus,topicKeys)
    getReleventWordstoTopics(cvectorizer,ldaModel,topicMatrix,10)
    
    #--------------
    
    wordsList=[TaggedDocument(to_unicode(str.encode(line)).split(),[to_unicode(str.encode(str(i)))]) for i, line in enumerate(inputCorpus)]      

    #--------------
    
    '''Doc2Vec: unsupervised learning for larger blocks of text'''
    
    doc2VecModel=buildD2VModel(wordsList)
    #doc2VecModel=Doc2Vec.load('./tweets.d2v')
    vocabs=[i for i in doc2VecModel.wv.index2word]
    tsneDoc2Vec = reduceDimentionalityTSNE(doc2VecModel.syn0)        
    plotTSNE('Doc2Vec TSNE',tsneDoc2Vec,vocabs)
    
    for i in range (0, doc2VecModel.syn0.shape[0]): 
        wordvector= doc2VecModel.syn0[i] 
        word=doc2VecModel.wv.index2word[i]
        #assert doc2VecModel.wv.index2word.index(word)==i
    
    vocabSize=len(doc2VecModel.vocab)
    print(vocabs)
    print("vocabSize")
    print(vocabSize)
     
    exampleWords=['budget','obama','trump','die','men','first']
    for example in exampleWords:
        exampleIndx=doc2VecModel.wv.index2word.index(example)
        exampleVec=doc2VecModel.syn0[exampleIndx]       
        resVextorForExample=doc2VecModel.similar_by_vector(exampleVec, topn=10, restrict_vocab=None)
        print("resVextorForExample")
        print(resVextorForExample)
        
    #--------------   
    
    '''Word2Vec'''
        
    word2VecModel=buildW2VModel(inputCorpus)
    #word2VecModel=Word2Vec.load('./tweets.d2v')
    bigramEnrichedVocabs=word2VecModel.vocab.keys()    
    vocabsW2V=[i for i in word2VecModel.wv.index2word]  
    tsneword2Vec = reduceDimentionalityTSNE(word2VecModel.syn0)  
    plotTSNE('Word2Vec TSNE',tsneword2Vec,vocabsW2V,topicKeys=None)
    
    exampleWords=['obama','president_obama']
    for example in exampleWords:
        exampleIndx=word2VecModel.wv.index2word.index(example)
        exampleVec=word2VecModel.syn0[exampleIndx]       
        resultVectorForExample=word2VecModel.similar_by_vector(exampleVec, topn=10, restrict_vocab=None)
        print("resultVectorForExample")
        print(resultVectorForExample)
        
    #-------------- 

