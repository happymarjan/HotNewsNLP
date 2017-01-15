#!/usr/bin/env python3.4

#Author: Marjan Alavi

import sys
import os.path
import inspect
from time import sleep
import tweepy
from tweepy.streaming import StreamListener
from beaker.middleware import SessionMiddleware
import logging
import json
import bottle as B
import sqlite3

#----------------------------------------------------------------------------

LOCALHOST = 'localhost'
CLIENT_PORT = 5000

#----------------------------------------------------------------------------

class DB():
    def __init__(self):
        self.db = sqlite3.connect('newsDB.db',check_same_thread = False)
        self.cursor = self.db.cursor()

    def disconnect(self):
        self.db.close()

    def descr(self):
        print("instance {0}".format(self))

#----------------------------------------------------------------------------

log = logging.getLogger("tweetd")
logging.basicConfig(level=logging.DEBUG,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
logging.getLogger("oauthlib").setLevel(logging.WARNING)
logging.getLogger("tweepy").setLevel(logging.WARNING)

#----------------------------------------------------------------------------

@B.route('/')
def getRequestToken():              
    redirect_url = None
    session = B.request.environ['beaker.session']
   
    log.debug('getRequestToken(): going through OAuth process')
    try:
        redirect_url = auth.get_authorization_url() 

    except tweepy.TweepError:
        log.error('getRequestToken(): failed to get request token.')
        disconnectDBConnection()
        sys.exit(1)

    session['request_token'] = (auth.request_token['oauth_token'],auth.request_token['oauth_token_secret'])
    session.save()

    if redirect_url is not None:
        B.redirect(redirect_url)
    else:
        B.redirect(callback_url)
        
#----------------------------------------------------------------------------

@B.route('/verify')
def getVerification():
    global access_token
    global access_token_secret
    global verifier
    if (access_token is "") or (access_token_secret is ""):
        session = B.request.environ['beaker.session']
        if verifier is "":
            if 'oauth_verifier' in B.request.GET:
                verifier = B.request.GET['oauth_verifier']
            else:
                log.error('getVerification(): unable to get oauth_verifier.')
                disconnectDBConnection()
                sys.exit(1)
                           
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

        if not 'request_token' in session:
            log.error('getVerification(): failed to get request token.')
            disconnectDBConnection()
            sys.exit(1)

        token = session['request_token']
        del session['request_token']

        auth.request_token={'oauth_token':token[0], 'oauth_token_secret':token[1]}

        try:
            auth.get_access_token(verifier)
        except tweepy.TweepError:
            log.error('getVerification(): failed to get access token.')
            disconnectDBConnection()
            sys.exit(1)

        access_token = auth.access_token
        access_token_secret = auth.access_token_secret

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    name = api.me().name
    screen_name = api.me().screen_name
    IdStr = api.me().id_str

    savedCreds['api'] = api
    savedCreds['access_token_key'] = access_token
    savedCreds['access_token_secret'] = access_token_secret
    savedCreds['name'] = name
    savedCreds['screen_name'] = screen_name
    savedCreds['Id'] = IdStr
    
    if not (all (k in savedCreds for k in ("api", "access_token_key","access_token_secret", "name","screen_name", "Id"))):
        log.error('getVerification(): savedCreds dictionary failed to get filled')
        disconnectDBConnection()
        sys.exit(1)  

    f = open('savedcredsFile.txt','w')
    f.write('access_token_key = ' + access_token + '\n' + 'access_token_secret = ' + access_token_secret + '\n' + 'screen_name = ' + screen_name + '\n' + 'Id =' + IdStr + '\n')
    f.close()

    log.debug('getVerification(): got access token for name = %s, screen_name = %s, id = %s',name,screen_name,IdStr)

    B.redirect(initialize_url)
    
#----------------------------------------------------------------------------

@B.route('/init')
def initialize():
    global strmlistener
    if (not 'screen_name' in savedCreds.keys()) or (not 'Id' in savedCreds.keys()):
        log.error('initialize(): could not access screen_name /Id in saved credentials dictionary')
        disconnectDBConnection()
        sys.exit(1)
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    log.debug('initialize(): setting up streaming api for fetching tweet updates & follower/friend updates in realtime')
    strmlistener =  streamInit(auth)
    strmlistener.startStream()
    B.redirect(start_url)
    
#----------------------------------------------------------------------------

@B.route('/start')
def start():
    if not 'api' in savedCreds.keys():
        log.error('start(): could not access created twitter interface info in saved credentials dictionary')
        disconnectDBConnection()
        disconnectStreamingConnection()
        sys.exit(1)
    api = savedCreds['api']
    
    if B.request.method == "GET":
        log.debug('start(): receiving GET request')
        fetchTimelineTweets()
    else:
        raise B.HTTPError(402)
    
#----------------------------------------------------------------------------

def fetchTimelineTweets():
    log.debug('in fetchTimelineTweets()...')
    api = savedCreds['api']
    
    for status in tweepy.Cursor(api.home_timeline, count = 200, monitor_rate_limit=True, wait_on_rate_limit=True, wait_on_rate_limit_notify=True).items():
    #c = tweepy.Cursor(api.home_timeline, count = 200, monitor_rate_limit=True, wait_on_rate_limit=True, wait_on_rate_limit_notify=True).items()
        try:
            #status=c.next()
            processStatus(status)
        except: 
            #tweepy.RateLimitError
            sleep(60 * 15)
            continue
    return

#----------------------------------------------------------------------------

def processStatus(status):
    if(type(status) == tweepy.models.Status):
        status = status._json
        
    id_str=str(status['id_str'])
    tweet = str(status['text'])
    ctime = str(status['created_at'])
    rtCnt = str(status['retweet_count'])
    saveNewsIntoDB(id_str,tweet,ctime,rtCnt)
    
    print(tweet)
    print(ctime)
    print(rtCnt)
    
    return

#----------------------------------------------------------------------------

def createNewsTbl():
    try: 
        cur.execute('CREATE TABLE IF NOT EXISTS newsTbl (id_str TEXT, tweet TEXT, ctime TEXT, rtCnt INTEGER)')
        con.commit()
    except sqlite3.Error as err:
        log.error('Error creating news: {0}'.format(err))
    return

#----------------------------------------------------------------------------

def saveNewsIntoDB(id_str, tweet, ctime,rtCnt):  
    params = id_str, tweet, ctime, rtCnt
    try:
        cur.execute('SELECT * FROM newsTbl WHERE id_str = ?',(id_str,))
        record = cur.fetchone()
        if record is None:
            cur.execute("INSERT INTO newsTbl (id_str, tweet, ctime,rtCnt) VALUES (?, ?, ?, ?)",params)
            con.commit()
            log.debug('saveNewsIntoDB(): Inserted new entry for tweet id {0} in newsTbl'.format(id_str))
            return
    
    except sqlite3.Error as err:
        log.error('Error updating newsTbl for tweet id {0}: {1}'.format(id_str, err))
    return

#----------------------------------------------------------------------------

def disconnectStreamingConnection():
    global strmlistener
    try:
        strmlistener.stopStream()
        log.debug('disconnectStreamingConnection(): disconnecting stream listener connection')
    except:
        pass
    return

#----------------------------------------------------------------------------

def disconnectDBConnection():
    global DbCon
    try:
        DbCon.disconnect()
        log.debug('disconnectDBConnection(): disconnecting database connection')
    except:
        pass
    return

#----------------------------------------------------------------------------

import atexit
atexit.register(disconnectDBConnection)
atexit.register(disconnectStreamingConnection)

#----------------------------------------------------------------------------

class stream_listener(StreamListener):
    def __init__(self):                 
        self.retry_time_start = 5.0
        self.retry_420_start = 60.0
        self.retry_time = self.retry_time_start  #5.0
        
    def on_data(self, data):
        try:
            log.debug('stream_listener(): received data from streaming API')
    
            payload = json.loads(data)

            if 'user' in payload.keys() and 'text' in payload.keys():
                processStatus(payload)
    
        except BaseException as e:
            log.debug('failed on: %s',str(e)) 
            sleep(self.retry_time_start)

    def on_error(self, status):
        log.debug('stream_listener(): streaming error: %s', status)
        #implemented exponential backoff as follows:
        if status == 420:
            self.retry_time = max(self.retry_420_start, self.retry_time) #60, 120, 240, 480,.. 
            sleep(self.retry_time) #60, 120, 240, 480,..
            self.retry_time = self.retry_time * 2 #120, 240, 480, ...

    def on_limit(self, status):
        log.debug('stream_listener(): streaming limit exceeded: %s', status)

    def on_timeout(self, status):
        log.debug('stream_listener(): streaming disconnected: %s', status)
        
#----------------------------------------------------------------------------

class streamInit():
    def __init__(self,auth):
        self.auth = auth
        self.is_streamCreated = 0
        self.is_userStreamCreated = 0
        self.retry_time_start = 60
        self.retry_time = self.retry_time_start
        self.retry_time_step = 30
        
    def startStream(self):
        st_lstner = stream_listener()
        
        while(not self.is_streamCreated):
            try:
                self.stream = tweepy.Stream(self.auth, st_lstner)
                self.is_streamCreated = 1
            except IOError as e:
                log.error('streamInit(): exception occured while establishing streaming api: {0}'.format(e))
                #incremental backoff in reconnection attempts
                self.retry_time += self.retry_time_step 
                sleep(self.retry_time)
                
        self.retry_time = self.retry_time_start  
        
        while(not self.is_userStreamCreated):
            try:
                self.listener = self.stream.userstream(_with='followings' , async=True)
                self.is_userStreamCreated = 1
            except IOError as e:
                log.debug('streamInit(): exception occured while establishing streaming api: {0}'.format(e))
                #incremental backoff in reconnection attempts
                self.retry_time += self.retry_time_step
                sleep(self.retry_time)
            
    def stopStream(self):
        self.stream.disconnect()
        
#----------------------------------------------------------------------------

if __name__ == "__main__": 
    savedCreds = dict()
    verifier= ""
    access_token = ""
    access_token_secret = ""
    start_url='http://{0}:{1}/start'.format(LOCALHOST, CLIENT_PORT)
    callback_url='http://{0}:{1}/verify'.format(LOCALHOST, CLIENT_PORT)
    initialize_url='http://{0}:{1}/init'.format(LOCALHOST, CLIENT_PORT)
    screen_name = ""
    global strmlistener

#-------------------

    log.info("Usage: \n Please fill the registered application credentials (consumer_key, consumer_secret) in the credfile.txt file" 
                " in the source directory.\nAfter that in a browser tab navigate to" 
                " http://{0}:{1}/\nAnd login to twitter with the twitter handler following some news websites.\n\n".format(LOCALHOST, CLIENT_PORT))

    currentPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    credfile = currentPath+'/credfile.txt'
    if(not os.path.isfile(credfile)):
        log.error('The file %s does not exist in the source directory!',credfile)
        sys.exit(1)
        
#-------------------

    tokenDict  = dict()
    with open(credfile) as fp:
        for line in fp:
            eq_index = line.find('=')
            varName = line[:eq_index].strip()
            value = line[eq_index + 1:].strip()
            tokenDict[varName] = value
            
    if ('consumer_key'  not in tokenDict.keys()) or ('consumer_secret'  not in tokenDict.keys()):
        log.error('the format of \'credfileName\' file is incorrect! provide consumer_key and consumer_secret lines')
        sys.exit(1)

    consumer_key = tokenDict.get('consumer_key').replace("'","")
    consumer_secret = tokenDict.get('consumer_secret').replace("'","")

#-------------------

    log.debug('creating an OAuthHandler instance')
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback_url)
    except Exception as e:
        log.error('creating OAuthHandler instance failed'.format(e))
        sys.exit(1)
        
#-------------------
   
    DbCon = DB()
    con = DbCon.db
    cur = DbCon.cursor
    createNewsTbl()

#-------------------

    session_options = {
        'session.type': 'memory',
        'session.data_dir': './data',
        'session.cookie_expires': 300,
        'session.auto': True
    }
    app = SessionMiddleware(B.app(), session_options)

    B.run(
      server = 'cherrypy',
      app = app,
      host = LOCALHOST,
      port = CLIENT_PORT,
      debug = True,
      reloader = False)
    
#----------------------------------------------------------------------------


