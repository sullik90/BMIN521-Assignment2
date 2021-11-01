'''
Created on Dec 29, 2019

@author: dweissen
'''
import configparser
import logging as log
import string
import sys
import nltk
from nltk.util import ngrams
from Handlers.CSVHandler import CSVHandler
from pandas.core.frame import DataFrame
from nltk.tokenize import TweetTokenizer
from pandas.core.series import Series
from nltk.text import TokenSearcher

from Evaluation.Evaluator import Evaluator

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class NaiveBayesFinder(object):
    """
    Define a binary classifier which discover tweets mentionning a drug using hand made features and a Naive Bayes
    """
    def __init__(self, config: configparser):
        self.config = config
        
    
    def __buildBigramsModel(self, trainingTweetsTokenized: Series, minFrequence = 1) -> dict:
        """
        :param trainingTweetsTokenized: all tweets from a training set tokenized
        :param minFrequence: the minimum frequence to keep a bigram, by default all are kept, i.e. minFrequence=1
        :return: a dictionary with the most frequent bigrams as key and the default value False to prepare next steps: the training and prediction steps
        :note: This is a lot of bigrams, we just filter the most unfrequent. Another choice could be to remove stop words (ex. the, of, a etc.) and the punctuations and then compute the bigrams.
        """
        #we create the bigrams of all tweets using ntlk function and store them in the list twBigrams
        twBigrams = []
        trainingTweetsTokenized.apply(lambda tweetTokenized: twBigrams.extend(ngrams(tweetTokenized, 2)))
        
        #we compute the frequence of each bigram (i.e. how many times a bigram appears in all tweets given)
        freq_dist = nltk.FreqDist(twBigrams)
        #we filter out less frequent bigrams using FreqDist and the threshold minFrequence
        bigramsModel = {}
        for bigram, frequence in freq_dist.items():
            if frequence >= minFrequence:
                bigramsModel[bigram] = False
            #else: the bigram is not frequent enough in the training set and discarded
        log.debug(f"{len(bigramsModel)} Bigrams will be kept in the model.")
        return bigramsModel

    def __buildUnigramsModel(self, trainingTweetsTokenized: Series, minFrequence=1) -> dict:
        """
                :param trainingTweetsTokenized: all tweets from a training set tokenized
                :param minFrequence: the minimum frequence to keep a unigram, by default all are kept, i.e. minFrequence=1
                :return: a dictionary with the most frequent unigrams as key and the default value False to prepare next steps: the training and prediction steps
                :note: This is a lot of unigrams, we just filter the most unfrequent. Another choice could be to remove stop words (ex. the, of, a etc.) and the punctuations and then compute the unigram.
                """

        # remove stop words (ex. the, of, a etc.)
        stop_words = set(stopwords.words('english'))
        word_tokens=word_tokenize(trainingTweetsTokenized)
        filtered_tweets = []
        filtered_tweets = [w for w in word_tokens if not w.lower() in stop_words]
        for w in word_tokens:
            if w not in stop_words:
                filtered_tweets.append(w)

        print(word_tokens)
        print(filtered_tweets)

        #remove punctuation
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        filtered_tweets.remove(punctuation)


        # we create the Unigrams of all tweets using ntlk function and store them in the list Unigrams
        Unigrams = []
        trainingTweetsTokenized.apply(lambda tweetTokenized: Unigrams.extend(ngrams(tweetTokenized, 1)))


    def __getF_Bigrams(self, counter: list, totalTweets:int, bigramsModels: dict, tweet: Series):
        """
        :param counter: a simple counter to keep track of the progress
        :param totalTweets: the total number of tweets to process to keep track of the progress
        :param bigramsModels: the formated dictionary of bigram that we keep in the model: keys are the selected bigrams, all values are set by default to false
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing an existing dictionary possibly empty
        :return: compute all bigrams occurring in the tweet and search for those bigrams in bigramsModels, the set of bigrams kept as features
        If a bigrams occuring in a tweet is found in bigramsModels it is set to True, otherwise nothing is done 
        tweetFeatures is updated with all bigrams of bigramsModels and their respective values True if the bigram occurs in the tweet false otherwise
        """
        #just to display the progress since the computation can be long
        counter[0] = counter[0] + 1
        if (totalTweets==counter[0]) or ((counter[0]%500)==0):
            log.debug(f"Bigrams have been computed for {counter[0]} on {totalTweets} tweets")
        #get the bigrams occurring in the tweet
        bigramsInTweet = ngrams(tweet['txt_tokenized'], 2)
        #bigramModels is copied as the bigrams and their values will be used for classifying this tweet
        cbgFeatures = bigramsModels.copy()
        # checking if the bigrams occurring in the tweets have been seen in the training set, that is are in the bigramsModel
        for bigram in bigramsInTweet:
            if bigram in cbgFeatures:
                # one bigram has been found so I mark it
                cbgFeatures[bigram]=True
            #else: this is a new bigram, unseen in the training set, we can't use it as we don't have any label associated so we ignore it
            
        #we have all bigrams updated, False if they do not appear in the tweet, true if they do
        #we add them in the list of features describing the tweet 
        tweet['tweetFeatures'].update(cbgFeatures)
        

    def __getF_DrugSuffixes(self, tweet: Series):
        """
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing a dictionary 
        :return: compute if a token ends with a suffix most likely associated with drug names, 
        the dictionary in the cell tweetFeatures is updated with the feature Has_Token_With_Suffix: true if a token is found, false otherwise
        """
        #by default no token finish with the suffixes
        suffixFound = False
        #Then we search token by token. The list of suffixes has to be completed...
        if TokenSearcher(tweet['txt_tokenized']).findall(r'<.*(ine|aine|xin|idol|sil|en|ive|in)>'):
            log.debug(f"=> found a token finishing by ine|aine|xin|idol|sil|en|ive|in) in tweet: {tweet['text']}")
            suffixFound = True
        #and we add the feature with the corresponding value given as boolean
        tweet['tweetFeatures'].update({'Has_Token_With_Suffix': suffixFound})


    def __computeFeatures(self, tweets: DataFrame):
        """
        :param tweets: the list of tweets to compute the features
        :return: the same DataFrame with a new column which contains a dictionary with feature names as key and their values as values
        """
        #instantiate the column with an empty dict which will contain the features describing the tweets
        tweets['tweetFeatures'] = tweets.apply(lambda x: {}, axis=1)
        #apply each function computing the features describing the examples
        counter = [0] # to display the progress
        tweets.apply(lambda row: self.__getF_Bigrams(counter, len(tweets), self.bigramsModel, row), axis=1)
        tweets.apply(lambda row: self.__getF_DrugSuffixes(row), axis=1)
        

    def train(self, trainingTweets: DataFrame):
        """
        :param: trainingTweets, a DataFrame containing a column label and a column text
        :return: add self.classifier an attribute to the object, a reference to the classifier fully trained and ready to be used for classification
        """
        assert 'text' in trainingTweets.columns, f'I was expecting the column text in the dataframe representing the training tweets but it is missing, check the data.'
        # start lower casing and tokenizing the tweets
        twtTkz = TweetTokenizer()
        trainingTweets['txt_tokenized'] = trainingTweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        #build the bigrams model with a threshold of 3 or more occurrences for the bigrams
        self.bigramsModel = self.__buildBigramsModel(trainingTweets['txt_tokenized'], 3)
        
        log.debug(f"Start computing the features for the {len(trainingTweets)} training tweets...")
        self.__computeFeatures(trainingTweets)
        assert 'tweetFeatures' in trainingTweets.columns, f'I was expecting the column tweetFeatures in the dataframe representing the training tweets after computing the features but it is missing, check the code.'
        
        log.debug(f"All features have been computed for the {len(trainingTweets)} tweets, start training the Naive Bayes...")
        #creating the list of tuples: ({feature_names: features_values}, label) to train a classifier
        examples = list(zip(trainingTweets['tweetFeatures'], trainingTweets['label']))
        self.classifier = nltk.NaiveBayesClassifier.train(examples)
        # you can try another machine learning algorithm, decision tree, for better results but it will take longer than the naive bayes
        #self.classifier = nltk.DecisionTreeClassifier.train(examples)


    def classify(self, unknownTweets: list) -> DataFrame:
        """
        :param unknownTweets, a DataFrame containing a column label and a column text
        :return: a copy of unknownTweets with a prediction column
        """
        assert 'text' in unknownTweets.columns, f'I was expecting the column text in the dataframe representing the new tweets to classify but it is missing, check the data.'
        # Similar preprocessing than in the training function: start lower casing and tokenizing the tweets
        twtTkz = TweetTokenizer()
        unknownTweets['txt_tokenized'] = unknownTweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        
        log.debug(f"Start computing the features for the {len(unknownTweets)} unknown tweets...")
        self.__computeFeatures(unknownTweets)
        assert 'tweetFeatures' in unknownTweets.columns, f'I was expecting the column tweetFeatures in the dataframe representing the new tweets to classify after computing their features but it is missing, check the code.'
        log.debug(f"All features have been computed for the {len(unknownTweets)} tweets, start classifying them...")
        #apply the classifier on unseen examples
        def __classify(tweet: Series):
            """
            :param tweet: a row representing the tweet to classify, the row should have a cell 'tweetFeatures' 
            """
            prediction = self.classifier.classify(tweet['tweetFeatures'])
            return prediction
        unknownTweets['prediction'] = unknownTweets.apply(lambda row: __classify(row), axis=1)
        
        return unknownTweets


if __name__ == '__main__':
    config = configparser.ConfigParser()
    #change with the path to your own configuration file
    config.read('C:\\Users\\Katie\\Documents\\DrugInTweetsLab2 (1)\\DrugInTweetsLab2\\config.properties.tmp')
    logFile = config['Logs']['logFile']
    log.basicConfig(level=log.DEBUG, 
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[log.StreamHandler(sys.stdout), log.FileHandler(logFile, 'w', 'utf-8')])
    
    
    log.info("Creating the classifier...")
    # Here we continue to use dataframe to represent the tweets since it is convenient.
    # An alternative would have been to create a nltk.corpus similar to twitter_samples
    # As an example, see the following blog for a task of sentiment classification on tweets: https://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/
    hdl = CSVHandler(config)
    dtf = NaiveBayesFinder(config)
    log.info("Classifier created.")
    
    log.info("Start training the Classifier...")
    dtf.train(hdl.getTraining())
    log.info("Classifier trained.")
    
    log.info("Start predicting on new examples...")
    tweetsClassified = dtf.classify(hdl.getValidation())
    tweetsClassified.to_csv(config['Data']['predictionsOutPath'], sep='\t')
    log.info("Prediction made.")
    
    evaluator = Evaluator()
    cf, prec, rec, f1 = evaluator.evaluate(tweetsClassified)
    log.info("Evaluation of the classifier on the validation set:")
    log.info(f"Confusion matrix:\n{cf}")
    log.info(f"performance: {round(prec, 4)} precision, {round(rec, 4)} recall, {round(f1, 4)} F1")
    