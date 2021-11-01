'''
Created on Dec 28, 2019

@author: dweissen
'''
import configparser
import logging as log
import sys
import re
import pandas as pd
from pandas.core.frame import DataFrame
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.text import TokenSearcher

from Evaluation.Evaluator import Evaluator
from Handlers.CSVHandler import CSVHandler
from pandas.core.series import Series


class RegExFinder(object):
    """
    Define a binary classifier which discovers tweets mentioning drug names by using manually defined regular expressions
    """

    def __init__(self, config: configparser):
        '''
        Constructor
        '''
        self.config = config
        #the list of REs to search for, all REs should be in lower case following the same tokenization rules than TweetTokenizer
        self.REs = [
            r'<prescribed> <with>',
            r'<doctor> <switched>',
            r'<had> <me> <feeling> <like>',
            r'<got> <me> <feeling>',
            r'<took> <.*>* <.*(ine|aine|xin|idol|sil)>'
            ]
        
    
    def searchDrugsByREs(self, tweets: DataFrame):
        """
        :param tweets: a dataframe representing the tweets to classify
        :return: the dataframe with 2 additional columns: 
        prediction, 0 if the tweet does not mention a drug, 1 otherwise; 
        pattern_matching: empty if no RE matched, other wise the list of the REs which recognized a sequence in the tweet
        """
        # start lower casing and tokenizing the tweets
        twtTkz = TweetTokenizer()
        tweets['txt_tokenized'] = tweets['text'].apply(str.lower).apply(twtTkz.tokenize).apply(nltk.Text)
        #a dic to remember which pattern matched
        tweetsMatching = {}
        #search for all REs in the tweets
        tweets.apply(lambda row: self.__applyREs(row, 'txt_tokenized', tweetsMatching), axis=1)
        
        #now that I have all tweets matched by a REs I add the label in the tweet dataframe
        # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['pattern_matching'] = ''
        for tweetID, patterns in tweetsMatching.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'pattern_matching'] = ' --- '.join(patterns)
        #remove the column nltkText which is now useless
        tweets.drop(['txt_tokenized'], axis=1, inplace=True)
        # return the dataframe
        return tweets
    
    
    def __applyREs(self, tweet: Series, columnToApplyREs: str, tweetsMatching: dict):
        """
        :param tweet: the tweet where the REs will be searched for, the tweet should contain a cell txt_tokenized
        :param columnToApplyREs: the name of the column to apply the REs
        :param tweetsMatching: a dictionary containing the tweets matching and the REs found in the tweets 
        :return: just an updated tweetsMatching dictionary
        """
        for rex in self.REs:
            if TokenSearcher(tweet[columnToApplyREs]).findall(rex):
                log.debug(f"\t-> the RE: [{rex}] found in {tweet['text']}")
                if tweet.name in tweetsMatching:
                    tweetsMatching[tweet.name].append(rex)
                else:
                    tweetsMatching[tweet.name] = [rex]
                    
                
    def searchDrugsByREsWithDrugTags(self, tweets: DataFrame):
        """
        :param tweets: a dataframe representing the tweets to classify
        :return: the dataframe with 2 additional columns: 
        prediction, 0 if the tweet does not mention a drug, 1 otherwise; 
        pattern_matching: empty if no RE matched, other wise the list of the REs which recognized a sequence in the tweet
        """
        log.warning("This function has to be implemented...")
#----------------------------------------------------------------------------------------------------------------------
#--------------------------------------- To be deleted for the students version ---------------------------------------
#----------------------------------------------------------------------------------------------------------------------
        # Just put all elements of the code in the same section 
        from Pipelines.LexiconFinder import LexiconFinder
        # Create the lexicon which will be used to replace the drug names by the tag
        lf = LexiconFinder(self.config)
        tweetsTagged = lf.getTweetsWithGenericDrugTag(tweets)
        
        #redefined the set of REs this time, taking into account the tag
        self.REs = [
            r'<prescribed> <.*>* <__@DRUG@__>',
            r'<doctor> <switched> <__@DRUG@__>',
            r'<__@DRUG@__> <.*>* <had> <me> <feeling> <like>',
            r'<__@DRUG@__> <.*>* <got> <me> <feeling>',
            r'<took> <.*>* <__@DRUG@__>'
            ]
        
        #a dic to remember which pattern matched
        tweetsMatching = {}
        #search for all REs in the tweets
        tweetsTagged.apply(lambda tweet: self.__applyREs(tweet, 'txt_tagged', tweetsMatching), axis=1)
        
        #now that I have all tweets matched by a REs I add the label in the tweet dataframe
        # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['pattern_matching'] = ''
        for tweetID, patterns in tweetsMatching.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'pattern_matching'] = ' --- '.join(patterns)
        #remove the column nltkText which is now useless
        tweets.drop(['txt_tokenized'], axis=1, inplace=True)
        # return the dataframe
        return tweets
#--------------------------------------- To be deleted for the students version ---------------------------------------
        
        
if __name__ == '__main__':
    config = configparser.ConfigParser()
    #change with the path to your own configuration file
    config.read('C:\\Users\\dweissen\\git\\drugintweetslab1\\DrugInTweetsLab\\config.properties')
    logFile = config['Logs']['logFile']
    log.basicConfig(level=log.DEBUG, 
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[log.StreamHandler(sys.stdout), log.FileHandler(logFile, 'w', 'utf-8')])
    
    log.info("Creating the classifier...")
    hdl = CSVHandler(config)
    ref = RegExFinder(config)
    log.info("Classifier created")
    
    log.info(f"Start applying the {len(ref.REs)} REs on the validation set...")
    
    # search tweets mentioning drugs using REs without drugs discovered by our lexicon being mentioned
    tweetsAnnotated = ref.searchDrugsByREs(hdl.getValidation())
    # Once the method has been implemented, comment the call of ref.searchDrugsByREs and uncomment ref.searchDrugsByREsWithDrugTags 
    #tweetsAnnotated = ref.searchDrugsByREsWithDrugTags(hdl.getValidation())
    log.info(f"REs applied on the validation set.")
    
    tweetsAnnotated.to_csv(f"{config['Data']['regexOutputPath']}\\drugREsFinder.tsv", sep='\t')
    evaluator = Evaluator()
    cf, prec, rec, f1 = evaluator.evaluate(tweetsAnnotated)
    log.info("Evaluation of the classifier on the validation set:")
    log.info(f"Confusion matrix: {cf}")
    log.info(f"performance: {round(prec, 4)} precision, {round(rec, 4)} recall, {round(f1, 4)} F1")    
