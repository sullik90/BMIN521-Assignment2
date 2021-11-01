'''
Created on Dec 19, 2019

@author: dweissen
'''
from deprecated import deprecated
import sys
import configparser
import pandas as pd
import logging as log
import os
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.core import series
from nltk.tokenize import TweetTokenizer

from Evaluation.Evaluator import Evaluator
from Handlers.CSVHandler import CSVHandler


class LexiconFinder(object):
    """
    Define a binary classifier which discover tweets mentionning a drug using a lexicon of drug names
    """
    
    def __init__(self, config: configparser):
        self.config = config
        self.__readDrugLexicon()
        

    def __readDrugLexicon(self):
        """
        Read the lexicon given in the config file, remove all duplicates if any, and keep the lexicon in memory
        The lexicon was downloaded from https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files, 12/20/2019 
        """
        # reading the file downloaded...
        lexPath = self.config['Data']['lexiconPath']
        self.lexicon = pd.read_csv(lexPath, sep='\t', encoding='utf8')
        # remove other columns as they are not useful
        self.lexicon.drop(['ApplNo','ProductNo','Form','Strength','ReferenceDrug','ActiveIngredient','ReferenceStandard'], axis=1, inplace=True)
        #removing the duplicates, caused by the different dosages
        self.lexicon.drop_duplicates(subset=['DrugName'], keep='first', inplace=True)
        #sanity check
        assert(len(self.lexicon)==7354), f"I was expecting 7354 unique drug names. I got {len(self.lexicon)}..."
        #just sort the drug names
        self.lexicon.sort_values(by=['DrugName'], inplace=True)
        # we are converting the drug name in lower case for further processing
        self.lexicon['DrugName'] = [str.lower(drug) for drug in self.lexicon['DrugName']]


    def naiveSearchDrugsFromLexiconInTweets(self, tweets: DataFrame):
        """
        :param tweets: the set of tweets to search for as a DataFrame with tweetID as index and a column text containing the texts of the tweets
        Apply the lexicon on the text column of the dataframe given
        Note: there are much faster ways of applying a lexicon on a DataFrame, but we keep the code simple for this lab.
        """
        #preprocessing the tweets, here we just lower case them
        tweets['text_lower'] = [str.lower(text) for text in tweets['text']]
        
        # a dictionary recoding the matches: index of the tweet => [list of drug mentioned]
        #ex: {669368304801198080: [arnica, hemp seed oil]}
        tweetsMentioningDrugs = {}
        
        #we search each drug name occurring in the lexicon in the tweets
        for drug in self.lexicon['DrugName']:            
            log.debug(f"=> Search tweets mentioning the drug: [{drug}]")
            # a simple code for searching the drug name in tweets using iterrows, easy to read but very slow...
            # it is equivalent to __naiveSearchForDrug
            #self.naiveSearchForDrugIterrows(drug, tweets)
            matches = self.__naiveSearchForDrug(drug, tweets)
            if matches is not None:
                #some tweets have been found containing the drug name, 
                #we append the drug name to any existing drug names already found mentioned in those tweets
                self.__appenMatches(tweetsMentioningDrugs, matches, drug)
                log.debug(f'\t-> Found {len(matches)} tweets mentioning the drug')
        
        #now that we have all tweets mentioning a name of drug we add the label in the tweets dataframe
        # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['drugs_predicted'] = ''
        for tweetID, drugs in tweetsMentioningDrugs.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'drugs_predicted'] = '; '.join(drugs)
        #remove the useless column text_lower
        tweets.drop(['text_lower'], axis=1, inplace=True)
        # return the dataframe
        return tweets


    @deprecated(version='0.0', reason='This method is easy to read and write but it is very slow, __naiveSearchForDrug should be used')
    def naiveSearchForDrugIterrows(self, drug: str, tweets:DataFrame):
        """
        :param drug: a drug name to search in the tweets
        :param tweets: a dataframe of tweets
        :return: the indexes of the tweets mentioning the drug name  
        """
        indexes = []
        for index, row in tweets.iterrows():
            #search for the drug name in the tweet
            if drug in row['text_lower']:
                log.info(f"I found {drug} in tweet: {row['text']} (Tweet ID:{index})")
                indexes.append(index)
            #else: nothing to do
        return indexes


    def __naiveSearchForDrug(self, drug: str, tweets:DataFrame) -> DataFrame:
        '''
        :param drug: search for the drug given in the list of tweets
        :param tweets: the list of tweets to search into
        :return: the subset of tweets mentioning the drug name as a dataframe, none if no tweets matches
        '''
        # we use comprehensive list to speed up the search
        matches = tweets[[drug in text for text in tweets['text_lower']]]
        if len(matches)>0:
            log.info(f"I found {len(matches)} tweets mentioning the drug {drug}")
            return matches
        else:
            return None
        

    def __appenMatches(self, tweetsMentioningDrugs: dict, matches: DataFrame, drug: str):
        '''
        :param tweetsMentioningDrugs: the tweets mentioning drugs previously found 
        :param matches: the tweets mentioning the new drug name
        :param drug: the drug name mentioned in the tweets
        Add the new tweets in the dictionary tweetsMentioningDrugs
        '''
        for tweetID in list(matches.index):
            if tweetID in tweetsMentioningDrugs:
                tweetsMentioningDrugs[tweetID].append(drug)
            else:
                tweetsMentioningDrugs[tweetID] = [drug] 


    def searchForDrugsInTweets(self, tweets: DataFrame) -> DataFrame:
        """
        :param tweets: a dataframe representing the tweets to classify
        :return: the dataframe with 2 additional columns, prediction: 0 if the tweet does not mention a drug, 1 otherwise; drug predicted: the list of names of drugs mentioned, tokenized, None if no drug name was found 
        """
        # start lower casing the tweets and tokenizing the tweets and the lexicon entries
        twtTkz = TweetTokenizer()
        tweets['txt_tokenized'] = tweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        self.lexicon['drug_tokenized'] = self.lexicon['DrugName'].apply(twtTkz.tokenize)
        tweetsMentioningDrugs = {}
        #just a counter to display the progress
        count = [0]
        #search for the drugs names in the tweets, the results of the searches are contained in the dict tweetsMentioningDrugs
        self.lexicon.apply(lambda row: self.__findDrugsInTweets(row, tweets, tweetsMentioningDrugs, count), axis=1)

        #now that we have all tweets mentioning a name of drug we add the label in the tweets dataframe
        # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['drugs_predicted'] = None
        for tweetID, drugs in tweetsMentioningDrugs.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'drugs_predicted'] = str(drugs)
        #remove the column txt_tokenized which is now useless
        tweets.drop(['txt_tokenized'], axis=1, inplace=True)
        # return the dataframe
        return tweets
    
    def __findDrugsInTweets(self, drug: series, tweets: DataFrame, tweetsMentioningDrugs: dict, count: list):
        """
        :param drug: the series representing the drug to search in the tweets
        :param tweets: the tweets to search into
        :param tweetsMentioningDrugs: a dictionary to return with the tweets found, tweetID -> drug names ; separated
        :param count: a counter counting the number of drug names already process, for displaying the progress 
        """
        # to display the progress of the computation            
        count[0] = count[0] + 1
        log.debug(f"=> Search tweets mentioning the drug: [{drug['DrugName']}] (drug #{count[0]})")
        
        #search for the drug name into the tweets
        tweets.apply(lambda row: self.__findDrugInTweets(row, drug['DrugName'], drug['drug_tokenized'], tweetsMentioningDrugs), axis=1)
        
    
    def __findDrugInTweets(self, tweet: series, drugName: str, drug: list, tweetsMentioningDrugs: dict):
        """
        :param tweet: a series representing one tweet
        :param drugName: a drug name to search into the tweet
        :param drug: the drug name tokenized
        :param tweetsMentioningDrugs: a dictionary to return with the tweets found, tweetID -> drug names ; separated        
        """
        if set(drug).issubset(set(tweet['txt_tokenized'])):
            # a drug name matches, need to check if all tokens of the drug appear in the right order in the tweet
            # that is we found the tokens <cortisone> <acetate> composing the drug name 'cortisone acetate' occurring in the tweet
            # we need to check if the tokens occur in the right order without anything between them in the tweet
            # for example in the tweet: <it> <has> <been> <found> <that> <cortisone> <acetate> <can> <help>, this is the case
            #             in the tweet: <i> <have> <tried> <acetate> <but> <not> <cortisone> <yet>, this is not the case
            if len(drug)>1:
                ind = 0
                while ind < len(tweet['txt_tokenized']):
                    tk = tweet['txt_tokenized'][ind]
                    # the token is not the start of a drug name, nothing to do
                    if tk != drug[0]:
                        ind = ind+1
                    else:
                        # this token is the first token of the drug name, we need to check if the next tokens are in the drug name
                        isDrugName = True
                        posInTweet = ind
                        for drugTk in drug:
                            if posInTweet < len(tweet['txt_tokenized']) and drugTk == tweet['txt_tokenized'][posInTweet]:
                                posInTweet = posInTweet+1
                            else:
                                isDrugName = False
                                break
                        if not isDrugName:
                            # it was the same token that the token starting a drug name but it is not the drug name
                            log.debug(f'\t-> Found all tokens of the drug {drugName} but not in the order expected in the following tweet: {tweet.text}')
                            # nothing to do
                            ind = ind+1
                        else:
                            # it is an occurrence of the drug name we need to subsitute it and move to the next position
                            log.debug(f'\t-> Found tweet {tweet.name} mentioning the drug: {drugName}')
                            if tweet.name not in tweetsMentioningDrugs:
                                tweetsMentioningDrugs[tweet.name] = []
                            #else: nothing to do
                            tweetsMentioningDrugs[tweet.name].append(drug)
                            # we found the first occurrence of the drug name, no need to search for the others
                            break
            #there is only one token
            else:
                log.debug(f'\t-> Found tweet {tweet.name} mentioning the drug: {drugName}')
                if tweet.name not in tweetsMentioningDrugs:
                    tweetsMentioningDrugs[tweet.name] = []
                tweetsMentioningDrugs[tweet.name].append(drug)
        #else: do nothing
        
                
    def getTweetsWithGenericDrugTag(self, tweets: DataFrame) -> DataFrame:
        """
        :param tweets: : a dataframe representing the tweets where drug names will be replaced by the generic tag __@DRUG@__
        :return: the same dataframe with two more column txt_tokenized with the text lower cased and tokenized, and 
        txt_tagged, where the original texts of the tweets are lower case and 
        where all occurrences of a drug name in our lexicon, found in the tweets have been replaced by the generic tag __@DRUG@__
        """
        # start lower casing the tweets and tokenizing the tweets and the lexicon entries
        twtTkz = TweetTokenizer()
        tweets['txt_tokenized'] = tweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        self.lexicon['drug_tokenized'] = self.lexicon['DrugName'].apply(twtTkz.tokenize)
        tweetsMentioningDrugs = {}
        # just a counter to display the progress
        count = [0]
        # search for the drugs names in the tweets, the results of the searches are contained in the dict tweetsMentioningDrugs
        self.lexicon.apply(lambda row: self.__findDrugsInTweets(row, tweets, tweetsMentioningDrugs, count), axis=1)        
        tweets['txt_tagged'] = tweets.apply(lambda row: self.__replaceDrugNameByTag(row, tweetsMentioningDrugs), axis=1)
        return tweets


    def __replaceDrugNameByTag(self, tweet: series, tweetsMentioningDrugs: dict):
        if tweet.name in tweetsMentioningDrugs:
            # the lexicon detected at least on drug name in the tweet, need to substitute it by the generic tag
            # the list can have nested drug names such as:
            # [['advil', 'pm'], ['advil']] extracted from the tweet: [i'm, out, of, advil, pm, .
            # so we sort the drug names by length and substitute in order the longest one and then the smaller one
            # this allow to correctly process tweets such as: "what is the difference between advil and advil pm?" 
            # which should be transform first as "what is the difference between advil and __@DRUG@__?" and then as what is the difference between __@DRUG@__ and __@DRUG@__? 
            drugNamesSorted = sorted(tweetsMentioningDrugs[tweet.name], key=len, reverse=True)
            txt_tagged = tweet['txt_tokenized']
            for dns in drugNamesSorted:
                txt_tagged = self.__substituteTagInTweet(dns, txt_tagged)
            return txt_tagged
        else:
            # the tweet did not contain a drug name or lexiconFinder did not find it, nothing to do we just return the tweet lower cased and tokenized 
            return tweet['txt_tokenized']
        
            
    def __substituteTagInTweet(self, drugName: list, tweet_tokenized: list) -> list:
        """
        :param drugName: the drug name tokenized to substitute in the tweet
        :param tweet_tokenized: the tweet tokenized where the drug name occurs and should be substituted, multiple occurrence are possible
        """
        tweet_tagged = []
        ind = 0
        while ind < len(tweet_tokenized):
            tk = tweet_tokenized[ind]
            # the token is not the start of a drug name, nothing to replace
            if tk != drugName[0]:
                tweet_tagged.append(tk)
                ind = ind+1
            else:
                # this token is the first token of the drug name, we need to check if the next tokens are in the drug name
                isDrugName = True
                posInTweet = ind
                for drugTk in drugName:
                    if posInTweet < len(tweet_tokenized) and drugTk == tweet_tokenized[posInTweet]:
                        posInTweet = posInTweet+1
                    else:
                        isDrugName = False
                        break
                if not isDrugName:
                    # it was the same token that the token starting a drug name but it is not the drug name
                    tweet_tagged.append(tk)
                    ind = ind+1
                else:
                    # it is an occurrence of the drug name we need to subsitute it and move to the next position
                    tweet_tagged.append('__@DRUG@__')
                    ind = ind + len(drugName)
                    
        return tweet_tagged

    
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
    lf = LexiconFinder(config)
    log.info("Classifier created")
    
    #First we will try with a exact string matching search
    tweetsAnnotated = lf.naiveSearchDrugsFromLexiconInTweets(hdl.getValidation())
    tweetsAnnotated.to_csv(f"{config['Data']['lexiconOutputPath']}\\drugExactMatch.tsv", sep='\t')
    evaluator = Evaluator()
    cf, prec, rec, f1 = evaluator.evaluate(tweetsAnnotated)
    log.info("Evaluation of the classifier on the validation set:")
    log.info(f"Confusion matrix: {cf}")
    log.info(f"performance: {round(prec, 4)} precision, {round(rec, 4)} recall, {round(f1, 4)} F1")
    
    #Second we are using a Twitter-aware tokenizer to perform the string matching
    tweetsAnnotated = lf.searchForDrugsInTweets(hdl.getValidation())
    tweetsAnnotated.to_csv(f"{config['Data']['lexiconOutputPath']}\\drugTokenMatch.tsv", sep='\t')
    evaluator = Evaluator()
    cf, prec, rec, f1 = evaluator.evaluate(tweetsAnnotated)
    log.info("Evaluation of the classifier on the validation set:")
    log.info(f"Confusion matrix: {cf}")
    log.info(f"performance: {round(prec, 4)} precision, {round(rec, 4)} recall, {round(f1, 4)} F1")
    