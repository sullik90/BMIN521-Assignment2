import configparser
import logging as log
import string
import sys
import nltk

nltk.download("stopwords")
from nltk.util import ngrams
from Handlers.CSVHandler import CSVHandler
from pandas.core.frame import DataFrame
from nltk.tokenize import TweetTokenizer
from pandas.core.series import Series
from nltk.text import TokenSearcher

from Evaluation.Evaluator import Evaluator

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Pipelines.LexiconFinder import LexiconFinder


class NaiveBayesFinder(object):
    """
    Define a binary classifier which discover tweets mentionning a drug using hand made features and a Naive Bayes
    """

    def __init__(self, config: configparser):
        self.config = config

    def __buildBigramsModel(self, trainingTweetsTokenized: Series, minFrequence=1) -> dict:
        """
        :param trainingTweetsTokenized: all tweets from a training set tokenized
        :param minFrequence: the minimum frequence to keep a bigram, by default all are kept, i.e. minFrequence=1
        :return: a dictionary with the most frequent bigrams as key and the default value False to prepare next steps: the training and prediction steps
        :note: This is a lot of bigrams, we just filter the most unfrequent. Another choice could be to remove stop words (ex. the, of, a etc.) and the punctuations and then compute the bigrams.
        """
        # we create the bigrams of all tweets using ntlk function and store them in the list twBigrams
        twBigrams = []
        trainingTweetsTokenized.apply(lambda tweetTokenized: twBigrams.extend(ngrams(tweetTokenized, 2)))

        # we compute the frequence of each bigram (i.e. how many times a bigram appears in all tweets given)
        freq_dist = nltk.FreqDist(twBigrams)
        # we filter out less frequent bigrams using FreqDist and the threshold minFrequence
        bigramsModel = {}
        for bigram, frequence in freq_dist.items():
            if frequence >= minFrequence:
                bigramsModel[bigram] = False
            # else: the bigram is not frequent enough in the training set and discarded
        log.debug(f"{len(bigramsModel)} Bigrams will be kept in the model.")
        return bigramsModel

    def __getF_Bigrams(self, counter: list, totalTweets: int, bigramsModels: dict, tweet: Series):
        """
        :param counter: a simple counter to keep track of the progress
        :param totalTweets: the total number of tweets to process to keep track of the progress
        :param bigramsModels: the formated dictionary of bigram that we keep in the model: keys are the selected bigrams, all values are set by default to false
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing an existing dictionary possibly empty
        :return: compute all bigrams occurring in the tweet and search for those bigrams in bigramsModels, the set of bigrams kept as features
        If a bigrams occuring in a tweet is found in bigramsModels it is set to True, otherwise nothing is done
        tweetFeatures is updated with all bigrams of bigramsModels and their respective values True if the bigram occurs in the tweet false otherwise
        """
        # just to display the progress since the computation can be long
        counter[0] = counter[0] + 1
        if (totalTweets == counter[0]) or ((counter[0] % 500) == 0):
            log.debug(f"Bigrams have been computed for {counter[0]} on {totalTweets} tweets")
        # get the bigrams occurring in the tweet
        bigramsInTweet = ngrams(tweet['txt_tokenized'], 2)
        # bigramModels is copied as the bigrams and their values will be used for classifying this tweet
        cbgFeatures = bigramsModels.copy()
        # checking if the bigrams occurring in the tweets have been seen in the training set, that is are in the bigramsModel
        for bigram in bigramsInTweet:
            if bigram in cbgFeatures:
                # one bigram has been found so I mark it
                cbgFeatures[bigram] = True
            # else: this is a new bigram, unseen in the training set, we can't use it as we don't have any label associated so we ignore it

        # we have all bigrams updated, False if they do not appear in the tweet, true if they do
        # we add them in the list of features describing the tweet
        tweet['tweetFeatures'].update(cbgFeatures)

    def __buildUnigramsModel(self, trainingTweetsTokenized: Series, minFrequence=1) -> dict:
        """
        :param trainingTweetsTokenized: all tweets from a training set tokenized
        :param minFrequence: the minimum frequence to keep a unigram, by default all are kept, i.e. minFrequence=1
        :return: a dictionary with the most frequent bigrams as key and the default value False to prepare next steps: the training and prediction steps
        :note: This is a lot of unigrams, we just filter the most unfrequent. Another choice could be to remove stop words (ex. the, of, a etc.) and the punctuations and then compute the bigrams.
        """
        # we create the unigrams of all tweets using ntlk function and store them in the list unigrams
        Unigrams = []
        trainingTweetsTokenized.apply(lambda tweetTokenized: Unigrams.extend(ngrams(tweetTokenized, 1)))

        # remove stop words from unigram (ex. the, of, a etc.)
        stop_words = set(stopwords.words('english'))
        # word_tokens = word_tokenize(trainingTweetsTokenized)
        filtered_unigrams = [w[0] for w in Unigrams if not w[0].lower() in stop_words]
        for w in Unigrams:
            if w not in stop_words:
                filtered_unigrams.append(w)

        print(filtered_unigrams)

        # remove punctuation from unigram
        punctuation = string.punctuation
        print(punctuation)
        while (punctuation in filtered_unigrams):
            test_list.remove(punctuation)

        # we compute the frequence of each unigram (i.e. how many times a unigram appears in all tweets given)
        freq_dist = nltk.FreqDist(filtered_unigrams)
        # we filter out less frequent unigrams using FreqDist and the threshold minFrequence
        unigramModel = {}
        for unigram, frequence in freq_dist.items():
            if frequence < minFrequence:
                pass
            else:
                unigramModel[unigram] = False
            # else: the unigram is not frequent enough in the training set and discarded
            log.debug(f"{len(unigramModel)} Unigrams will be kept in the model.")
        return unigramModel

    def __getF_Unigrams(self, counter: list, totalTweets: int, unigramsModels: dict, tweet: Series):
        """
        :param counter: a simple counter to keep track of the progress
        :param totalTweets: the total number of tweets to process to keep track of the progress
        :param unigramModels: the formated dictionary of the unigram that we keep in the model: keys are the selected unigrams, all values are set by default to false
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing an existing dictionary possibly empty
        :return: compute all unigrams occurring in the tweet and search for those unigrams in unigramModels, the set of unigrams kept as features
        If a unigram occuring in a tweet is found in unigramsModels it is set to True, otherwise nothing is done
        tweetFeatures is updated with all unigrams of unigramsModels and their respective values True if the unigrams occurs in the tweet false otherwise
        """
        # just to display the progress since the computation can be long
        counter[0] = counter[0] + 1
        if (totalTweets == counter[0]) or ((counter[0] % 500) == 0):
            log.debug(f"Unigrams have been computed for {counter[0]} on {totalTweets} tweets")

        # TODO - Figure out why we're not using this variable
        # get the unigrams occurring in the tweet
        tokenized_tweet = tweet['txt_tokenized']
        unigramsInTweet = list(ngrams(tokenized_tweet, 1))

        # unigramsModels is copied as the unigrams and their values will be used for classifying this tweet
        cbgFeatures = unigramsModels.copy()

        # TODO - Figure out what variable is supposed to be here, best guess is unigrams in tweets

        # checking if the unigrams occurring in the tweets have been seen in the training set, that is are in the unigramsModel
        for unigram in unigramsInTweet:
            if unigram in cbgFeatures:
                # one unigram has been found so I mark it
                cbgFeatures[unigram] = True
            # else: this is a new unigram, unseen in the training set, we can't use it as we don't have any label associated so we ignore it

        # we have all unigrams updated, False if they do not appear in the tweet, true if they do
        # we add them in the list of features describing the tweet
        tweet['tweetFeatures'].update(cbgFeatures)

    def __getF_DrugSuffixes(self, tweet: Series):
        """
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing a dictionary
        :return: compute if a token ends with a suffix most likely associated with drug names,
        the dictionary in the cell tweetFeatures is updated with the feature Has_Token_With_Suffix: true if a token is found, false otherwise
        """
        # by default no token finish with the suffixes
        suffixFound = False
        # Then we search token by token. The list of suffixes has to be completed...
        if TokenSearcher(tweet['txt_tokenized']).findall(r'<.*(ine|aine|xin|idol|sil|en|ive|in)>'):
            log.debug(f"=> found a token finishing by ine|aine|xin|idol|sil|en|ive|in) in tweet: {tweet['text']}")
            suffixFound = True
        # and we add the feature with the corresponding value given as boolean
        tweet['tweetFeatures'].update({'Has_Token_With_Suffix': suffixFound})

    def __getF_DrugNameCapitalized(self, tweet: Series):
        """
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing a dictionary
        :return: compute if a token begins with a capital letter as drugs are most often proper nouns,
        the dictionary in the cell tweetFeatures is updated with the feature Token_is_Capitalized: true if a token is capitalized, false otherwise
        """
        # by default no token is capitalized
        Token_Capitalized = False
        for tk in tweet['original_txt_tokenized']:
            if tk[0].isupper():
                log.debug(f"=> found a token starting with a capital letter) in tweet: {tweet['text']}")
                Token_Capitalized = True
            tweet['tweetFeatures'].update({'Token_is_Capitalized': Token_Capitalized})
        # and we add the feature with the corresponding value given as boolean

    def __getF_Phrases_Associated_With_Drugs(self, tweet: Series):
        """
        :param tweet: a series with the tweet tokenized in the cell txt_tokenized and a column 'tweetFeatures' containing a dictionary
        :return: compute if a complex phrase associated with the drug is found as a regular expression
        the dictionary in the cell tweetFeatures is updated with the feature Phrases_Associated_With_Drugs: true if the phrase exists, false otherwise
        """

        # List of phrases to search for in tweets
        REs = [
            r'(<just> <got> <prescribed>|<the> <doctor> <prescribed>|<he><prescribed><me>|<she><prescribed><me>)',
            r'(<can> <i> <take> | <i> <might> <need> <to> <take>)',
            r'(<got><me><feeling><like>)',
            r'(<any><recommendation><for>|<recommend><me><some>|<advice><for>)',
            r'(<please><bring><me><some>)'
        ]

        phrasesFound = False
        for rex in REs:
            print(rex)
            if TokenSearcher(tweet['txt_tokenized']).findall(rex):
                log.debug(f"\t-> the RE: [{rex}] found in {tweet['text']}")
                phrasesFound = True

            # and we add the feature with the corresponding value given as boolean
            tweet['tweetFeatures'].update({'Phrases_Associated_With_Drugs': phrasesFound})

    def __getF_HasDrugNames(self, tweet: Series):
        HasDrugName=False
        if tweet['drugs_predicted'] is not None:
            HasDrugName=True
        tweet['tweetFeatures'].update({'HasDrugName': HasDrugName})



    def __computeFeatures(self, tweets: DataFrame):
        """
        :param tweets: the list of tweets to compute the features
        :return: the same DataFrame with a new column which contains a dictionary with feature names as key and their values as values
        """
        # instantiate the column with an empty dict which will contain the features describing the tweets
        tweets['tweetFeatures'] = tweets.apply(lambda x: {}, axis=1)
        # apply each function computing the features describing the examples
        counter = [0]  # to display the progress

        tweets.apply(lambda row: self.__getF_Bigrams(counter, len(tweets), self.bigramsModel, row), axis=1)
        tweets.apply(lambda row: self.__getF_Unigrams(counter, len(tweets), self.unigramsModel, row), axis=1)
        tweets.apply(lambda row: self.__getF_DrugSuffixes(row), axis=1)
        tweets.apply(lambda row: self.__getF_DrugNameCapitalized(row), axis=1)
        tweets.apply(lambda row: self.__getF_Phrases_Associated_With_Drugs(row), axis=1)
        tweets.apply(lambda row: self.__getF_HasDrugNames(row), axis=1)
        # add your own features

    def train(self, trainingTweets: DataFrame):
        """
        :param: trainingTweets, a DataFrame containing a column label and a column text
        :return: add self.classifier an attribute to the object, a reference to the classifier fully trained and ready to be used for classification
        """
        assert 'text' in trainingTweets.columns, f'I was expecting the column text in the dataframe representing the training tweets but it is missing, check the data.'


        #trainingTweets = trainingTweets[:20]

        lexicon_finder = LexiconFinder(self.config)
        lexicon_finder.searchForDrugsInTweets(trainingTweets)
        # start lower casing and tokenizing the tweets
        twtTkz = TweetTokenizer()
        trainingTweets['txt_tokenized'] = trainingTweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        trainingTweets['original_txt_tokenized']=trainingTweets['text'].apply(twtTkz.tokenize)
        # build the unigrams model with a threshold of 3 or more occurrences for the unigrams
        self.unigramsModel = self.__buildUnigramsModel(trainingTweets['txt_tokenized'], 3)
        self.bigramsModel = self.__buildBigramsModel(trainingTweets['txt_tokenized'], 3)

        log.debug(f"Start computing the features for the {len(trainingTweets)} training tweets...")
        self.__computeFeatures(trainingTweets)
        assert 'tweetFeatures' in trainingTweets.columns, f'I was expecting the column tweetFeatures in the dataframe representing the training tweets after computing the features but it is missing, check the code.'

        log.debug(
            f"All features have been computed for the {len(trainingTweets)} tweets, start training the Naive Bayes...")
        # creating the list of tuples: ({feature_names: features_values}, label) to train a classifier
        examples = list(zip(trainingTweets['tweetFeatures'], trainingTweets['label']))
        self.classifier = nltk.NaiveBayesClassifier.train(examples)
        # you can try another machine learning algorithm, decision tree, for better results but it will take longer than the naive bayes
        # self.classifier = nltk.DecisionTreeClassifier.train(examples)

    def classify(self, unknownTweets: list) -> DataFrame:
        """
        :param unknownTweets, a DataFrame containing a column label and a column text
        :return: a copy of unknownTweets with a prediction column
        """
        assert 'text' in unknownTweets.columns, f'I was expecting the column text in the dataframe representing the new tweets to classify but it is missing, check the data.'
        # Similar preprocessing than in the training function: start lower casing and tokenizing the tweets
        lexicon_finder = LexiconFinder(self.config)
        lexicon_finder.searchForDrugsInTweets(unknownTweets)
        twtTkz = TweetTokenizer()
        unknownTweets['txt_tokenized'] = unknownTweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        unknownTweets['original_txt_tokenized'] = unknownTweets['text'].apply(twtTkz.tokenize)

        log.debug(f"Start computing the features for the {len(unknownTweets)} unknown tweets...")
        self.__computeFeatures(unknownTweets)
        assert 'tweetFeatures' in unknownTweets.columns, f'I was expecting the column tweetFeatures in the dataframe representing the new tweets to classify after computing their features but it is missing, check the code.'
        log.debug(f"All features have been computed for the {len(unknownTweets)} tweets, start classifying them...")

        # apply the classifier on unseen examples
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
    # change with the path to your own configuration file
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
