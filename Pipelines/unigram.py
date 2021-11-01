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

    def __buildUnigramsModel(self, trainingTweetsTokenized: Series, minFrequence=1) -> dict:
        """
        :param trainingTweetsTokenized: all tweets from a training set tokenized
        :param minFrequence: the minimum frequence to keep a bigram, by default all are kept, i.e. minFrequence=1
        :return: a dictionary with the most frequent bigrams as key and the default value False to prepare next steps: the training and prediction steps
        :note: This is a lot of bigrams, we just filter the most unfrequent. Another choice could be to remove stop words (ex. the, of, a etc.) and the punctuations and then compute the bigrams.
        """
        # we create the unigrams of all tweets using ntlk function and store them in the list unigrams
        Unigrams = []
        trainingTweetsTokenized.apply(lambda tweetTokenized: Unigrams.extend(ngrams(tweetTokenized, 1)))

        # remove stop words (ex. the, of, a etc.)
        stop_words = set(stopwords.words('english'))
        # word_tokens = word_tokenize(trainingTweetsTokenized)
        filtered_unigrams = [w[0] for w in Unigrams if not w[0].lower() in stop_words]
        for w in Unigrams:
            if w not in stop_words:
                filtered_unigrams.append(w)

        print(filtered_unigrams)

        # remove punctuation
        punctuation = string.punctuation
        print(punctuation)
        while (punctuation in filtered_unigrams):
            test_list.remove(punctuation)


        #filtered_unigrams.remove(punctuation)

        # we compute the frequence of each unigram (i.e. how many times a unigram appears in all tweets given)
        freq_dist = nltk.FreqDist(filtered_unigrams)
        # we filter out less frequent unigrams using FreqDist and the threshold minFrequence
        unigramModel = {}
        for unigram, frequence in freq_dist.items():
            if frequence >= minFrequence:
                unigramModel[unigram] = False
            # else: the bigram is not frequent enough in the training set and discarded
        log.debug(f"{len(unigramModel)} Unigrams will be kept in the model.")
        return unigramModel

    def __getF_Unigrams(self, counter: list, totalTweets: int, unigramsModel: dict, tweet: Series):
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
        # get the bigrams occurring in the tweet
        UnigramsInTweet = ngrams(tweet['txt_tokenized'], 2)
        # unigramsModels is copied as the unigrams and their values will be used for classifying this tweet
        cbgFeatures = unigramsModels.copy()
        # checking if the unigrams occurring in the tweets have been seen in the training set, that is are in the unigramsModel
        for unigram in UnigramsInTweetInTweet:
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

    # Feature1: Extract tweets

    def __init__(self, config: configparser):
        '''
        Constructor
        '''
        self.config = config
        # the list of REs to search for, all REs should be in lower case following the same tokenization rules than TweetTokenizer
        self.REs = [
            r'(<prescribed>|<prescribes>|<recommended>|<advised>) '
            r'(<with>|<for>)',
            r'(<doctor>|<nurse>)(<switched>|<changed>)',
            r'<had> <me> <feeling>|<like>',
            r'<got> <me> <feeling>',
            r'<better>|<worse>|<same>',
            r'(<took>|<popped>|<swallowed>) <pill> <.*>* <.*(ine|aine|xin|idol|sil|olol|en|an|s)>',
            r'(<mg>|<milligram>|<g>|<gram>|<application>)',

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
        # a dic to remember which pattern matched
        tweetsMatching = {}
        # search for all REs in the tweets
        tweets.apply(lambda row: self.__applyREs(row, 'txt_tokenized', tweetsMatching), axis=1)

        # now that I have all tweets matched by a REs I add the label in the tweet dataframe
        # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['pattern_matching'] = ''

        for tweetID, patterns in tweetsMatching.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'pattern_matching'] = ' --- '.join(patterns)
        # remove the column nltkText which is now useless
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
            print(rex)
            if TokenSearcher(tweet[columnToApplyREs]).findall(rex):
                log.debug(f"\t-> the RE: [{rex}] found in {tweet['text']}")
                if tweet.name in tweetsMatching:
                    tweetsMatching[tweet.name].append(rex)
                else:
                    tweetsMatching[tweet.name] = [rex]

    # Feature 2: Extract tweets which contain 2 or more symptoms.

    def __init__(self, config: configparser):
        '''
        Constructor
        '''
        self.config = config
        # the list of REs to search for, all REs should be in lower case following the same tokenization rules than TweetTokenizer
        self.REs = [
            r'(<pain>|<discomfort>|<hurt>)',
            r'(<nausea>|<sick>)(<vomiting>|<puking>)',
            r'(<stomach> <ache>|<belly> <ache>)',
            r'(<fever>|<hot>|<warm>|<flushed>)',
            r'(<diarrhea>|<poo>)',
            r'(<blood>|<bleed>|<blood><loss>)',
            r'(<rash>)(<spot>|<spots>)',

        ]

    def symptom_count_in_tweet(self, tweet: string):
        counter = 0

        for symptom_re in self.REs:
            searcher = TokenSearcher(tweet)
            matches = searcher.findall(symptom_re)
            if len(matches) >= 1:
                counter = counter + 1

        return counter

    def symptom_count_in_tweets(self, tweets):

        for tweet in tweets:
            # TODO: Store the count of the symptoms to a variable
            symptom_count = symptom_count_in_tweet(tweet)

            # TODO: Add an if statement that runs if there is 1 or more symptoms
            # If t

            # now that we have all tweets mentioning a name of drug we add the label in the tweets dataframe
            # write 0 and '' by default and, update later for the tweets mentioning drugs
        tweets['prediction'] = 0
        tweets['drugs_predicted'] = ''
        for tweetID, drugs in tweetsMentioningDrugs.items():
            tweets.loc[tweetID, 'prediction'] = 1
            tweets.loc[tweetID, 'drugs_predicted'] = str(drugs)
        # remove the column txt_tokenized which is now useless
        return tweets

    def __computeFeatures(self, tweets: DataFrame):
        """
        :param tweets: the list of tweets to compute the features
        :return: the same DataFrame with a new column which contains a dictionary with feature names as key and their values as values
        """
        # instantiate the column with an empty dict which will contain the features describing the tweets
        tweets['tweetFeatures'] = tweets.apply(lambda x: {}, axis=1)
        # apply each function computing the features describing the examples
        counter = [0]  # to display the progress
        tweets.apply(lambda row: self.__getF_Unigrams(counter, len(tweets), self.unigramsModel, row), axis=1)
        tweets.apply(lambda row: self.__getF_DrugSuffixes(row), axis=1)
        #add your own features

    def train(self, trainingTweets: DataFrame):
        """
        :param: trainingTweets, a DataFrame containing a column label and a column text
        :return: add self.classifier an attribute to the object, a reference to the classifier fully trained and ready to be used for classification
        """
        assert 'text' in trainingTweets.columns, f'I was expecting the column text in the dataframe representing the training tweets but it is missing, check the data.'
        # start lower casing and tokenizing the tweets
        twtTkz = TweetTokenizer()
        trainingTweets['txt_tokenized'] = trainingTweets['text'].apply(str.lower).apply(twtTkz.tokenize)
        # build the unigrams model with a threshold of 3 or more occurrences for the unigrams
        self.unigramsModel = self.__buildUnigramsModel(trainingTweets['txt_tokenized'], 3)
        # create an instance of LexiconFinder and call searchForDrugsInTweets on the training set
        lexicon_finder = LexiconFinder(self.config)
        lexicon_finder.searchForDrugsInTweets(trainingTweets)

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


# def __getF_HasDrugNames (self,trainingTweets:DataFrame) -> dict:


## Additional Features
# pos word (words associated with drugs)--search drugs by REs?

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
