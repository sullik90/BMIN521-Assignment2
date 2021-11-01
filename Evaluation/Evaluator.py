'''
Created on Dec 23, 2019

@author: dweissen
'''
from pandas.core.frame import DataFrame
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from nltk.metrics.scores import precision, recall, f_measure
from nltk.metrics import ConfusionMatrix
import collections

class Evaluator(object):
    '''
    An object evaluating the performance of a classifier
    '''
    def evaluate(self, annotatedTweets: DataFrame):
        '''
        :param annotatedTweets: expect a dataframe with 2 columns: 
        - label which contains the gold truth annotation of a tweet
        - prediction which contains the predicted annotation
        :return: evaluate the prediction of the classifier
        '''        
        assert set(['label', 'prediction']).issubset(set(annotatedTweets.columns)), f"One of the columns label or prediction is missing, check the input."
        cf = ConfusionMatrix(list(annotatedTweets['label']), list(annotatedTweets['prediction']))
        
        #nltk is expecting two sets, one for the ground truth, the other one for the prediction
        #here we build 2 sets in the dicts: label -> {indexes of tweets annotated with this label}
        refset = collections.defaultdict(set)
        #here we build 2 sets in the dicts: label -> {indexes of tweets predicted with this label}
        predset = collections.defaultdict(set)
        def __buildEvaluationSets(tweet, refset: collections.defaultdict, predset: collections.defaultdict):
            refset[tweet['label']].add(tweet.name)
            predset[tweet['prediction']].add(tweet.name)
        annotatedTweets.apply(lambda tweet: __buildEvaluationSets(tweet, refset, predset), axis=1)
        
        #now we can compute the precision, recall and f1 for the positive class (here 1) and the negative class (here 0)
        #we just compute the prec,rec, f1 for the positive class
        prec = precision(refset[1], predset[1])
        rec = recall(refset[1], predset[1])
        f1 = f_measure(refset[1], predset[1])
        return cf, prec, rec, f1
    
    