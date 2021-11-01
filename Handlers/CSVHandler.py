'''
Created on Dec 28, 2019

@author: dweissen
'''
import pandas as pd
import configparser


class CSVHandler(object):
    '''
    A convenient class to read the csv files and access the training, validation and test set of examples
    '''

    def __init__(self, config: configparser):
        self.config = config
        self.__readData()
        
    
    def __readData(self):
        """
        Read the data given in the config path and keep them in memory as train, val and test
        """
        #reading the file with their original format
        self.train = pd.read_csv(self.config['Data']['trainPath'], sep='\t', encoding='utf8')
        self.train.set_index('tweet_id', inplace=True)
        self.val = pd.read_csv(self.config['Data']['valPath'], sep='\t', encoding='utf8')
        self.val.set_index('tweet_id', inplace=True)
        self.test = pd.read_csv(self.config['Data']['testPath'], sep='\t', encoding='utf8')
        self.test.set_index('tweet_id', inplace=True)
        #sanity checking
        assert 'text' in self.train.columns, "I was expecting the column 'text' in the train dataframe, but I can't find it."
        assert 'text' in self.val.columns, "I was expecting the column 'text' in the val dataframe, but I can't find it."
        assert 'text' in self.test.columns, "I was expecting the column 'text' in the test dataframe, but I can't find it."
        assert(len(self.train)==7698), f"I was expecting training examples, I got {len(self.train)}..."
        assert(len(self.val)==962), f"I was expecting training examples, I got {len(self.val)}..."
        assert(len(self.test)==962), f"I was expecting training examples, I got {len(self.test)}..."
    
    def getTraining(self):
        return self.train
    def getValidation(self):
        return self.val
    def getTest(self):
        return self.test
    
    