BMIN 522 Laboratory 2: Finding drug names in tweets with standard machine learning

The goal of this second laboratory is the same as the first laboratory: writing a program that can find automatically tweets mentioning drugs. But this time, we will use a more advanced technique, a Naive Bayes classifier which is a standard machine learning algorithm. The main advantage of this technique is that, if a labeled corpus is provided for training, the algorithm will be able to learn the "rules" to automatically discriminate tweets mentioning drugs from those which do not. A limitation of this type of algorithms is that the features representing the examples have to be given to the algorithm.

The way it works for nltk is simple. We first have to define a dictionary that will represent the properties, aka 'features', of a given tweet that we want to classify. In this dictionary, the keys are string denoting the names of the features and the values are the actual values of the features describing a tweet observed. To take a concrete example, consider the tweet: 'I love her last song !'. To describe this tweet, we could use its length as a feature, a length expressed as the number of tokens occurring in the tweet; a second feature 'pos_word' could be the mention of word associated with positive feeling such as love, this feature could be equal to 1 if a word from a predefined list of positive words appear in the tweet, 0 otherwise. Given these two features, the dictionary representing the tweet would be {'length': 6, 'pos_word': 1} since the tweet is 6 tokens long and it contains the word love which one would expect to be in a list of positive words.

For this lab, we have implemented a simple feature 'Has_Token_With_Suffix' computed by the method __getF_DrugSuffixes which is equal to 1 if any token in a tweet is found to end with a suffix of interest, 0 otherwise. We have also implemented a more complex set of features based on the notion of 'Ngrams'. We computed all Bigrams with the methods __getF_Bigrams and __buildBigramsModel. Each bigram observed in the training set, if it is frequent enough, becomes a feature. When a tweet is presented to the classifier, the classifier compute all bigrams of the tweet and remove from the list of the bigrams computed the bigrams that are not used as features (i.e. bigrams which have not been observed in the training set). A small example is probably better than a long explanation: 

Consider the following training set composed of only 3 tweets:
T1: 'i love dogs !' -> bigrams: (<i>, <love>), (<love>, <dogs>), (<dogs>, <!>)
T2: 'i love cats too ^^' -> bigrams: (<i>, <love>), (<love>, <cats>), (<cats>, <too>), (<too>, <^^>) 
T3: 'but i hate birds ...' -> bigrams: (<but>, <i>), (<i>, <hate>), (<hate>, <birds>), (<birds>, <...>)
In addition to the feature 'Has_Token_With_Suffix', we keep all bigrams occurring in our training set as features (only (<i>, <love>) occurs twice in the training set, all others occur only once.). Therefore, our dictionary of features representing a tweet contains 11 features:
['Has_Token_With_Suffix', '<i>, <love>', '<love>, <dogs>', '<dogs>, <!>', '<love>, <cats>', '<cats>, <too>', '<too>, <^^>', '<but>, <i>', '<i>, <hate>', '<hate>, <birds>', '<birds>, <...>']

Assuming our classifier is presented with the following tweet: 
T4: 'i hate cats too !' -> bigrams: (<i>, <hate>), (<hate>, <cats>), (<cats>, <too>), (<too>, <!>)
Its representation becomes:
{'Has_Token_With_Suffix': 0, '<i>, <love>': 0, '<love>, <dogs>': 0, '<dogs>, <!>': 0, '<love>, <cats>': 0, '<cats>, <too>': 1, '<too>, <^^>': 0, '<but>, <i>': 0, '<i>, <hate>': 1, '<hate>, <birds>': 0, '<birds>, <...>': 0]
'Has_Token_With_Suffix': 0, since none of the tokens in T4 ends with ine|aine|xin|idol|sil
'<i>, <love>': 0, since the bigram (<i>, <love>) does not occur in T4, idem for '<love>, <dogs>': 0 etc.
'<cats>, <too>': 1 and '<i>, <hate>': 1, since both bigrams occur in T4 and they are used as feature in our representation
Note that, the bigrams (<hate>, <cats>) and (<too>, <!>), while they appear in T4, do not appear in the feature dictionary since they are not used as features. This is the first time the classifier 'sees' these 2 bigrams. Since they did not occur in the training set, the classifier does not know how to use them in its inference.

How the classifier learns to discriminate tweets mentioning drugs from the others using the features representation of the tweets in the training set, this will be discussed in the class. But you can guess from the name of the classifier used in the instruction nltk.NaiveBayesClassifier.train(examples) that frequencies and the Bayes' theorem will be used to compute specific probabilities.

Question 1: We used a few features to represent the tweets, we can add more.
-> Complete the list of suffixes used for the feature 'Has_Token_With_Suffix'
-> Following the example of the methods implemented to compute the bigrams, implement __buildUnigramsModel and __getF_Unigrams which should compute the most frequent unigrams and used them as features to help the classification. Remember from the class that a unigram is a simple token. So for our previous example, 'I love her last song !', we would have 6 unigrams. Some unigrams, called stop words (the set of grammatical and very frequent words), will be too frequent and will probably not help the classifier to discriminate tweets. Use the function stopwords.words('english') which return a list of predefined stop words, to remove them from the list of unigrams used as features. Punctuation will also be too frequent to be useful, use the function string.punctuation to remove them as well.
-> Our lexicon can also be re-used as a feature. Implement a method __getF_HasDrugNames which computes the feature Has_DrugNames. The feature should be equal to 1 if a tweet contains one or more drug names found by LexiconFinder, 0 otherwise. Tips: in the train method of NaiveBayesFinder create an instance of LexiconFinder and call searchForDrugsInTweets on the training set this will add the column drugs_predicted which values are equal to None if tweets do not contain names from the lexicon, the list of tokens of the names discovered otherwise. Using this column to implement the feature should then be straightforward. Applying searchForDrugsInTweets on the training corpus will take some time. You can reduce the size of the training set to develop the method and make your tests. You can use the command trainingTweets = trainingTweets[:20] for example. It will take only 20 training examples.
-> Find and implement two other features you think may help the classification.

Question 2: How many features are needed and which features are useful for the classification is still a research question for the branch of machine learning called feature engineering. Adding more features does not necessarily mean getting better results, it may also decrease the overall performance of a classifier. In this second part, you will run a manual ablation study to evaluate the classifier.
-> By commenting out the call of the methods __getF_*** in the method __computeFeatures, evaluate the performance of the classifier by adding the features one by one:
a. evaluate the classifier by computing only the unigrams.
b. evaluate the classifier by computing the unigrams and the bigrams.
c. evaluate the classifier by computing the unigrams, bigrams and the suffixes
d. evaluate the classifier by computing the unigrams, bigrams, suffixes, and your first feature
e. evaluate the classifier by computing the unigrams, bigrams, suffixes, your first feature, and your second feature
f. evaluate the classifier by computing the unigrams, bigrams, suffixes, your two features, and the feature Has_DrugNames
-> report the results of the ablation study in a table.
-> select the best set of features and analyze 100 FPs and 100 FNs made by the classifier.