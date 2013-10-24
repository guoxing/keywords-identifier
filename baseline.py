"""
CS221 2013
AssignmentID: spam
"""

import pdb, sys, operator
from collections import defaultdict

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self._blacklist = set(blacklist) if (k < 0) else set(blacklist[0:k])
        self._threshold = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        num_spam_words = 0
        for word in text.split():
            if word in self._blacklist:
                num_spam_words = num_spam_words + 1;
                if num_spam_words >= self._threshold:
                    return 1
        return -1
        # TODO take threshold into account
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$.
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    # key: the word in the text
    # value: the frequency
    cnt = defaultdict(int)
    for word in x.split():
        cnt[word] += 1
    return cnt
    # END_YOUR_CODE

def sparseVectorDotProduct(v1, v2):
    # smaller vector,
    smallerVector = []
    largerVector = []

    if len(v1) <= len(v2):
        smallerVector = v1
        largerVector = v2
    else:
        largerVector = v1
        smallerVector = v2

    # iterate over the keys
    dot_product = 0
    for key in smallerVector.keys():
        dot_product += smallerVector[key] * largerVector[key]
    return dot_product

class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = defaultdict(int, params)

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        v1 = self.featureFunction(x)
        return sparseVectorDotProduct(v1, self.params)
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('positive', 'negative'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    positiveLabel = labels[0]
    weights = defaultdict(int)
    featuresList = []
    for index in range(len(trainExamples)):
        features = featureExtractor(trainExamples[index][0])
        featuresList.append(features)
    for iteration in range(iters):
        for index in range(len(trainExamples)):
            message = trainExamples[index][0]
            guessValue = sparseVectorDotProduct(featuresList[index], weights)
            if guessValue >= 0 and trainExamples[index][1] != positiveLabel:
                keys = featuresList[index].keys()
                for key in keys:
                    weights[key] -= featuresList[index][key]
            elif guessValue < 0 and trainExamples[index][1] == positiveLabel:
                keys = set(message.split())
                for key in keys:
                    weights[key] += featuresList[index][key]
    return weights
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$.

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    cnt = defaultdict(int)

    # add the unigrams
    words = x.split()
    for word in words:
        cnt[word] += 1

    # add the bigrams
    num_words = len(words)
    cnt['-BEGIN- ' + words[0]] += 1
    for i in range(num_words - 1):
        cnt[words[i] + " " + words[i+1]] += 1

    return cnt
    # END_YOUR_CODE

def allClassifiersTie(all_results):
    val = all_results[0][1]
    for i in range(1, len(all_results)):
        if all_results[i][1] != val:
            return False
    return True

def bestLabel(all_results):
    best = -sys.maxint -1
    best_label = ""
    for i in range(len(all_results)):
        if (all_results[i][1] > best):
            best = all_results[i][1]
            best_label = all_results[i][0]
    return best_label # which should I choose when they all tie?

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self._labels = labels
        self._classifiers = classifiers
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores
        """
        raise NotImplementedError

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        all_results = self.classify(x)
        return bestLabel(all_results)
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        result = []
        for i in range(len(self._classifiers)):
            classifier = self._classifiers[i]
            result.append((classifier[0], classifier[1].classify(x))) # check type
        return result
        # END_YOUR_CODE

def learnOneVsAllClassifiers( trainExamples, featureFunction, labels, perClassifierIters = 10 ):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    retval = []
    for label in labels:
        labelParams = [label, "!" + label] # e.g. comp and !com
        weights = learnWeightsFromPerceptron(trainExamples, featureFunction, labelParams)
        retval.append((label, WeightedClassifier(labels, featureFunction, weights)))
    return retval

    # the results of this function look good.
    # for 'god', works like 'program' and 'disk' are -56, and the top scored words are
    # {'he': 513, 'Jesus': 480, 'Law': 400, 'his': 283, 'makes': 270, 'did': 263,
    # 'Are': 235, 'shall': 225, 'earth': 200, 'Law,': 200, 'read': 181
    # for 'comp', 'Jesus' is -12[ and membrane is -80
    # the top socred words for comp are:
    # {'I': 642, '|>': 637, 'a': 468, '=': 447, '|': 405, '*': 400, 'Subject:': 359, 'From:': 342,
    # 'OFF': 288, 'DOS': 274, 'X': 273, 'lut_index': 272, 'Windows': 267, 'file': 259,
    # END_YOUR_CODE


