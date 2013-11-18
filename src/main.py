from HTMLParser import HTMLParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import numpy as np
import pdb
import re
import util
import NaiveBayes as nb
from collections import defaultdict

def multiLabelClassifier(X_train, Y_train, ngram_range=None):
    if ngram_range:
        vectorizer = CountVectorizer(ngram_range=ngram_range,
                                     token_pattern=r'\b\w+\b',
                                     min_df=1)
    else:
        vectorizer = CountVectorizer(min_df=1)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()), # reduces the importance of words that are very frequent, like "the"
        ('clf', OneVsRestClassifier(LinearSVC())),
    ])
    return classifier

def printPrediction(X, predicted):
    labelsArray = []
    count = 0
    for item, labels in zip(X, predicted):
        if len(labels) > 0:
            count += 1
            print '%s\n=> %s\n\n' % (item, ', '.join(labels))
            labelsArray.append(labels)
    print "Out of %d predictions, %d of them have labels." % (len(predicted), count)
    print "====================================================="
    for labels in labelsArray:
        print repr(labels)

def printResultsTable(frequencies, numPredictions):
    print ""
    print "------------------"
    print "0:       %.2f" % ( 100 * (float(frequencies[0]) / float(numPredictions)))
    print "1-25:    %.3f" % ( 100 * (float(sum(frequencies[1:25])) / float(numPredictions)))
    print "25-50:   %.3f" % ( 100 * (float(sum(frequencies[25:50])) / float(numPredictions)))
    print "50-75:   %.3f" % ( 100 * (float(sum(frequencies[50:75])) / float(numPredictions)))
    print "75-100:  %.3f" % ( 100 * (float(sum(frequencies[75:100])) / float(numPredictions)))
    print "------------------"
    print ""

def plotPrediction(predicted, Y_train):
    print "There are %d predictions" % len(predicted)
    X_labels = 100*[0]
    for i in range(100):
        X_labels[i] = str(i)
    frequencies = [0]*100
    totalNumTags = 0
    totalTagsCorrect = 0
    for index, prediction in enumerate(predicted):
        numTags = max(len(Y_train[index]), len(predicted[index]))
        totalNumTags += numTags
        correctCount = 0
        for i in range(len(predicted[index])):
            if predicted[index][i] in Y_train[index]:
                correctCount += 1
        totalTagsCorrect += correctCount
        percentCorrect = float(correctCount) / float(numTags)
        bucketIndex = int(100*percentCorrect)
        if bucketIndex == 100:
            bucketIndex = 99 # put 100% in the last bucket
        frequencies[bucketIndex] += 1
    printResultsTable(frequencies, len(predicted))
    print "%.3f percent of tags are correct" % (100.0 * (float(totalTagsCorrect) / float(totalNumTags)))
    pos = np.arange(len(X_labels))
    width = 1.0 # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2)) # center the ticks
    ax.set_xticklabels(X_labels)
    plt.bar(pos, frequencies, width, color='r')
    plt.show()
    # TODO the histogram of predictions that predicted too many tags

def testSVM(X_train, Y_train, testingSet):
    print "TESTING SVM"
    classifier = multiLabelClassifier(X_train, Y_train)
    X_test, Y_test = mergeTitlesAndBodies(testingSet)
    print "Parsed the testing data"
    classifier.fit(X_train, Y_train)
    print "Fit the training data"
    predicted = classifier.predict(X_test)
    #printPrediction(X_test, predicted)
    return predicted

if __name__ == '__main__':
    trainingSet = util.loadDataSet('out_1000_0')
    testingSet = util.loadDataSet('out_1000_1')
    my_nb = nb.NaiveBayes(trainingSet)

    my_nb.train()
    my_nb.test(testingSet)
