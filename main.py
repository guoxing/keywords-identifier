import util, pdb, re
import numpy as np
import matplotlib.pyplot as plt

from HTMLParser import HTMLParser

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

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

class HTMLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def stripHTMLTags(html):
    s = HTMLStripper()
    s.feed(html)
    return s.get_data()

def stripNewlines(str):
    return re.sub(r'\n', '', str)

# for now, disregard the id
def mergeTitlesAndBodies(dataset, isTrain=True): # vs. isTest
    # each entry in the trainingSet looks like this
    # ['996004',
    #  '.NET Dictionary: is only enumerating thread safe?',
    #  '<p>Is simply enumerating a .NET Dictionary from multiple threads safe? </p>\n\n<p>No modification of the Dictionary takes place at all.</p>\n',
    #  '.net multithreading dictionary thread-safety'
    # ]
    X = []
    Y = []
    for example in dataset:
        if isTrain:
            id, title, body, tags = example
        else:
            id, title, body = example
        title = stripHTMLTags(title)
        body = stripNewlines(body)
        body = stripHTMLTags(body)
        X.append(title + ' ' + body)
        if isTrain:
            Y.append(tags.split())
    X = np.array(X)
    Y = np.array(Y)
    if isTrain:
        return (X, Y)
    else:
        return X

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

def plotPrediction(predicted, Y_train):
    print "There are %d predictions" % len(predicted)
    #X_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    X_labels = 100*[0]
    for i in range(100):
        X_labels[i] = str(i)
    #frequencies = [0]*10
    frequencies = [0]*100
    for index, prediction in enumerate(predicted):
        numTags = max(len(Y_train[index]), len(predicted[index]))
        correctCount = 0
        for i in range(len(predicted[index])):
            if predicted[index][i] in Y_train[index]:
                correctCount += 1
        percentCorrect = float(correctCount) / float(numTags)
        bucketIndex = int(100*percentCorrect)
        if bucketIndex == 100:
            bucketIndex = 99 # put 100% in the last bucket
        frequencies[bucketIndex] += 1
    pos = np.arange(len(X_labels))
    width = 1.0 # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2)) # center the ticks
    ax.set_xticklabels(X_labels)
    plt.bar(pos, frequencies, width, color='r')
    plt.show()
    # TODO the histogram of predictions that predicted too many tags

if __name__ == '__main__':
    trainingSet = util.loadTrainingSet('xzz')
    X_train, Y_train = mergeTitlesAndBodies(trainingSet)
    print "Parsed the training data"
    classifier = multiLabelClassifier(X_train, Y_train)
    testingSet = util.loadTestingSet('xxa')
    X_test = mergeTitlesAndBodies(testingSet, isTrain=False)
    print "Parsed the testing data"
    classifier.fit(X_train, Y_train)
    print "Fit the training data"
    predicted = classifier.predict(X_train)
    printPrediction(X_train, predicted)
    plotPrediction(predicted, Y_train)
    predicted = classifier.predict(X_test)
