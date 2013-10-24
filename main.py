import numpy as np
import baseline, util, pdb, re

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
    predicted = classifier.predict(X_test)
    count = 0
    for item, labels in zip(X_test, predicted)[:20]:
        if len(labels) > 0:
            print '%s => %s' % (item, ', '.join(labels))
