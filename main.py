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

def mergeTitlesAndBodies(dataset):
    # each entry in the trainingSet looks like this
    # ['996004',
    #  '.NET Dictionary: is only enumerating thread safe?',
    #  '<p>Is simply enumerating a .NET Dictionary from multiple threads safe? </p>\n\n<p>No modification of the Dictionary takes place at all.</p>\n',
    #  '.net multithreading dictionary thread-safety'
    # ]
    X = []
    Y = []
    for example in dataset:
        if len(example) == 3:
            qid, title, body = example
            tags = ""
        elif len(example) == 4:
            qid, title, body, tags = example
        else:
            continue

        title = stripHTMLTags(title)
        body = stripNewlines(body)
        body = stripHTMLTags(body)
        X.append(title + ' ' + body)
        Y.append(tags.split())
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

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

def tagCountNaiveBayes():
    pass

def trainNaiveBayes(X_train_nb, Y_train, vocab, tags):
    phi_k_list = []
    phi_y_list = []
    for tag in tags:
        phi_k, phi_y = trainBaseNaiveBayes(X_train_nb, Y_train, len(vocab), tag)
        phi_k_list.append(phi_k)
        phi_y_list.append(phi_y)

    return phi_k_list, phi_y_list

def trainBaseNaiveBayes(X_train_nb, Y_train, num_words, tag):
    num_questions = len(X_train_nb)

    # phi_k_d: phi_k denominator
    # phi_k_n: phi_k nominator

    # laplace smoothing
    phi_k_d = np.ones(2) * num_words
    phi_k_n = np.ones((2, num_words))
    
    phi_y = 0
    for i in range(num_questions):
        idx = int(tag in Y_train[i])
        phi_k_d[idx] += sum(X_train_nb[i])
        for k in range(num_words):
            phi_k_n[idx, k] += X_train_nb[i][k]
        phi_y += idx

    phi_y /= float(num_questions)

    phi_k = np.zeros((2, num_words))
    for i in range(2):
        for k in range(num_words):
            phi_k[i, k] = phi_k_n[i, k] / phi_k_d[i]

    return phi_k, phi_y

def testNaiveBayes(phi_k_list, phi_y_list, X_test_nb, num_words, tags):
    """
    return a list of lists. Each row represents a question,
    and is composed of tags predicted.
    """

    print "TESTING BAYES"
    num_questions = len(X_test_nb)
    for i in range(num_questions):
        tag_prob = []
        # compute a probability for each tag
        for j in range(len(tags)):
            phi_k = phi_k_list[j]
            phi_y = phi_y_list[j]
            log_p_1 = np.log(phi_y)
            log_p_0 = np.log(1 - phi_y)
            for k in range(num_words):
                log_p_1 += np.log(phi_k[1, k]) * X_test_nb[i][k]
                log_p_0 += np.log(phi_k[0, k]) * X_test_nb[i][k]
            # need real probability instead of log_p_0 and log_p_1 comparison
            # use log because product is too small that can result in divide-by-zero
            tag_prob.append(1 / (1 + np.exp(log_p_0 - log_p_1)))

        tag_prob = sorted(dict(zip(range(len(tag_prob)), tag_prob)).items(),\
                          key=lambda x:x[1], reverse=True)

    #TODO use classfication/regression to predict k (#tags) for each question 


def createVocabularyForNaiveBayes(X_train):
    """
    Create a vocabulary (dictionary) for all the words appeared.
    Used for multinomial Naive Bayes.
    
    NOTE: Vocabulary doens't have to be generated this way.
          Can come up with better vocabulary. (say filter out some words)
    """
    vocab = set()
    for x in X_train:
        # trivial word split and purification
        for word in x.split():
            word = util.purify(word)
            vocab.add(word)
    vocab.remove("")
    return dict(zip(vocab, range(len(vocab))))

def generateTagsForNaiveBayes(Y_train):
    """
    Generate a set of tags from Y_train.
    """
    return list(set([tag for l in Y_train for tag in l]))


def convertDataForNaiveBayes(X, vocab):
    """
    converts X_train to a list of lists. Each row represents a question,
    and is composed of word frequencies

    Can be used for X_train or X_test
    """
    # each row contains a dictionary of word frequencies.
    rows = []
    for x in X:
        row = [0] * len(vocab)
        for word in x.split():
            word = util.purify(word)
            if word == "":
                continue
            if not vocab.has_key(word):
                continue
            row[vocab[word]] += 1
        rows.append(row)

    return rows

def numQuestionsWithTag(Y_train, tag):
    num = 0
    for y in Y_train:
        if tag in y:
            num += 1
    return num

# guoxing: not sure what you're doing here. I've almost finished the testNaiveBayes.
# check if that is what you want. If so, please delete this.
#def testNaiveBayes(X_train, Y_train, testingSet):
    #"""
    #Our implementation of multinomial Naive Bayes
    #"""
    #print "TESTING BAYES"
    #numTrainQuestions = len(X_train)
    #x, y = convertTrainDataForNaiveBayes(X_train, Y_train)
    ## now train a classifier for each tag

    #numQuestions = len(x)

    ## get the priors per tag, in a dict
    #priors = {}
    #for tag in tags:
        ## the fraction of questions that have the tag out of all questions
        #priors[tag] = float(numQuestionsWithTag(Y_train, tag)) / float(numQuestions)

    ## find the probability of word x given the tag
    ## the fraction of words out of all words that are tagged with the tag




    ## when we get a new test question, run each classifier on it
    ## and capture the top k classifiers


# TODO I ran the classifier on the whole training set for ten minutes
# until I ran into this error: UnicodeDecodeError: 'ascii' codec can't
# decode byte 0xe2 in position 45: ordinal not in range(128).
if __name__ == '__main__':
    trainingSet = util.loadDataSet('out0')
    X_train, Y_train = mergeTitlesAndBodies(trainingSet)
    vocab = createVocabularyForNaiveBayes(X_train)
    X_train_nb = convertDataForNaiveBayes(X_train, vocab)
    tags = generateTagsForNaiveBayes(Y_train)
    phi_k_list, phi_y_list = trainNaiveBayes(X_train_nb, Y_train, vocab, tags)

    # This is HUUUGE amount of work even for 100 questions
    # might want to reduce the size of vocabulary
    print "tags count: ", len(tags)
    print "words count: ", len(vocab)
    print "questions count: ", len(X_train_nb)
    print "time complexity: ", len(tags) * len(vocab) * len(X_train_nb)
    print "Parsed the training data"

    testingSet = util.loadDataSet('out1')
    X_test, Y_test = mergeTitlesAndBodies(testingSet)
    X_test_nb = convertDataForNaiveBayes(X_test, vocab)
    testNaiveBayes(phi_k_list, phi_y_list, X_test_nb, len(vocab), tags)


    #testingSet = util.loadTrainingSet('xwi')
    #predicted = testNaiveBayes(X_train, Y_train, testingSet)
    #plotPrediction(predicted, Y_test)

