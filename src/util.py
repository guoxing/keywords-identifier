import collections
import csv
import operator
import os
import pdb
import random
import string
import numpy as np
import re
from HTMLParser import HTMLParser

# The data files were generated via split -l 355100 Train.csv
# That leaves partial CSV entries on the top and bottom of
# each file, so delete these partial entries by hand before loading
# the CSV files
# returns (trainingSet, tags)
# where tags are all the tags in the training set

# works for both training dataset and testing dataset
def loadDataSet(filename):
    tags = set()
    with open('../data/' + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            yield row
    # in xzz there are 3980 tags, 1694 of which are unique

def computeErrorRate(examples, classifier):
    """
    @param list examples
    @param dict params: parameters
    @param function predict: (params, x) => y
    @return float errorRate: fraction of examples we make a mistake on.
    """
    numErrors = 0
    for x, y in examples:
        if classifier.classifyWithLabel(x) != y:
            numErrors += 1
    return 1.0 * numErrors / len(examples)

def countTags(dataset, suffix):
    tags_count = collections.defaultdict(lambda: 0)
    for example in dataset:
        tags = example[-1]
        for tag in tags.split():
            tags_count[tag] += 1;
    total_count = sum(tags_count.values());
    with open("tags_count_" + suffix, 'w') as f:
        print >> f, "total_count: ", total_count
        sorted_tags = sorted(tags_count.items(), key=lambda x:x[1], reverse=1)
        for tag, count in sorted_tags:
            print >> f, '{0:40} : {1:10} : {2:.2f}%'\
                    .format(tag, count, count / float(total_count) * 100)

def readinStopwords():
    with open("stop-words-list.txt", "r") as f:
        return [purify(word.strip()) for word in f]

def removeStopwords(vocab, stopWordsList):
    filteredList = []
    for word in vocab:
        if word not in stopWordsList:
            filteredList.append(word)
    return filteredList

def purify(word):
    """
    remove punctuations/digits, lower case word
    """
    # TODO: dont' want to remove c++, c#, .net, etc.
    punc = set(string.punctuation)
    digit = set(string.digits)
    filt = punc.union(digit)
    word = "".join(ch for ch in word if ch not in filt)
    word = word.lower()
    return word

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
    qids = []
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
        qids.append(qid)
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y, qids)
