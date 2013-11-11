import collections
import csv
import operator
import os
import pdb
import random

# The data files were generated via split -l 355100 Train.csv
# That leaves partial CSV entries on the top and bottom of
# each file, so delete these partial entries by hand before loading
# the CSV files
# returns (trainingSet, tags)
# where tags are all the tags in the training set
def loadTrainingSet(filename):
    tags = set()
    with open('train_data/' + filename, 'rb') as csvfile:
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

def countTags(dataset):
    tags_count = collections.defaultdict(lambda: 0)
    for example in dataset:
        tags = example[-1]
        for tag in tags.split():
            tags_count[tag] += 1;
    total_count = sum(tags_count.values());
    with open("tags_count", 'w') as f:
        print >> f, "total_count: ", total_count
        sorted_tags = sorted(tags_count.items(), key=lambda x:x[1], reverse=1)
        for tag, count in sorted_tags:
            print >> f, '{0:40} : {1:10} : {2:.2f}%'\
                    .format(tag, count, count / float(total_count) * 100)
