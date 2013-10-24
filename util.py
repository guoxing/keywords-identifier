import os, random, operator, collections, pdb, csv

# The data files were generated via split -l 355100 Train.csv
# That leaves partial CSV entries on the top and bottom of
# each file, so delete these partial entries by hand before loading
# the CSV files
# returns (trainingSets, tags)
# where tags are all the tags in the training set
def loadTrainingSets(filename):
    trainingSets = []
    tags = []
    with open('train_data/' + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            pdb.set_trace()
            trainingSets.append(row)
    return (trainingSets, tags)

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
