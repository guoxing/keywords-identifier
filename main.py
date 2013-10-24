import baseline, util

if __name__ == '__main__':
    trainingSets = util.loadTrainingSets('xzz')
    classifiers = baseline.learnOneVsAllClassifiers( trainingSets, baseline.extractBigramFeatuers, labels, 5)

