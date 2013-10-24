import baseline, util, pdb

if __name__ == '__main__':
    trainingSets, tags = util.loadTrainingSets('xzz')
    pdb.set_trace()
    classifiers = baseline.learnOneVsAllClassifiers(trainingSets,
                                                    baseline.extractUnigramFeatuers,
                                                    tags,
                                                    5)
