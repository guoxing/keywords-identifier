import util
import re
import copy
import numpy as np

class NaiveBayes:

    def __init__(self, trainingSet):
        self.X_train, self.Y_train, self.qids = util.mergeTitlesAndBodies(trainingSet)
        self._createVocabulary_()
        self.X_train = self._convertData_(self.X_train)
        self._generateTags_()
        print "tags count: ", len(self.tags)
        print "words count: ", len(self.vocab)
        print "questions count: ", len(self.X_train)
        print "time complexity: ", len(self.tags) * len(self.vocab) * len(self.X_train)
        print "Parsed the training data"
        #for word in self.vocab:
            #print word

    def _createVocabulary_(self):
        """
        Create a vocabulary (dictionary) for all the words appeared.
        Used for multinomial Naive Bayes.
        
        NOTE: Vocabulary doens't have to be generated this way.
            Can come up with better vocabulary. (say filter out some words)
        """
        vocab = set()
        for x in self.X_train:
            # split text using regex, split on characters other than [\w']
            for word in re.findall(r"[\w']+", x):
                word = util.purify(word)
                vocab.add(word)
        vocab.remove("")
        # remove one-character word
        vocab = [word for word in vocab if len(word) > 1]
        stopwords_list = util.readinStopwords()
        vocab = util.removeStopwords(vocab, stopwords_list)
        self.vocab = dict(zip(vocab, range(len(vocab))))

    def _generateTags_(self):
        """
        Generate a set of tags from Y_train.
        """
        self.tags = list(set([tag for l in self.Y_train for tag in l]))
    
    def _convertData_(self, X):
        """
        converts X_train to a list of lists. Each row represents a question,
        and is composed of word frequencies

        Can be used for X_train or X_test
        """
        # each row contains a dictionary of word frequencies.
        rows = []
        for x in X:
            row = [0] * len(self.vocab)
            for word in re.findall(r"[\w']+", x):
                word = util.purify(word)
                if not self.vocab.has_key(word):
                    continue
                row[self.vocab[word]] += 1
            rows.append(row)

        return rows

    def _trainOnOneTag_(self, tag):
        """
        train on one tag
        """
        num_questions = len(self.X_train)
        num_words = len(self.vocab)

        # phi_k_d: phi_k denominator
        # phi_k_n: phi_k numerator

        # laplace smoothing
        phi_k_d = np.ones(2) * num_words
        phi_k_n = np.ones((2, num_words))
        
        phi_y = 0
        for i in range(num_questions):
            idx = int(tag in self.Y_train[i])
            phi_k_d[idx] += sum(self.X_train[i])
            for k in range(num_words):
                phi_k_n[idx, k] += self.X_train[i][k]
            phi_y += idx

        phi_y /= float(num_questions)

        phi_k = np.zeros((2, num_words))
        for i in range(2):
            for k in range(num_words):
                phi_k[i, k] = phi_k_n[i, k] / phi_k_d[i]

        return phi_k, phi_y

    def _trainOnNumOfTags_(self):
        """
        use linear regression to predict number of tags
        """
        Y_num_tags = [len(y) for y in self.Y_train]
        
        X_clip = []
        for question in self.X_train:
            row = [int(num > 0) for num in question]
            X_clip.append(row)

        X = np.vstack([np.ones(len(X_clip)), np.array(X_clip).T]).T
        self.theta = np.matrix(np.linalg.lstsq(X, Y_num_tags)[0])

    def train(self):
        """
        train for all tags
        """
        self._trainOnNumOfTags_()

        self.phi_k_list = []
        self.phi_y_list = []
        for tag in self.tags:
            phi_k, phi_y = self._trainOnOneTag_(tag)
            self.phi_k_list.append(phi_k)
            self.phi_y_list.append(phi_y)

    def test(self, testingSet):
        """
        return a list of lists. Each row represents a question,
        and is composed of tags predicted.
        """
        X_test, Y_test, test_qids = util.mergeTitlesAndBodies(testingSet)
        X_test = self._convertData_(X_test)

        num_words = len(self.vocab)
        num_questions = len(X_test)


        numtags_err = 0
        accuracy = 0
        for i in range(num_questions):

            X_lr = copy.deepcopy(X_test[i])
            X_lr.insert(0, 1)
            X_lr = [int(num > 0) for num in X_lr]
            num_of_tags = int(round(np.dot(self.theta, X_lr)))
            num_of_tags = max(1, num_of_tags)
            numtags_err += (num_of_tags - len(Y_test[i])) ** 2
            if num_of_tags == len(Y_test[i]):
                accuracy += 1

            tag_prob = []
            # compute a probability for each tag
            for j in range(len(self.tags)):
                phi_k = self.phi_k_list[j]
                phi_y = self.phi_y_list[j]
                log_p_1 = np.log(phi_y)
                log_p_0 = np.log(1 - phi_y)
                for k in range(num_words):
                    log_p_1 += np.log(phi_k[1, k]) * X_test[i][k]
                    log_p_0 += np.log(phi_k[0, k]) * X_test[i][k]
                # need real probability instead of log_p_0 and log_p_1 comparison
                # use log because product is too small that can result in divide-by-zero
                tag_prob.append(1 / (1 + np.exp(log_p_0 - log_p_1)))

            tag_prob = sorted(dict(zip(range(len(tag_prob)), tag_prob)).items(),\
                              key=lambda x:x[1], reverse=True)
            
            #print test_qids[i], ' : ',
            #for j in range(5):
                #print self.tags[tag_prob[j][0]],
            #print 

        print numtags_err / float(num_questions)
        print accuracy / float(num_questions)
