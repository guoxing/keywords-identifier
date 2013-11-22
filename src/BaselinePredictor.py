import util
import collections
import random

class BaselinePredictor:

    def __init__(self, trainingSet, numTags):
        self.numTags = numTags
        self.X_train, self.Y_train, self.qids = util.mergeTitlesAndBodies(trainingSet)

    def train(self):
        tags_count, total_count = util.getCountsOfTags(self.Y_train)
        self.tags = tags_count[:100]
        total_count = sum([ct for tag, ct in self.tags])
        self.tags = [(tag, ct / float(total_count)) for tag, ct in self.tags]
        self.num_tags = collections.defaultdict(lambda: 0)
        num_total_tags = 0
        for tags in self.Y_train:
            self.num_tags[len(tags)] += 1
            num_total_tags += 1

        for num_tag, ct in self.num_tags.items():
            self.num_tags[num_tag] =  ct / float(num_total_tags)

    def test(self, testingSet):
        X_test, Y_test, test_qids = util.mergeTitlesAndBodies(testingSet)
        
        numtags_err = 0
        numtags_accuracy = 0
        false_negative_5 = 0
        false_positive_5 = 0
        false_positive = 0
        false_negative = 0
        num_tested_questions = 0
        total_tags = 0

        #tags_counter_predict_5 = collections.defaultdict(lambda: 0)

        tag_set = set([tag for tag, prob in self.tags])
        for y_test in Y_test:
            if not set(y_test).issubset(tag_set):
                continue
            
            num_tested_questions += 1
            total_tags += len(y_test)

            rand = random.random()
            for num_tag, prob in self.num_tags.items():
                if rand > prob:
                    rand -= prob
                else:
                    num_of_tags = num_tag
                    break
            numtags_err += (num_of_tags - len(y_test)) ** 2
            if num_of_tags == len(y_test):
                numtags_accuracy += 1

            predict_5_tags = []
            predict_tags = []
            for i in range(max(5, num_of_tags)):
                rand = random.random()
                for tag, prob in self.tags:
                    if rand > prob:
                        rand -= prob
                    else:
                        curr_tag = tag
                        break
                if i < 5:
                    predict_5_tags.append(curr_tag)
                if i < num_of_tags:
                    predict_tags.append(curr_tag)
            for real_tag in y_test:
                if real_tag not in predict_5_tags:
                    false_negative_5 += 1
                if real_tag not in predict_tags:
                    false_negative += 1
            for predict_tag in predict_5_tags:
                if predict_tag not in y_test:
                    false_positive_5 += 1

            for predict_tag in predict_tags:
                if predict_tag not in y_test:
                    false_positive += 1

            #for predict_tag in predict_5_tags:
                #tags_counter_predict_5[predict_tag] += 1

        #total_count_predict_5 = sum(tags_counter_predict_5.values())
        #tags_counter_predict_5 = sorted(tags_counter_predict_5.items(), key=lambda x:x[1], reverse=1)
        #for tag, count in tags_counter_predict_5:
            #print tag, count / float(total_count_predict_5)

        print "#questions tested: ", num_tested_questions
        print "#tags square error : ", numtags_err / float(num_tested_questions)
        print "#tags accuracy : ", numtags_accuracy / float(num_tested_questions)
        print "precision (5): ", (total_tags - false_negative_5) / float(total_tags - false_negative_5 + false_positive_5)
        print "recall (5): ", (total_tags - false_negative_5) / float(total_tags)
        print "precision : ", (total_tags - false_negative) / float(total_tags - false_negative + false_positive)
        print "recall : ", (total_tags - false_negative) / float(total_tags)
