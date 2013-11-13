tagger
======

cs221 final project


Experiments we ran
===================
Each table contains the percentage of questions
that have correct tag percentages in the categories
0, 0-25, 25-50, 50-75, and 75-100
(see printResultsTable)

- SVM vs Naive Bayes using unigrams

SVM
--------------
0:       89.11
1-25:    1.047
25-50:   5.443
50-75:   3.350
75-100:  1.047
--------------

Naive Bayes
--------------


- data size
- n-grams
- question contains code markup (blockquote, LaTeX, grid of data)
- If there’s code, classify its language based on some library
- question contains latex markup
- unigrams (this is a huge one)
- bigrams
- trigrams?
- length of description
- length of question
- Is there a high frequency of one pronoun?
- questions contains numbers
- misspellings
- redundancy on question and description
- is some text in a foreign language
- poor grammar
- number of question marks
- confidence of asker (do they use weak phrases that inexperienced posters typically do, like “Any
- use of a TextBlock
- Use of blockquote
- human names (Jeff, Amanda)
- are there many other questions that may be the same as this one? It probably has a popular tag.
- is there emoji? :)
- are sentences capitalized?

Feature selection process
    We first use filter feature selection to compute a score
    for how likely each feature influences the output tag. The
    purpose of this selection method was to get a general sense
    of whether the feature could successfully inform the tags at
    least a little. After we eliminated all the features that
    had next to no effect on the tags, we’ll further prune our
    feature set by using forward feature search.















Dependencies
============
- scikit-learn

Note
====
A CS221 instructor said, "For the CS221 project, we
have strict policy that at most 2 students do the work.
You can share some of the code infrastructure with the
CS229 project, but will need to clearly separate an
interesting problem on which only two of you will be
working for the CS221 project."

After we try a few more classification schemes and test
some features, I will take a part of the
project to work on that will be separate from what
Ben, Guoxing, and I will be working on.

Papers
======
Here's the treasure trove: http://scholar.google.com/scholar?espv=210&es_sm=91&um=1&ie=UTF-8&lr&q=related:b9kLL_qDc8XssM:scholar.google.com/
http://arxiv.org/pdf/1202.4063.pdf
http://arxiv.org/pdf/1208.3623v1.pdf
really good for background on SVM and text classification: http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf_
on kNN: http://clair.si.umich.edu/~radev/papers/tc.pdf

Set up
======
Download the data and split it.
sudo pip install -U scikit-learn
git clone https://github.com/matplotlib/matplotlib.git
cd matplotlib
python setup.py build
python setup.py install

Testing and Continuous Integration
==================================
When changes are pushed to Github, the new code is tested on the complete training and test data sets.

tagger CI (Jenkins):
http://ci.perfmode.com/job/tagger/

TODO(roseperrone): grant repo access to Jenkins server

What we need to figure out
==========================
How to determine how many tags to give a question.
    - there needs to be at least one
    - learn the variance of the number of tags
    - learn what makes a question have many tags.
        - e.g. History questions only have 1-3 tags,
               but coding questions have 4-7
    - learn what makes a question have more tags. Should it
      simply be the tags that pass a certain threshold?

I split up the training data using
split -l 355100 Train.csv

I split up the testing set using
split -l 100000 Test.csv

Note that the competitive success of our algorithm will
likely depend most on how we can apply domain knowledge
to select our features. For example, we may want to predict
how many tags a question has, and use that to inform
the threshold for labeling.

Features in order of how much we estimate the feature will contribute to classification:
e.g.
    - question contains code markup
    - If there’s code, classify its language based on some library
    - question contains latex markup
    - unigrams (this is a huge one)
    - bigrams
    - trigrams?
    - length of description
    - length of question
    - Is there a high frequency of one pronoun?
    - questions contains numbers
    - misspellings
    - redundancy on question and description
    - is some text in a foreign language
    - poor grammar
    - number of question marks
    - confidence of asker (do they use weak phrases that inexperienced posters typically do, like “Any help would be appreciate(d)” and “My Question:”, “Thank you.”, “Thanks for any ideas”, “I am clueless”
    - use of a TextBlock
    - Use of blockquote
    - human names (Jeff, Amanda)
    - are there many other questions that may be the same as this one? It probably has a popular tag.
    - is there emoji? :)
    - are sentences capitalized?
