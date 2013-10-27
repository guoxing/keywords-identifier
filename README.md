tagger
======

cs221 final project

the project proposal will go here.
For now it's in a google doc.

Dependencies
============
- scikit-learn

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
http://i12a.com:8080/job/tagger/

TODO(roseperrone): grant repo access to Jenkins server

What we need to figure out
==========================
How to determine how many tags to give a question.
    - there needs to be at least one
    - learn the distribution of the number of tags
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
