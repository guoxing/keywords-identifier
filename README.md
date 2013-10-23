tagger
======

cs221 final project

the project proposal will go here.
For now it's in a google doc.

I split up the training data using
split -l 355100 Train.csv

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
