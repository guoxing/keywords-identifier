#!/bin/sh

# run this script, then unarchive train_data/Train.zip and test_data/Test.zip
# then run generate_data_part_2.sh

mkdir train_data
mkdir test_data
curl --silent --output SampleSubmission.csv http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/download/SampleSubmission.csv
curl --silent --output Train.zip http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/download/Train.zip
curl --silent --output Test.zip http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/download/Test.zip
mv Train.zip train_data
mv Test.zip test_data
