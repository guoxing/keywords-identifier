#!/bin/sh

rm -f train_data/Train.zip
rm -f test_data/Test.zip

split -l 355100 train_data/Train.csv
