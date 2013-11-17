#!/usr/bin/python

"""
Split data.

How to use:
    chmod 777 split.py
    ./split.py

What does it do:
    It outputs num_files files. Each file contains num_qs questions.
    (fetched from the start of input_file)

    Instead of breaking data by lines, this script ensures questions' integrity
    (no breaking questions at the beginning and end)

"""
num_qs = 50
num_files = 2
input_file = "../data/Train.csv"
output_prefix = "../data/out"

with open(input_file, "r") as csv:
    curr_qs = 0
    curr_files = 0
    out_f = open(output_prefix + str(curr_files), "w")
    for line in csv:
        if out_f.closed:
            out_f = open(output_prefix + str(curr_files), "w")
        out_f.write(line)
        if line.find('\r\n') >= 0:
            curr_qs += 1
            if curr_qs == num_qs:
                curr_qs = 0
                out_f.close()
                curr_files += 1
                if curr_files == num_files:
                    break
