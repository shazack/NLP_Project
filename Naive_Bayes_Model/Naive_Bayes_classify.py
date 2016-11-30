import os
import sys
import math
import re
import operator
import numpy
import functools
import math
import string
from collections import defaultdict
import collections
#from random import sample
import json
import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords


def read_all_data(file):
    temp_list = []
    onlyfiles = []
    test = pd.read_csv("/Users/nisharazack/Documents/Bag of words/labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)

    # print(len(train))
    for i in range(len(file)):
        clean_review = review_to_words(test["review"][i])
        id = test["id"][i].replace('"', '').strip()
        temp_list = [id, clean_review, test["sentiment"][i]]
        onlyfiles.append(temp_list)
    return onlyfiles

def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(meaningful_words))

# count the total number of files - train

test = pd.read_csv("/Users/nisharazack/Documents/Bag of words/testData.tsv", header=0, \
                        delimiter="\t", quoting=3)

totalfiles = read_all_data(test)
print(totalfiles[0],totalfiles[67])
count_totalfiles = len(totalfiles)


with open("nboutput.txt", 'w'):
    pass

prior_list = []

with open("nbmodel.txt") as file:
    lines = file.read().split("\n")
    for i in lines:
        each = i.split(":")
        prior_list.append(each)

pos_dict = defaultdict(float)
neg_dict = defaultdict(float)
all_dict = defaultdict(list)

for i in prior_list:
    if i[0] == "postotal":
        prob_pos = i[1]
    if i[0] == "negtotal":
        prob_neg = i[1]
    if i[0] == "pos":
        if len(i[2]) > 0:
            pos_dict[i[1]] = i[2]
    if i[0] == "neg":
        if len(i[2]) > 0:
            neg_dict[i[1]] = i[2]
    if i[0] == "pos" or i[0] == "neg":
        all_dict[i[1]] = i[2], i[0]

result = 0.0
tokens = []
prob_word_list = []
results_dict = defaultdict(str)
message_given_neg = 0.0
message_given_pos = 0.0
delimiters = ['\n', ' ', ',', '.', '?', '!', ':', '-', ')', '(', '$']
for i in totalfiles:
    tokens = []
    prob_word_list = []

    tokens = i[1].strip().split()

    for token in tokens:
        # add the probability to list to multiply the independent probs only if its already present in the list
        if token in pos_dict:
            prob_word_list.append(float(pos_dict[token]))

    log_message_given_pos = sum([math.log(word) for word in prob_word_list])

    prob_word_list = []

    for token in tokens:
        # add the probability to list to multiply the independent probs only if its already present in the list
        if token in neg_dict:
            if len(neg_dict[token]) > 0:
                prob_word_list.append((float(neg_dict[token])))

    log_message_given_neg = sum([math.log(word) for word in prob_word_list])

    log_prob_pos = math.log(float(prob_pos))
    log_prob_neg = math.log(float(prob_neg))

    log_result_pos = log_prob_pos + log_message_given_pos
    log_result_neg = log_prob_neg + log_message_given_neg

    with open("nboutput.txt", "a+") as f:
        if log_result_pos >= log_result_neg:
            f.write("pos" + " " + str(i[0]))
            results_dict[i[0]] = "pos"
        else:
            f.write("neg" + " " + str(i[0]))
            results_dict[i[0]] = "neg"
        f.write("\n")
# prob(pos | message ) = prob_pos* message_given_pos / prob_pos * message_given_pos + prob_neg *message_given_neg

orig_dict = defaultdict(str)
print(results_dict)
for i in totalfiles:
    if i[2] == 1:
        orig_dict[i[0]] = "pos"
    elif i[2] == 0:
        orig_dict[i[0]] = "neg"

count = 0
cpos = 0
cneg = 0
for k, v in results_dict.items():
    if k in orig_dict:
        if v == orig_dict[k]:
            count += 1
        if v == "pos":
            cpos += 1
        if v == "neg":
            cneg += 1

accuracy = float(count) / float(len(results_dict))
correctpos = 0
correctneg = 0
totpos = 0
totneg = 0

for k, v in results_dict.items():
    if k in orig_dict:
        if v == orig_dict[k] and v == "pos":
            correctpos += 1
        if v == orig_dict[k] and v == "neg":
            correctneg += 1

for k, v in orig_dict.items():
    if v == "pos":
        totpos += 1
    if v == "neg":
        totneg += 1

print(accuracy)

# calculate precision, recall and recall

#
# preneg = correctneg / cneg
# prepos = correctpos / cpos
#
# recallneg = correctneg / totneg
# recallpos = correctpos / totpos
#
# f1neg = 2 * ((preneg * recallneg) / (preneg + recallneg))
# f1pos = 2 * ((prepos * recallpos) / (prepos + recallpos))

# prepos = round(prepos,2)
# preneg = round(preneg,2)
# recallneg = round(recallneg,2)
# recallpos = round(recallpos,2)
# f1neg = round(recallneg,2)
# f1pos = round(recallpos,2)
# accuracy = round(accuracy,2)
#
# print("Precision, Recall, F1")
# print("neg")
# print(preneg, recallneg, f1neg)
# print("pos")
# print(prepos, recallpos, f1pos)
# print("Weighted avg")
# print(accuracy)
#

