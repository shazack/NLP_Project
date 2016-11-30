import os
import collections
import sys
#from random import sample
import random
from collections import defaultdict
import json
from collections import Counter
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
    # print(len(train))
    for i in range(len(file)):
        clean_review = review_to_words(train["review"][i])
        id = train["id"][i].replace('"', '').strip()
        temp_list = [id, clean_review, train["sentiment"][i]]
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

train = pd.read_csv("/Users/nisharazack/Documents/Bag of words/labeledTrainData.tsv", header=0, \
                        delimiter="\t", quoting=3)
totalfiles = read_all_data(train)
totalfiles =totalfiles[:18750]
print(len(totalfiles))
print(totalfiles[0],totalfiles[67])
count_totalfiles = len(totalfiles)


#count number of pos files

pos_files = []
neg_files = []
for i in totalfiles:
    if i[2] == 1:
        pos_files.append(i[1])

    elif i[2] == 0:
        neg_files.append(i[1])

count_pos = len(pos_files)
count_neg = len(neg_files)

print(count_pos)
print(count_neg)
# calculate prob of pos and neg
prob_pos = float(count_pos) / float(count_totalfiles)
prob_neg = float(count_neg) / float(count_totalfiles)
print("prob of neg")
print(prob_neg)
print(prob_pos)
# calculate all the probabilities within the pos folder

pos_freq = {}
neg_freq = {}
total_freq = {}
count = 0
pos_count = 0
pos_freq = collections.Counter()
for i in pos_files:
    pos_freq.update(i.split())

neg_freq = collections.Counter()
for i in neg_files:
    neg_freq.update(i.split())



for i in totalfiles:
    tokenlist = i[1].split()
    for token in tokenlist:
        if token not in total_freq:
            total_freq[token] = 1
        else:
            total_freq[token] += 1

pos_count = sum(pos_freq.values())
neg_count = sum(neg_freq.values())

prob_neg_words = {}
prob_pos_words = {}

print("distinct words")
# print(total_freq)
# print(len(total_freq))



with open("nbmodel_3.txt","w") as f1:
    f1.write("postotal"+":"+str(prob_pos))
    f1.write("\n")
    f1.write("negtotal"+":"+str(prob_neg))
    f1.write("\n")

    for word, freq in total_freq.items():
        if word in neg_freq:
            prob_word_neg = (float(neg_freq[word]) + 1.0) / (float(neg_count) + float(len(total_freq)))
            f1.write("neg" + ":" + str(word) + ":" + str(prob_word_neg))
            f1.write("\n")
        else:
            prob_word_neg = (1 / (float(neg_count) + float(len(total_freq))))
            f1.write("neg" + ":" + str(word) + ":" + str(prob_word_neg))
            f1.write("\n")

        if word in pos_freq:
            prob_word_pos = (float(pos_freq[word]) + 1.0) / (float(pos_count) + float(len(total_freq)))
            f1.write("pos" + ":" + str(word) + ":" + str(prob_word_pos))
            f1.write("\n")
        else:
            prob_word_pos = (1 / (float(pos_count) + float(len(total_freq))))
            f1.write("pos" + ":" + str(word) + ":" + str(prob_word_pos))
            f1.write("\n")

