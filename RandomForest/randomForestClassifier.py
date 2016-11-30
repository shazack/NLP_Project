import os
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review,"html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    #test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    test=train[18750:25000]
    train=train[:18750]
    
    clean_train_reviews = []
    print "Cleaning and parsing the training set movie reviews...\n"
    for i in xrange( 0, len(train["review"])):
        clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i], True)))
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit( train_data_features, train["sentiment"] )

    clean_test_reviews = []
    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(18750,25000):
        clean_test_reviews.append(" ".join(review_to_wordlist(test["review"][i], True)))
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    result = forest.predict(test_data_features)
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"

    actual=test["sentiment"]
    predicted=output["sentiment"]
    true=0.0
    total=0.0
    for i in range(18750,25000):
        if(actual[i]==predicted[i]):
            true+=1.0
        total+=1.0
    print("Accuracy is "+str(true/total))
