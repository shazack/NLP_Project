import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    return sentences

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = BeautifulSoup(review,"html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter = counter + 1.
    return reviewFeatureVecs

def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews

if __name__ == '__main__':
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )
    test=train[18750:25000]
    train=train[:18750]

    print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size ))
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = [] 
    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    num_features = 300    
    min_word_count = 40   
    num_workers = 4       
    context = 10          
    downsampling = 1e-3   
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)
    model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print("Creating average feature vecs for training reviews")
    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
    print("Creating average feature vecs for test reviews")
    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )
    forest = RandomForestClassifier( n_estimators = 10 )
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit( trainDataVecs, train["sentiment"] )
    result = forest.predict( testDataVecs )
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
    print("Wrote Word2Vec_AverageVectors.csv")

    actual=test["sentiment"]
    predicted=output["sentiment"]
    true=0.0
    total=0.0
    for i in range(18750,25000):
        if(actual[i]==predicted[i]):
            true+=1.0
        total+=1.0
    print("Accuracy is "+str(true/total))