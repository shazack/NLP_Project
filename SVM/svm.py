import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from bs4 import BeautifulSoup
import pandas as pd
import re
from nltk.corpus import stopwords
import os

def review_to_wordlist( review):
    review_text = BeautifulSoup(review,"html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return(words)

if __name__ == "__main__":
	trainDataPath=os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv')
	testDataPath=os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv')

	train = pd.read_csv(trainDataPath, header=0, delimiter="\t", quoting=3)
	#test = pd.read_csv(testDataPath, header=0, delimiter="\t", quoting=3 )
	test=train[18750:25000]
	train=train[:18750]

	clean_train_reviews = []
	for i in range( 0, len(train["review"])):
		clean_train_reviews.append(" ".join(review_to_wordlist(train["review"][i])))
	print("cleaned train data")

	clean_test_reviews = []
	for i in range( 18750, 25000):
		clean_test_reviews.append(" ".join(review_to_wordlist(test["review"][i])))
	print("cleaned test data")

	m_f = 100
	
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = m_f)
	train_data_features = vectorizer.fit_transform(clean_train_reviews)
	train_data_features = train_data_features.toarray()
	print("tokens for train made")
	
	vecTest = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = m_f) 
	test_data_features = vecTest.fit_transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()
	print("tokens for test made")

	kernel="rbf"
	model = SVC(kernel=kernel, gamma='auto') 
	model = model.fit( train_data_features, train["sentiment"] )
	print("model made")

	result = model.predict(test_data_features)
	print("predictions done")

	output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
	output.to_csv("outputSVM.csv", index=False, quoting=3 )
	print("output file saved as outputSVM.csv")

	actual=test["sentiment"]
	predicted=output["sentiment"]
	true=0.0
	total=0.0
	for i in range(18750,25000):
		if(actual[i]==predicted[i]):
			true+=1.0
		total+=1.0
	print("Accuracy is "+str(true/total))