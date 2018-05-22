import os, re, string, nltk, logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', 180)
os.chdir(r'C:\Users\Dropbox\SentimentAnalysis\03_Reddit\RedditSarcasmTraininData\PrincetonData')
logging.basicConfig(filename='redditForestClassifier.log', level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def clean_text(text):
	import nltk
	import string
	stopwords = nltk.corpus.stopwords.words('english')
	ps = nltk.PorterStemmer()
	wn = nltk.WordNetLemmatizer()
	text = "".join([word for word in text if word not in string.punctuation])
	tokens = re.split('\W+', text)
	wn.lemmatize(word)
	ps.stem(word)
	text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
	return text

def count_punct(text):
	import string
	count = sum([1 for char in text if char in string.punctuation])
	return round(count/(len(text) - text.count(" ")), 3)*100


dfblob = pd.read_table('test-balanced_pol0.csv', encoding='latin2', header=None)
dfblob['text'] = dfblob[[1, 9]].astype(str).apply(lambda x: ''.join(x), axis=1)
dfblob = dfblob.drop([1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)
dfblob.columns = ['sarcasm', 'body']

dfblob['punct%'] = dfblob['body'].apply(lambda x: count_punct(x))

dfblob['comment_length'] = dfblob['body'].apply(lambda x: len(x) - x.count(" "))

# tfidf for the decision tree
tfidf = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf.fit_transform(dfblob['body'])
print(x_tfidf.shape)
feature_names = tfidf.get_feature_names()
print(type(feature_names))

x_features = pd.concat([dfblob['comment_length'], dfblob['punct%'], pd.DataFrame(x_tfidf.toarray())], axis=1)
print(x_features.head())


X_train, X_test, y_train, y_test = train_test_split(x_features, dfblob['sarcasm'], test_size=0.2)

model = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1).fit(X_train, y_train)
print(model.feature_importances_.argsort())

def fit_classifier():

	joblib.dump(model, 'test1.pkl')
	y_pred = model.predict(X_test)
	precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')

	logging.info('Precision:{}/Recall:{}/Accuracy:{}'.format(round(precision, 3),round(recall, 3),round((y_pred==y_test).sum() / len(y_pred),3)))


def tree_feature_importances(file):
	feature_labels = np.array(feature_names)

	load_model = joblib.load(file)
	importance = load_model.feature_importances_
	features = importance.argsort()
	# Print each feature label, from most important to least important (reverse order)
	for index in features:
		print("{} - {:.2f}%".format(feature_labels[index], (importance[index] * 100.0)))

tree_feature_importances('test1.pkl')




































































