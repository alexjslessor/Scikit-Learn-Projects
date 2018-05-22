import pandas as pd
import re, string, nltk, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
pd.set_option('display.max_colwidth', 180)
os.chdir(r'C:\Users\Dropbox\SentimentAnalysis\03_Reddit\RedditSarcasmTraininData\PrincetonData')

def timefunc(func):
	def f(*args, **kwargs):
		start = time()
		rv = func(*args, **kwargs)
		finish = time()
		print('Run time is.', finish - start)
		return rv
	return f

def clean_text(text):
	import nltk
	import string
	stopwords = nltk.corpus.stopwords.words('english')
	# ps = nltk.PorterStemmer()
	wn = nltk.WordNetLemmatizer()
	text = "".join([word for word in text if word not in string.punctuation])
	tokens = re.split('\W+', text)
	# wn.lemmatize(word)
	# ps.stem(word)
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

tfidf = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf.fit_transform(dfblob['body'])
print(x_tfidf.shape)
print(tfidf.get_feature_names())

x_features = pd.concat([dfblob['comment_length'], dfblob['punct%'], pd.DataFrame(x_tfidf.toarray())], axis=1)
print(x_features.head())

# Starting to build ML here
rf = RandomForestClassifier(n_jobs=-1)
k_fold = KFold(n_splits=5)
cross_val_score(rf, x_features, dfblob['sarcasm'], cv=k_fold, scoring='accuracy', n_jobs=-1)
# End Here

#################















