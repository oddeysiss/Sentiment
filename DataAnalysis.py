# This script describes the word/sentence count of the processed ISEAR datafile and
# outputs a plot of words most strongly associated with each labeled emotion.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

filename = 'Processed_ISEAR'
df = pd.read_csv('Processed_ISEAR.csv', encoding='latin1')

# Setting up path to save plots
basepath = os.path.dirname('__file__')
plotpath = os.path.abspath(os.path.join(basepath, 'FeaturePlots'))

# Get an idea of avg word / sentence count in data
df['words'] = df['Story'].apply(lambda line: len(line.split()))
df['sentences'] = df['Story'].apply(lambda line: len(line.split('.')))

print('Analysis of file: ', filename)
print('Average word count    :', np.average(df['words']))
print('Average sentence count:', np.average(df['sentences']))

# I want to see what the top indicator words are for each emotion.
# To do this, I need to split the data, build a vectorizer, and build a classifier first.
X, y = df['Story'], df['Emotion']
text_train, text_test, y_train, y_test = train_test_split(X, y, train_size=0.8, 
	stratify=y, random_state=0)

tfidfvect = TfidfVectorizer(ngram_range=(1, 3))
tfidfvect.fit(text_train)
X_train, X_test = tfidfvect.transform(text_train), tfidfvect.transform(text_test)

clf = LinearSVC(C=1)
clf.fit(X_train, y_train)

print()
print('Using LinearSVC classifier with C = 1')
print('Classifier training score: ', clf.score(X_train, y_train))
print('Classifier test score    : ', clf.score(X_test, y_test))


# Visualization of important features (words) for each emotion
def visualize_coefficients(clf, feature_names, class_labels, n_top_features=15):
	for i, class_label in enumerate(sorted(class_labels)):
		n_top = np.argsort(clf.coef_[i])[-n_top_features:]
		plt.bar(np.arange(n_top_features), clf.coef_[i][n_top])
		#print(feature_names[[1, 2, 3]])
		plt.xticks(np.arange(n_top_features), [feature_names[j] for j in n_top],
		 rotation=60, ha='right')
		plt.xlabel('Features')
		plt.ylabel('Tfidf Score')
		plt.title('Important Features for Label: %s' % (class_label))
		plt.savefig(os.path.abspath(os.path.join(plotpath, 
			'Important_Features_%s' % (class_label))), bbox_inches='tight')
		plt.clf()

visualize_coefficients(clf, tfidfvect.get_feature_names(), y_train.unique())