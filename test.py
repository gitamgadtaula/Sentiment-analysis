import numpy as np  # for linear algebra
import pandas as pd
import sklearn  # for dataprocessing; csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.linear_model import LogisticRegression
import pdb
# form sklearn.linear_model import
from sklearn import tree

dataset = pd.read_csv('combined_news.csv')
# temp=dataset.head(3)
# for i in range(2, 26):
# print('top'+str(i))
# dataset.pop('Top'+str(i))
# print(dataset.head())
news = dataset['Top1'].values
labels = dataset['Label'].values



# for i in range(2, 23):
#     # merging all feature columns
#     col = 'Top'+str(i)
#     news = news + dataset[col].values
#     labels += labels



news_train, news_test, y_train, y_test = train_test_split(
    news, labels, test_size=0.05, random_state=1000)

punctuations = string.punctuation
parser = English()
stopwords = list(STOP_WORDS)


# Using spaCy to Create a Custom Tokenizer (Optional)

def spacy_tokenizer(utterance):
    tokens = parser(utterance)
    return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip() not in stopwords and token.text not in punctuations]


# Transforming Text into Numerical Feature Vectors
vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
# By default, the vectorizer might be created as follows:
#vectorizer = CountVectorizer()
vectorizer.fit(news_train)


X_train = vectorizer.transform(news_train)
X_test = vectorizer.transform(news_test)

classifier = LogisticRegression()
# classifier = tree.DecisionTreeClassifier()
# >>> clf = clf.fit(X, Y)
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print("Accuracy: ", accuracy)

new_news = ['Old version of python useless',
            'Very good effort, but not five stars', 'crashed']
X_new = vectorizer.transform(new_news)
print(
    classifier.predict(X_new)
)
