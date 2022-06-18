# import numpy as np  # for linear algebra
import pandas as pd
# import sklearn  # for dataprocessing; csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.linear_model import LogisticRegression
# import pdb
# form sklearn.linear_model import
import numpy as np
from sklearn import tree
import os  # for environment variables
import tweepy  # for twitter api
from dotenv import load_dotenv
load_dotenv()


# Getting environment variables of twitter
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

#  authenticating twitter api
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

dataset = pd.read_csv('combined_news.csv')
finalNews = dataset['Top1'].values
finalLabels = dataset['Label'].values
for i in range(2, 23):
    # merging all feature columns
    col = 'Top'+str(i)
    news = dataset[col].values
    finalNews = np.append(finalNews, news)
    finalLabels = np.append(finalLabels, dataset['Label'].values)


news_train, news_test, y_train, y_test = train_test_split(
    finalNews, finalLabels, test_size=0.2, random_state=1000)

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
# vectorizer = CountVectorizer()
vectorizer.fit(news_train)
X_train = vectorizer.transform(news_train)
X_test = vectorizer.transform(news_test)

classifier = LogisticRegression(max_iter=3000)
# classifier = tree.DecisionTreeClassifier()
# >>> clf = clf.fit(X, Y)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print("Accuracy: ", accuracy)
tweets = api.search_tweets(q='nepse', lang='en',
                           count=3, tweet_mode="extended")
search_results = [tweet.full_text
                  for tweet in tweets]  # mapping to fetch tweet.full_text only
print(search_results)
# search_results = ['Old version of python useless',
#                   'Very good effort, but not five stars', 'plane crashed']
X_new = vectorizer.transform(search_results)
print(
    classifier.predict(X_new)
)
