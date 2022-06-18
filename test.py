import tweepy
from textblob import TextBlob
import statistics

consumer_key = "QUwsNYdC2UQScynZeB0MTqEUb"
consumer_secret = "r0nVOCL0knw0e3M8xVnpJLENsWdZVZUJXwz5rFwpFSq9bF0v74"
access_token = "457289895-8LpGsq0X4tfmKx2T2Z5q26Dob5bIyT1NypmFf0aU"
access_token_secret = "3vOJyuCftDeiMAWl7zdmExy9nDG7ZFc6rgatS5D19s2D0"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# public_tweets = api.home_timeline()

print('Enter Stock to predict:')
x = input()

search_results = api.search_tweets(q=x, count=10)


def sentiment_scores(blobText):
    if blobText >= 0.05:
        print("Polarity: ", blobText, " Positive !! Buy" )

    elif blobText <= - 0.05:
        print("Polarity: ", blobText, " Negative !! SELL")

    else:
        print("Polarity: ", blobText, " Neutral !! HOLD")


polarityLists = []

for tweet in search_results:
    blob = TextBlob(tweet.text)
    polarityLists.append(blob.polarity)
    print(tweet.text)
    # print("polarity ",blob.polarity)

meanPolarity = statistics.mean(polarityLists)
print('______________________________________________')
sentiment_scores(meanPolarity)
