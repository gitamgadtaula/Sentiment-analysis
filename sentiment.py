from textblob import TextBlob

# # //scaffold only
# # // string to be generated from twitter api

# str1 = 'i would hate you but you are my everything'
# str2 = 'i love you, you are so good, marvellous excellent for me, bravo, yes, wow ,amazingly'
# blob1 = TextBlob(str2)
# print(blob1.sentiment)

import tweepy
consumer_key="QUwsNYdC2UQScynZeB0MTqEUb"
consumer_secret="r0nVOCL0knw0e3M8xVnpJLENsWdZVZUJXwz5rFwpFSq9bF0v74"
access_token="457289895-8LpGsq0X4tfmKx2T2Z5q26Dob5bIyT1NypmFf0aU"
access_token_secret="3vOJyuCftDeiMAWl7zdmExy9nDG7ZFc6rgatS5D19s2D0"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# public_tweets = api.home_timeline()

search_results = api.search_tweets(q="nepse", count=10)

def sentiment_scores(blobText) :
    if blobText >= 0.05 :
        print("Positive !! BUYY") 
  
    elif blobText <= - 0.05 : 
        print("Negative !! SELL") 
  
    else : 
        print("Neutral !! HODL")
        
for tweet in search_results:
    blob=TextBlob(tweet.text)
    print(tweet.text)
    print("polarity ",blob.polarity)
    sentiment_scores(blob.polarity)


