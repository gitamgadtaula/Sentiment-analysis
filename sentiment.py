from textblob import TextBlob


//testing random string sentiment, later on a twitter string api will be generated and calculated for sentiments

str1 = 'i would hate you but you are my everything'
str2 = 'i love you, you are so good, marvellous excellent for me, bravo, yes, wow ,amazingly'
blob1 = TextBlob(str2)
print(blob1.sentiment)
