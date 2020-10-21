from textblob import TextBlob

//scaffold only
// string to be generated from twitter api

str1 = 'i would hate you but you are my everything'
str2 = 'i love you, you are so good, marvellous excellent for me, bravo, yes, wow ,amazingly'
blob1 = TextBlob(str2)
print(blob1.sentiment)
