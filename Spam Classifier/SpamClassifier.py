# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:50:29 2024

@author: Shrinil
"""

import pandas as pd
messages=pd.read_csv('data/spam.csv',encoding='cp1252')
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['v2'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()#using bag of words

Y=pd.get_dummies(messages['v1'])
Y=Y.iloc[:,1]

#perform train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB()
spam_detect_model.fit(X_train,Y_train)
Y_predict=spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,Y_predict)
print(score)

predictions_df=pd.DataFrame(Y_predict,columns=['Predictions'])
predictions_df.to_csv('text_spam_predictions.csv',index=False)