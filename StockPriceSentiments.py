"""
Project Name: Sentiment analysis for stock prices
Project Description: This is a project where we are taking input data

By Harsh Kumar Dewangan
"""

# importing the packages
import pandas as pd
import numpy as np

desired_width=800
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

# load the dataset
df = pd.read_csv('Data.csv', encoding='ISO-8859-1')
#print(df.head(5))

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Data preprocessing
# Since dataset contains may UTF-8 characters, we should remove them
# Removing punctuation marks
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Renaming column names for ease of access
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
#print(data.head(5))

# Converting headlines to lower case
for index in new_Index:
    data[index] = data[index].str.lower()
#print(data.head(1))

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
#print(headlines[0])


## CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

## implement BAG OF WORDS
countvector = CountVectorizer(ngram_range = (2,2))
traindataset = countvector.fit_transform(headlines)
#print(traindataset[0])

# implement RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
#print(predictions)

# analyzing dataset accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
score = accuracy_score(test["Label"], predictions)
print("CountVectorizer:", score)
report = classification_report(test['Label'], predictions)
#print(report)


## TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

## implement BAG OF WORDS
tfvector = TfidfVectorizer(ngram_range = (2,2))
traindataset = tfvector.fit_transform(headlines)
#print(traindataset[0])

# implement RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])

## Predict for the Test Dataset
test_transform= []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = tfvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)
#print(predictions)

# analyzing dataset accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
score = accuracy_score(test["Label"], predictions)
print("TfidfVectorizer:", score)
report = classification_report(test['Label'], predictions)
#print(report)


## Naive Byes
from sklearn.naive_bayes import MultinomialNB

## implement BAG OF WORDS
naive = MultinomialNB()
naive.fit(traindataset, train['Label'])
#print(traindataset[0])

## Predict for the Test Dataset
test_transform= []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = tfvector.transform(test_transform)
predictions = naive.predict(test_dataset)
#print(predictions)

# analyzing dataset accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
score = accuracy_score(test["Label"], predictions)
print("Naive_Byes:", score)
report = classification_report(test['Label'], predictions)
#print(report)
