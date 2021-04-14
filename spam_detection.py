import pandas as pd
import numpy as np
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#extracting CSV files
df1 = pd.read_csv("Youtube01-Psy.csv",encoding = "ISO-8859-1")
df2 = pd.read_csv("Youtube02-KatyPerry.csv",encoding = "ISO-8859-1")
df3 = pd.read_csv("Youtube03-LMFAO.csv",encoding = "ISO-8859-1")
df4 = pd.read_csv("Youtube04-Eminem.csv",encoding = "ISO-8859-1")
df5 = pd.read_csv("Youtube05-Shakira.csv",encoding = "ISO-8859-1")

#Combining all data
data = [df1,df2,df3,df4,df5]
total_df = pd.concat(data)
#Just keep the CONTENT and TAG columns, delete others
total_df.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1,inplace=True)

#Processing the comments in such a way that they contains only alphabets
def process_content(content):
    return " ".join(re.findall("[A-Za-z]+",content.lower()))

total_df['processed_content'] = total_df['CONTENT'].apply(process_content)
#The revised comments row is preserved by deleting the old CONTENT column
total_df.drop(['CONTENT'],axis=1,inplace=True)

#85% of the pre-processed data set was randomly allocated as training and 15% as test.
concent_train, concent_test, class_train, class_test = train_test_split(total_df['processed_content'],total_df['CLASS'],test_size=0.15,random_state=57)

#It builds a dictionary of features and transform documents to feature vectors.
count_vect = CountVectorizer()
con_train_counts = count_vect.fit_transform(concent_train)
con_test_counts = count_vect.transform(concent_test)

tfidf_transformer = TfidfTransformer()
con_train_tfidf = tfidf_transformer.fit_transform(con_train_counts)
con_test_tfidf = tfidf_transformer.transform(con_test_counts)

#Naive Bayessian Model-------------------------------------------------------------
model_NB = MultinomialNB()
model_NB.fit(con_train_tfidf,class_train)
predicted_NB = model_NB.predict(con_test_tfidf)

#success criterion:
f1 = f1_score(class_test, predicted_NB)
print('NB_f1score = ', f1)


#SVM Model-------------------------------------------------------------------------
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='modified_huber', penalty='l2', 
                     alpha=1e-5, max_iter=20, random_state=42))])

text_clf_svm.fit(concent_train,class_train)
predicted_svm = text_clf_svm.predict(concent_test)

#success criterion:
f1 = f1_score(class_test, predicted_svm)
print('svm_f1score = ', f1)

#Logistic Regression Model-------------------------------------------------------
model_log = LogisticRegression()
model_log.fit(con_train_tfidf,class_train)
predictions = model_log.predict(con_test_tfidf)

#success criterion:
f1 = f1_score(class_test, predictions)
print('log_f1score = ', f1)

#Random Forest Classifier Model--------------------------------------------------
model_forest = RandomForestClassifier(n_estimators = 20, random_state = 0)
model_forest.fit(con_train_tfidf,class_train)
predictions = model_forest.predict(con_test_tfidf)

#success criterion:
f1 = f1_score(class_test, predictions)
print('forest_f1score = ', f1)

