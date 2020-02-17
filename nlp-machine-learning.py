
# coding: utf-8

# In[178]:


########################
#NLP PROJECT WITH SVM, NAIVE BAYES, LOGISTIC REGRESSION#
########################


# In[38]:


# This code has followed https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/


# In[6]:


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# df = pd.read_excel('sentiment_emails.xlsx', sheet_name='Sheet1')
df = pd.read_excel('IMDB-Dataset.xlsx', sheet_name='Sheet1')
# print("Column headings:")
# print(df.columns)
# print (df[:10])


# In[7]:


# Once we have our data ready, it is time to do some preprocessing. We will focus on removing useless variance for our task at hand. First, we have to convert the labels from strings to binary values for our classifier:
df['label'] = df.label.map({'positive': 0, 'negative': 1})
# print (df[:10])


# In[8]:


# Second, convert all characters in the message to lower case:
df['message']  = df["message"].map(lambda x: x if type(x)!=str else x.lower())
# print (df[:10])


# In[9]:


# Third, remove any punctuation:

from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

df['message'] = df.message.apply(lambda x: strip_tags(x))

df['message'] = df.message.str.replace('[^\w\s]', '')
# print (df[:10])


# In[10]:


df['message'] = df['message'].apply(str)
# Now we can apply the tokenization:
print (df[:10])

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['message'] = df.message.apply(lambda x: clean_text(x))


# In[12]:


# Fourth, tokenize the messages into into single words using nltk. First, we have to import and download the tokenizer from the console:

import nltk
nltk.download()


# In[13]:


df['message'] = df['message'].apply(str)
# Now we can apply the tokenization:
# print (df[:10])

df['message'] = df['message'].apply(nltk.word_tokenize)
# print (df[:10])


# In[38]:


# # Fifth, we will perform some word stemming. The idea of stemming is to normalize our text for all variations of words carry the same meaning, regardless of the tense. One of the most popular stemming algorithms is the Porter Stemmer:

# from nltk.stem import PorterStemmer

# stemmer = PorterStemmer()

# df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[39]:


# print (df[:10])


# In[14]:


# Finally, we will transform the data into occurrences, which will be the features that we will feed into our model:

from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])


# In[15]:


# We could leave it as the simple word-count per message, but it is better to use Term Frequency Inverse Document Frequency, more known as tf-idf:

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)


# In[11]:


####################
#SPLITTING THE DATA#
####################


# In[24]:


# Now that we have performed feature extraction from our data, it is time to build our model. We will start by splitting our data into training and test sets:

from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.2, random_state=69)
X_train_val, X_test, y_train_val, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=1)

# X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=1)


# In[25]:


print (X_train_val.shape)
print (X_train.shape)
print (X_val.shape)
print (X_test.shape)


# In[46]:


######################
#VISUALIZING THE DATA#
######################


# In[47]:


# Splitting the dataset into train and test set
# data = df[['label','message']]
# train, test = train_test_split(data,test_size = 0.1)
# train_pos = train[ train['label'] == 'pos']
# train_pos = train_pos['message']
# train_neg = train[ train['label'] == 'neg']
# train_neg = train_neg['message']
# def wordcloud_draw(data, color = 'black'):
#     words = ' '.join(data)
#     cleaned_word = " ".join([word for word in words.split()
#                             if 'http' not in word
#                                 and not word.startswith('@')
#                                 and not word.startswith('#')
#                                 and word != 'RT'
#                             ])
#     wordcloud = WordCloud(stopwords=STOPWORDS,
#                       background_color=color,
#                       width=2500,
#                       height=2000
#                      ).generate(cleaned_word)
#     plt.figure(1,figsize=(13, 13))
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.show()


# In[48]:


####################
#TRAINING THE MODEL#
####################


# In[18]:


# Then, all that we have to do is initialize the Naive Bayes Classifier and fit the data. For text classification problems, the Multinomial Naive Bayes Classifier is well-suited:

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model1 = MultinomialNB()
model2 = LinearSVC()
model3 = LogisticRegression()
model4 = RandomForestClassifier()
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)


# In[19]:


######################
#CROSS VALIDATING THE MODEL#
######################

import numpy as np

result1 = model1.predict(X_val)
result2 = model2.predict(X_val)
result3 = model3.predict(X_val)
result4 = model4.predict(X_val)

print(np.mean(result1 == y_val))
print(np.mean(result2 == y_val))
print(np.mean(result3 == y_val))
print(np.mean(result4 == y_val))


# In[56]:


# 0.8538271604938271
# 0.8962962962962963
# 0.8908641975308642
# 0.7162962962962963


# In[20]:


######################
#TRAINING the model oN TRAIN+VAL DATA#
######################

model1.fit(X_train_val,y_train_val)
model2.fit(X_train_val,y_train_val)
model3.fit(X_train_val,y_train_val)
model4.fit(X_train_val,y_train_val)


# In[66]:


######################
#EVALUATING THE MODEL#
######################


# In[21]:


# Once we have put together our classifier, we can evaluate its performance in the testing set:

import numpy as np

result1 = model1.predict(X_test)
result2 = model2.predict(X_test)
result3 = model3.predict(X_test)
result4 = model4.predict(X_test)

print(np.mean(result1 == y_test))#NB
print(np.mean(result2 == y_test))#SVM
print(np.mean(result3 == y_test))# LR
print(np.mean(result4 == y_test))#Random
# Our simple Naive Bayes Classifier has 98.2% accuracy with this specific test set! But it is not enough by just providing the accuracy, since our dataset is imbalanced when it comes to the labels (86.6% legitimate in contrast to 13.4% spam).


# In[22]:


# 0.858
# 0.9036
# 0.893
# 0.7514


# In[57]:


# It could happen that our classifier is over-fitting the legitimate class while ignoring the spam class. To solve this uncertainty, let's have a look at the confusion matrix:

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, result1))
print(confusion_matrix(y_test, result2))
print(confusion_matrix(y_test, result3))
print(confusion_matrix(y_test, result4))
# The confusion_matrix method will print something like this:


# In[59]:


# [[2065  401]
#  [ 301 2233]]
# [[2254  212]
#  [ 245 2289]]
# [[2243  223]
#  [ 292 2242]]
# [[2038  428]
#  [ 971 1563]]


# In[83]:


from sklearn.metrics import f1_score
print(f1_score(y_test, result1, average='macro'))
print(f1_score(y_test, result2, average='macro'))
print(f1_score(y_test, result3, average='macro'))
print(f1_score(y_test, result4, average='macro'))

# In[144]:
