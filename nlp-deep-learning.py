
# coding: utf-8

# In[32]:


########################
#NLP PROJECT WITH DEEP LEARNING#
########################


# In[31]:


# This code has followed https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy


# In[37]:


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# df = pd.read_excel('sentiment_emails.xlsx', sheet_name='Sheet1')
df = pd.read_excel('IMDB-Dataset.xlsx', sheet_name='Sheet1')
# print("Column headings:")
# print(df.columns)
# print (df[:10])


# In[38]:


# Once we have our data ready, it is time to do some preprocessing. We will focus on removing useless variance for our task at hand. First, we have to convert the labels from strings to binary values for our classifier:
df['label'] = df.label.map({'positive': 0, 'negative': 1})
# print (df[:10])


# In[39]:


# Second, convert all characters in the message to lower case:
df['message']  = df["message"].map(lambda x: x if type(x)!=str else x.lower())
# print (df[:10])


# In[40]:


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


# In[16]:


# Fourth, tokenize the messages into into single words using nltk. First, we have to import and download the tokenizer from the console:

# import nltk
# nltk.download()


# In[41]:


df['message'] = df['message'].apply(str)
# Now we can apply the tokenization:
# print (df[:10])

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

df['Processed_Reviews'] = df.message.apply(lambda x: clean_text(x))


# In[42]:


df.head()


# In[43]:


df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()


# In[44]:


df.head()


# In[48]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.1, random_state=1)

# train, val = train_test_split(train_val, test_size=0.09, random_state=1)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(df['Processed_Reviews'], df['label'], test_size=0.2, random_state=1)


# In[55]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.constraints import max_norm

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(train['Processed_Reviews'])

maxlen = 300
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = train['label']

# x_val = sequence.pad_sequences(val, maxlen=maxlen)
# x_test = sequence.pad_sequences(test, maxlen=maxlen)

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 100
epochs = 3
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.09)


# model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))

# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# print('Train...')
# model.fit(X_t, y,
#           batch_size=batch_size,
#           epochs=15,
#           validation_split=0.09)



# In[54]:


print (X_t[:10])


# In[52]:


# test=pd.read_csv("testData.tsv",header=0, delimiter="\t", quoting=3)
# test.head()
test["message"]=test.message.apply(lambda x: clean_text(x))
# test["sentiment"] = test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)
print (test[:10])
y_test = test["label"]
list_sentences_test = test["message"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)


# In[24]:


# F1-score: 0.8808
# Confusion matrix:
# array([[4404,  631],
#        [ 561, 4404]])


# In[30]:


# some thoughts:
#     train validation test
#     apply all of the preprocessing method of naive bayes here
