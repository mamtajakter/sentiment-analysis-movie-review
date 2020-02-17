
# coding: utf-8

# In[32]:


########################
#NLP PROJECT WITH DEEP LEARNING#
########################


# In[31]:


# https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy


# In[57]:


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# df = pd.read_excel('sentiment_emails.xlsx', sheet_name='Sheet1')
df = pd.read_excel('IMDB-Dataset.xlsx', sheet_name='Sheet1')
# print("Column headings:")
# print(df.columns)
# print (df[:10])
#
#
# # In[95]:
#
#
# print(df.info())
# print(df.shape)
# print(df['message'].nunique())


# In[2]:


# # Data visualisation
# import matplotlib.pyplot as plt
# import seaborn as sns

# # # Statistics
# # from scipy import stats
# # import statsmodels.api as sm
# # from scipy.stats import randint as sp_randint
# # from time import time

# # Set up graph
# fig, ax = plt.subplots(1, 1, dpi = 100, figsize = (10, 5))

# # Get data
# sentiment_labels = df['label'].value_counts().index
# sentiment_count = df['label'].value_counts()

# # Plot graph
# sns.barplot(x = sentiment_labels, y = sentiment_count)

# # Plot labels
# ax.set_ylabel('Count')
# ax.set_xlabel('Sentiment Label')
# ax.set_xticklabels(sentiment_labels , rotation=30)


# In[58]:


# Once we have our data ready, it is time to do some preprocessing. We will focus on removing useless variance for our task at hand. First, we have to convert the labels from strings to binary values for our classifier:
df['label'] = df.label.map({'positive': 0, 'negative': 1})
# print (df[:10])


# In[59]:


# Second, convert all characters in the message to lower case:
df['message']  = df["message"].map(lambda x: x if type(x)!=str else x.lower())
# print (df[:10])


# In[67]:


# from wordcloud import WordCloud
# # Creating a list of train and test data to analyse
# # df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)
# imdb_list = df["message"][df.label.isin(['0'])].unique().tolist()
# imdb_bow = " ".join(imdb_list)

# # Create a word cloud for psitive words
# imdb_wordcloud = WordCloud().generate(imdb_bow)

# # Show the created image of word cloud
# plt.figure(figsize=(20, 20))
# plt.imshow(imdb_wordcloud)
# plt.show()


# In[5]:


# data=df
# data.shape
# data.dtypes
# data.isnull().sum()

# data = data.dropna(subset=['message'])

# from wordcloud import WordCloud, STOPWORDS
# stopwords = set(STOPWORDS)

# def show_wordcloud(data, bg, title = None):
#     wordcloud = WordCloud(
#         background_color=bg,
#         stopwords=stopwords,
#         max_words=200,
#         max_font_size=40,
#         scale=3,
#         random_state=1 # chosen at random by flipping a coin; it was heads
# ).generate(str(data))

#     fig = plt.figure(1, figsize=(15, 15))
#     plt.axis('off')
#     if title:
#         fig.suptitle(title, fontsize=20)
#         fig.subplots_adjust(top=2.3)

#     plt.imshow(wordcloud)
#     plt.show()
# d1= data[data.label.isin(['0'])]
# d2= data[data.label.isin(['1'])]
# show_wordcloud(d1['message'], 'white')
# show_wordcloud(d2['message'],'black')


# In[6]:


# # Second, convert all characters in the message to lower case:
# df['message']  = df["message"].map(lambda x: x if type(x)!=str else x.lower())
# print (df[:10])


# In[60]:


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
print (df[:10])


# In[61]:


import matplotlib.pyplot as plt
data=df
data.shape
data.dtypes
data.isnull().sum()

data = data.dropna(subset=['message'])

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, bg, title = None):
   wordcloud = WordCloud(
       background_color=bg,
       stopwords=stopwords,
       max_words=200,
       max_font_size=40,
       scale=3,
       random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

   fig = plt.figure(1, figsize=(15, 15))
   plt.axis('off')
   if title:
       fig.suptitle(title, fontsize=20)
       fig.subplots_adjust(top=2.3)

   plt.imshow(wordcloud)
   plt.show()
d1= data[data.label.isin(['0'])]
d2= data[data.label.isin(['1'])]
show_wordcloud(d1['message'], 'white')
show_wordcloud(d2['message'],'black')


# In[125]:


dataframe=df


# In[12]:


import nltk
nltk.download()


# In[62]:


# print (df[:10])
# df1=df[:10]


df['message'] = df['message'].apply(str)
# Now we can apply the tokenization:
# print (df1[:10])


count = df['message'].str.count(' ')
print(count.mean())
# count.index = count.index.astype(str) + ' words:'
# count.sort_index(inplace=True)
# print (count[:10])
df['count']=count
# print (df1[:10])
df['message'] = df['message'].apply(nltk.word_tokenize)
# print (df1[:10])


# In[63]:


# print(count.mean())


# In[65]:


# df1=df.groupby("count").count()
# dff=df1[:10]
# print(df1[:10])




# In[23]:


import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# mu, sigma = 100, 15
x = df['count']

# the histogram of the data
n, bins, patches = plt.hist(x, bins=np.arange(df['count'].min(), df['count'].max()+1))


plt.ylabel('Number of Reviews')
plt.xlabel('Number of Words in Review')
plt.title('Histogram of Word Count')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(0, 500)
# plt.ylim(0, 25000)
# plt.grid(True)
plt.show()


# In[24]:


# # plotting the points
# # plt.plot(x, y)
# import numpy as np
# # naming the x axis
# plt.ylabel('Number of Words in Review')
# # naming the y axis
# plt.xlabel('Number of Reviews')
# # df.count.plot(kind='bar')
# # # giving a title to my graph
# # plt.title('My first graph!')

# # # function to show the plot
# # plt.show()
# # print (df[:10])

# # df.plot(x='A', use_index=True)
# plt.hist(df['count'], bins=np.arange(df['count'].min(), df['count'].max()+1), orientation="horizontal")


# In[16]:


import matplotlib.pyplot as plt1
# naming the x axis
plt1.xlabel('Review Index')
# naming the y axis
plt1.ylabel('Word Count of Each Review')

plt1.plot(df.index.values, df['count'])


# In[93]:
