#!/usr/bin/env python
# coding: utf-8

# # FAKE - JOB

# ## Aurreprozesamendua

# In[58]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from wordcloud import WordCloud


# In[59]:


#Datuak path
datuak='./CSV/fake_job_postings.csv'
gorde='./CSV/'


# In[60]:


#datuak kargatu
df_job=pd.read_csv(datuak)
df_job.head()


# In[61]:


df_job.describe()


# In[62]:


df_job.info()


# In[63]:


#Dataren dimentsioa
print("Data dims is: ",df_job.shape)


# In[64]:


# num of fake jobs in the Dataset
print("Number of real (label as 0) and fake jobs (label as 1) in the dataset :")


print(df_job["fraudulent"].value_counts())
sb.catplot(x="fraudulent", data = df_job, kind = "count")


# In[65]:


fig, axes = plt.subplots(ncols=2, figsize=(17, 5), dpi=100)
plt.tight_layout()

df_job["fraudulent"].value_counts().plot(kind='pie', ax=axes[0], labels=['Real Post (95%)', 'Fake Post (5%)'])
temp = df_job["fraudulent"].value_counts()
sb.barplot(temp.index,temp, ax=axes[1])

axes[0].set_ylabel(' ')
axes[1].set_ylabel(' ')
axes[1].set_xticklabels(["Real Post (17014) [0's]", "Fake Post (866) [1's]"])

axes[0].set_title('Target Distribution in Dataset', fontsize=13)

plt.show()


# In[66]:


'''Konkatenatu zutabe guztiak atributu bakar bat lortzeko '''

text = df_job[df_job.columns[1:-1]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
target = df_job['fraudulent']

print(len(text))
print(len(target))


# In[67]:


'''Post eta klasea duen dataframe-a sortu'''

text=df = pd.DataFrame(text,columns=['post'])
text['Fraudulent']=target
print(text)

'''Gorde'''
text.to_csv(gorde+'textMiningFakeJob.csv')


# In[68]:


def clean_text(text):
    ''' remove text in square brackets,remove links,remove punctuation.'''
    text=text.replace(",", " ")
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
   # text = re.sub('[^a-zA-Z]', ' ', text) 
    
    return text


# Applying the cleaning function
text['post'] = [clean_text(i) for i in text['post'] ] 

print(text)


# In[69]:


#TOKENIZE
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# appling tokenizer5
text['post'] = text['post'].apply(lambda x: tokenizer.tokenize(x))
text.head(5)


# In[70]:


#STOPWORDS
stop_words = stopwords.words('english')
def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    words = [w for w in text if w not in stop_words]
    words = map(lambda word: word.lower(), words)
    return list(words)


text['post'] = text['post'].apply(lambda x : remove_stopwords(x))

print(text)


# In[71]:


#LEMATIZATION
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

#df = pd.DataFrame(['this was cheesy', 'she likes these books', 'wow this is great'], columns=['text'])
text['post'] = text.post.apply(lemmatize_text)


# In[72]:


print(text)


# In[73]:


def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

text['post'] = text['post'].apply(lambda x : combine_text(x))
text.head(5)


# In[74]:


df=text
print(text)


# In[75]:


#True post WordCloud (fraudulent = 0)
plt.figure(figsize = (20,20)) 
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.Fraudulent == 0].post))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[76]:


#Fake post WordCloud (fraudulent=1)
plt.figure(figsize = (20,20)) 
wc = WordCloud(width = 1600 , height = 800 , max_words = 3000).generate(" ".join(df[df.Fraudulent == 1].post))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[77]:


#gorde aurreprozesamendua
df.to_csv (gorde+'preprocess.csv', index = False, header=True)

