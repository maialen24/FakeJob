#!/usr/bin/env python
# coding: utf-8

# # MLP

# In[41]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn import metrics


# ## Doc2vec - MLP

# In[ ]:





# In[42]:


#Datuak path
datuak='./CSV/preprocess.csv'
gorde='./CSV/'


# In[43]:


#datuak kargatu
df=pd.read_csv(gorde+'preprocess.csv')
df.head()


# In[44]:


train, test = train_test_split(df, test_size=0.3, random_state=42)
print(train.shape[0])
print(test.shape[0])


# Doc2Vec

# In[45]:


import nltk
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['post']), tags=[r.Fraudulent]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['post']), tags=[r.Fraudulent]), axis=1)


# In[46]:


model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


# In[47]:


get_ipython().run_cell_magic('time', '', 'for epoch in range(30):\n    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)\n    model_dbow.alpha -= 0.002\n    model_dbow.min_alpha = model_dbow.alpha')


# In[48]:


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors 


# In[49]:


y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)


# In[50]:


#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)


# In[51]:


#Fitting the training data to the network
classifier.fit(X_train, y_train)


# In[52]:


#Predicting y for X_val
y_pred = classifier.predict(X_test)


# In[53]:


#Importing Confusion Matrix
from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations in y_val
cnf_matrix = confusion_matrix(y_pred, y_test)
cnf_matrix


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[55]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[56]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[57]:


y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:





# ## TFIDF - MLP

# In[58]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[59]:


#Datuak path
datuak='./CSV/preprocess.csv'
gorde='./CSV/'
#datuak kargatu
df=pd.read_csv(gorde+'preprocess.csv')
df.head()


# In[62]:


y=df.Fraudulent.values


# In[63]:


xtrain, xvalid, ytrain, yvalid = train_test_split(df.post.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.3, shuffle=True)


# In[65]:


# Always start with these features. They work (almost) everytime!
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)


# In[66]:


X_train=xtrain_tfv
y_train=ytrain
X_test=xvalid_tfv
y_test=yvalid
print(X_train.shape[0])
print(X_test.shape[0])


# In[67]:


#Importing MLPClassifier
from sklearn.neural_network import MLPClassifier

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)


# In[68]:


#Fitting the training data to the network
classifier.fit(X_train, y_train)


# In[69]:


#Predicting y for X_val
y_pred = classifier.predict(X_test)


# In[70]:


#Predicting y for X_val
y_pred = classifier.predict(X_test)


# In[71]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[72]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[73]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[74]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[76]:


y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[77]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




