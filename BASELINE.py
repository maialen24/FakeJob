#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION

# In[39]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[40]:


#Datuak path
datuak='./CSV/preprocess.csv'
gorde='./CSV/'


# In[41]:


#datuak kargatu
df=pd.read_csv(gorde+'preprocess.csv')
df.head()


# In[42]:


train, test = train_test_split(df, test_size=0.3, random_state=42)
print(train.shape[0])
print(test.shape[0])


# In[43]:


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



# In[44]:


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


# In[45]:


#vectorsize=300
vectorsize=150
mincount=1
#mincount=2


# In[46]:


model_dbow = Doc2Vec(dm=0, vector_size=vectorsize, negative=5, hs=0, min_count=mincount, sample = 0)
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

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[51]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[52]:


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


# In[53]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[54]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# # LOGISTIC REGRESION - TFIDF
# 

# In[55]:


y=df.Fraudulent.values


# In[56]:


xtrain, xvalid, ytrain, yvalid = train_test_split(df.post.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.3, shuffle=True)


# In[57]:


mindf=1
#mindf=3


# In[58]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=mindf,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)


# In[59]:


X_train=xtrain_tfv
y_train=ytrain
X_test=xvalid_tfv
y_test=yvalid
print(X_train.shape[0])
print(X_test.shape[0])


# In[60]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


# In[61]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[62]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


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


# In[64]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[65]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[67]:





# In[ ]:





# In[ ]:





# In[ ]:




