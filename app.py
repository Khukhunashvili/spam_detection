
# coding: utf-8

# In[54]:


import nltk
from nltk.corpus import stopwords
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# In[23]:


messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['type', 'message'])


# In[24]:


messages.head()


# 
# 

# In[25]:


def text_process(message):
    # Remove punctuations()
    message = ''.join([c for c in message if c not in string.punctuation])
    # Clean message from 'Stop Words'
    cleaned_message = [word for word in message.split() if word.lower() not in stopwords.words('english')]
    # return cleaned text
    return cleaned_message


# In[55]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['type'], test_size=0.33)


# In[56]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[57]:


pipeline.fit(msg_train, label_train)


# In[65]:


predictions = pipeline.predict(msg_test)


# In[66]:


print(classification_report(label_test, predictions))

