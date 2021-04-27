#!/usr/bin/env python
# coding: utf-8

# # CE 314/887 Using NLTK for text classification
# 
# ### 1: Text classification with NLTK
# 
# 

# In[4]:


import nltk
import random
from nltk.corpus import movie_reviews
# load the movie review corpus
print (movie_reviews.categories())


# In[5]:


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# put all documents in a random order
random.shuffle(documents)


# In[3]:


print (len(documents))
print(documents[1])


# In[17]:


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(100))


# In[18]:


print(all_words["stupid"])
print (all_words['excellent'])


# <!-- 2： Converting word to features with NLTK -->

# In[5]:


import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)


# In[7]:


print (len(list(all_words.keys())))
word_features = list(all_words.keys())[:5000]
print (word_features[:100])
for per_word in word_features[:100]:
    print (per_word,all_words[per_word])


# In[10]:


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

def find_features(document):
    words = set(document)
    features_prob = {}
    for w in word_features:
        features_prob[w] = document.count(w) / len(documents)  ## compute frequency
    return features_prob


# In[12]:


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# In[12]:


featuresets = [(find_features(rev), category) for (rev, category) in documents]
print (len(featuresets[1][0].keys()))
print (len(featuresets[1][0].values()))


# ### 3: Naive Bayes Classifier with NLTK

# In[14]:


print(len(featuresets))
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]


# In[15]:


classifier = nltk.NaiveBayesClassifier.train(training_set)
testing_set_content=[i[0] for i in testing_set]
golden_label=[i[1] for i in testing_set]
tested_label=classifier.classify_many(testing_set_content)
print (golden_label)
cm = nltk.ConfusionMatrix(golden_label, tested_label) 
print (cm)


# In[16]:


print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)


# In[18]:


classifier.show_most_informative_features(100)


# In[ ]:




