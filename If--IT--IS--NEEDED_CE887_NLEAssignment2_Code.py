#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Link : https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386 
##       https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
##       Class3 NLTK and Sentiment Analysis 



##Important various libraries
import nltk
## stopwords brings all the stopwords I need in removing stopwords in each files
from nltk.corpus import stopwords
## the movie reviews files that I will work on
from nltk.corpus import movie_reviews
## this package will be used in tokenize is tokenization of each words
from nltk.tokenize import word_tokenize
## this package will be used in getting confusion matrix. Additionally the values in conusion matrix will be important
## in calculating f1 score
from nltk.metrics.confusionmatrix import ConfusionMatrix

##Import regular expression
import re
##Importing random
import random

import collections
from collections import defaultdict


# In[2]:


movie_reviews.fileids()[1000:]


# In[3]:


###Deleting the punctuations

##In here, I will add all the documents in nltk movie reviews to this list called "documents"
documents = []

#for category in movie_reviews.categories():
#    for fileid in movie_reviews.fileids(category):
#            documents.append((list(movie_reviews.words(fileid)),category))

##In movie reviews in nltk, is being added to list called "documents"
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
           documents.append((list(movie_reviews.words(fileid))))


## documents_nopunct is a list which will be used later
documents_nopunct = []        

#print(len(documents[0]))
#documents0= documents[0]
#documents0_nopunct = [word.lower() for word in documents0 if re.search("\w", word)] 
#print(len(documents0_nopunct))
#print(documents0_nopunct)


index = 0
while index < 2000:
    document= documents[index]
    ## By using regular expressions, I am searching for just word with below command and append each word to "documents_nopunct"
    documents_nopunct.append(list([word.lower() for word in document if re.search("\w", word)] ))
    index = index + 1

##I then emptied the list. Since I got each review with out punctuation in documents_nopunct
documents = []
index = 0
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
            #then I append each review in "documents_nopunct" list to documents. However, I do it with tuple. The second
            #element in tuple is the label which is either "pos" for positive review and "neg" for negative review
            documents.append((documents_nopunct[index],category))
            index = index + 1


# In[4]:


#print(documents[1999])


# In[5]:


##After getting rid of the punctuation from each review, then I shuffle the whole list containing the tuple which contains
## review and the label o the review
random.shuffle(documents)

#print(documents[0][1])
#print(nltk.FreqDist(documents[0][0]).most_common(75))
#print(nltk.FreqDist(documents[0][0]).most_common(75))


# In[6]:


##In this function, I will remove stop words.
def  deal_with_stop_words(_list):
    
    ##I store the list of words in a review in a variable called word_list
    word_list = _list
    ##I created a varaible called "list_of_stopwords". I stored list of stop words in english which I got from
    ##from nltk.corpus import stopwords.
    list_of_stopwords = list(stopwords.words('english'))
    
    #print(wordsList_mostCommon)
    
         
    filtered_list=[]
    ##In this loop, I will compare words in "word_list" and "list_of_stopwords". Then if the word in comparision is not
    ## a stop word then add it to "filtered_list"
    for w in _list:
        if w not in list_of_stopwords:
            filtered_list.append(w)
    
    ###return the the filtered_list as a result. 
    return filtered_list

##This is a list which is currently empty. In this list I will fill it with tuple. 
##In tuple, first element will be a list and second element will be the label
_resultList = []

index = 0
while index < len(documents):
    ##I will remove stop words in each document. I wrote this process in a function to make it more managable
    ##In here, I append a tuple. deal_with_stop_words(documents[index][0]) will be the first element which will be the 
    ##latest form of the review. ordering will not change so documents[index][1] will bring the label
    _resultList.append((deal_with_stop_words(documents[index][0]),documents[index][1]))
    index = index + 1
    
##In the end we got a list of tuples. In first tuple we have list of words each from the 
## movie reviews which are filtered from punctuations and stop words


# In[7]:


#print(_resultList[3][1])


# In[8]:


##Then I will tag each words in each review. Before doing that I created different lists

##in this list, I will add reviews which are labeled as pos
pos_reviews = []
##in this list, I will add reviews which are labeled as ned
neg_reviews = []

##in this list, I will add words which are seen in positive reviews
words_in_Pos = []
##in this list, I will add words which are seen in negative reviews
words_in_Neg = []
##in this list, I will add words which are tagged as adjective
adj_words =[]
##This I made a target_tag. In this list JJ is adjective according to NLTK. I made it as a list in order to make it modifiable.
##For example, I can change it to "NN" and search for Nouns in future 
target_tag = ["JJ"]

##In this list, features will be added to this list 
##It will look like :({good:True,bad:False,Astonishing:True},pos)
featuresets = []


# In[9]:


index = 0
##In this loop I put all the reviews which are tagged with "pos" in to list "pos_reviews"
while index < 2000: 
    if _resultList[index][1]  == "pos":
        pos_reviews.append(_resultList[index])
    index = index + 1


# In[10]:


index = 0
##In this loop I put all the reviews which are tagged with "neg" in to list "neg_reviews"
while index < 2000:
    if _resultList[index][1]  == "neg":
        neg_reviews.append(_resultList[index])
    index = index + 1


# In[11]:


#number of positive reviews
number_of_pos = len(pos_reviews)
#number of negative reviews
number_of_neg = len(neg_reviews)


# In[12]:


index1 = 0
##In this loop, each word are added into a list named "words_in_Pos"
##"words_in_Pos" means all the words within positive reviews
while index1<number_of_pos:
    index2 = 0
    while index2< len(pos_reviews[index1][0]):
        words_in_Pos.append(pos_reviews[index1][0][index2])
        index2 = index2 + 1
    index1 = index1 +1


# In[13]:


index1 = 0
##In this loop, each word are added into a list named "words_in_Neg"
##"words_in_Pos" means all the words within negative reviews
while index1<number_of_neg:
    index2 = 0
    while index2< len(neg_reviews[index1][0]):
        words_in_Neg.append(neg_reviews[index1][0][index2])
        index2 = index2 + 1
    index1 = index1 +1


# In[14]:


all_words = []
##In each loop each word will be added to list named "all_words"
##"all_words" will be used like bag of words
for words in words_in_Neg:
    all_words.append(words)
for words in words_in_Pos:
    all_words.append(words)
##Shuffle the bag of words to prevent overfitting
random.shuffle(all_words)
len(all_words)


# In[ ]:





# In[15]:


##Tag each word in the "all_words" with part of speech 
tag_words = nltk.pos_tag(all_words)
tag_index = 0
##Add adjectives to list "adj_words"  
##means add all the words in "all_words" list tagged with "JJ" or adjectives to list "adj_words"
while tag_index < len(tag_words):
    if tag_words[tag_index][1] in target_tag:
        adj_words.append(tag_words[tag_index][0])
    tag_index = tag_index + 1


# In[16]:


##Get the frequency of the adjectives from highest to lowest frequency
adj_words_freq = nltk.FreqDist(adj_words)
##Add them to "adj_features" list. adj_words_freq.keys() are our adjectives
adj_features = list(adj_words_freq.keys())


# In[17]:


#print(adj_features)
#print(adj_words_freq.most_common(100))


# In[18]:


def find_features(document):
    #There is a dictionary idea is to create a vector like below
    #{astonishing:True,bad:False,Exciting:true}
    features = {}
    #each word in a document is being search in adj_features
    #adj_features contains every adjective
    for w in adj_features :
        ##in a document if there is an adjective then make it True else make it False
        ###It will look like :{good:True,bad:False,Astonishing:True}
        features[w] = (w in document)
    return features


for (rev,category) in _resultList :
    ##This will happen to each document
    ##For example, an item in the "featuresets" will look like this: 
    ##It will look like :({good:True,bad:False,Astonishing:True},pos)
    featuresets.append((find_features(rev),category))


# In[19]:


#print(featuresets[0])


# In[20]:


random.shuffle(featuresets)


# In[21]:


#90 percent of the data will be used as training set
training_set = featuresets[:1800]
#10 percent of the data set will be used as testing set
testing_set = featuresets[1800:]
#train the training data
classifier =nltk.NaiveBayesClassifier.train(training_set)
##Show the overall accuracy of the model
print("Overall accuracy percent: ",(nltk.classify.accuracy(classifier,testing_set))*100)


# In[22]:


#Show most informative features
classifier.show_most_informative_features(15)


# In[23]:


##Get every testing_set
testing_set_content=[i[0] for i in testing_set]
##Get ever tag for each testing_set
golden_label=[i[1] for i in testing_set]
##classify each reviews in testing_set_content
tested_label=classifier.classify_many(testing_set_content)
##then compare the prediction done for reviews or items in 
##testing_set_content with tags associated already with testing_set
cm = nltk.ConfusionMatrix(golden_label, tested_label) 
print (cm)


# In[30]:


##For final evaluation I used sklearn.metrics classificiation_report which include prceision, recall and f1-score
##I was able to trust sklearn.metrics's evaluation since they generate same confusion matrix and 
## use same calssifier was remain the same with the same model with same feature set. '


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(golden_label,tested_label))
print(classification_report(golden_label,tested_label))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




