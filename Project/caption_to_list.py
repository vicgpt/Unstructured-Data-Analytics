#!/usr/bin/env python
# coding: utf-8
# THIS IS STALE, NEEDS UPDATE FROM NOTEBOOK'

import pandas as pd
import numpy as np
import string
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
from nltk import word_tokenize, Counter
from nltk.corpus import stopwords
import itertools
import spacy
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import TfidfVectorizer


# ### Objective - Use the raw captions from scrapped data and convert it into a list of words and tf/idf




# function to lemmatize all words in captions
def lemmatization(text):
    text = nlp(text)
    text_lemma = [word.lemma_ for word in text]
    return " ".join(text_lemma)


# function to clean caption obtained from scrapper
def wrangle(text):
    text = text.replace('“', '"').replace('”','"').replace('’', "'")
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
   
    return text


# function to remove stop words and punctuations from the list of caption words
def remove_stopwords(lst):
    return [word for word in lst if 
            ( (word not in stopwords.words()) &
            (word not in list(string.punctuation)) &
            (word not in list(string.digits)))]


# applying the cleaning functions
def caption_cleaning(data):
    # subsetting captions only
    captions = data[['caption']]
    captions['caption'] = captions['caption'].astype(str).str.strip()
    captions['caption'] = captions['caption'].map(lambda s: wrangle(s))
    captions['caption_lemma'] = captions['caption'].map(lemmatization)

    # creating caption list
    captions['caption_list'] = captions['caption_lemma'].map(
        lambda row: word_tokenize(row.lower()))
    
    # removing stop words and punctuation
    captions['caption_list'] = captions['caption_list'].map(lambda row: remove_stopwords(row))
    
    return captions


# get tf idf dataframe
def tf_idf(col):
    # creating tf-idf vector
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(col.values)
    columns = vectorizer.get_feature_names()

    # creating tf idf df
    tf_idf_df = pd.DataFrame(X.toarray(), columns=columns)
    
    return tf_idf_df


# creating tf-idf vectors
def caption_tf_idf(captions):
    # creating a list of caption words
    caption_words_list = []
    for l in list(captions['caption_list'].values):
        caption_words_list = caption_words_list + l

    # removing duplicates 
    caption_words_list = list(set(caption_words_list))

    # removing words with length <= 2
    caption_words_list = [w for w in caption_words_list if len(w) > 2]

    # converting the text to list
    captions['caption_cleaned'] = captions['caption_list'].map(lambda lst: ' '.join(lst))
    
    # get tf idf vec
    caption_tf_idf = tf_idf(captions['caption_cleaned'])
    caption_words_list = list(set(caption_words_list).intersection(set(caption_tf_idf.columns)))
    caption_tf_idf = caption_tf_idf[caption_words_list]
    
    return caption_tf_idf



# # reading the data
# nike_data = pd.read_csv('nike_data.csv')
# print(nike_data.shape)
# nike_captions = caption_cleaning(nike_data)
# nike_tf_idf = caption_tf_idf(nike_captions)