#!/usr/bin/env python3

import logging
import sys
import string
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download("punkt")
# nltk.download('averaged_perceptron_tagger')

def case_normalization(text):
    '''Lower cases'''
    return text.lower()

def tokenize(text):
    '''Tokenization!'''
    return [word_tokenize(sent) for sent in sent_tokenize(text)]

def clean(text):
    '''Remove punctuation'''
    # print(text[:10])
    text = [[tok for tok in sent if tok not in string.punctuation] for sent in text]
    # print(text[:10])
    return text

def _lemmatize(word, tag, wn_lemmatizer):
    '''Single word and tag'''
    return wn_lemmatizer.lemmatize(word, tag[0].lower() if tag[0] in ["N", "V", "R"] else "n")

def lemmatize(text):
    '''Lemmatization'''
    wn_lemmatizer = WordNetLemmatizer()
    tagged_text = [nltk.pos_tag(sent) for sent in text]
    text = [[_lemmatize(word, tag, wn_lemmatizer) for (word, tag) in sent] for sent in tagged_text]
    return text

def remove_stopwords(text):
    '''Stop word removal with NLTK list'''
    stop_words = nltk.corpus.stopwords.words('english')
    text = [[word for word in sent if word not in stop_words] for sent in text]
    return text


def preprocess(text, case_normalized=True, tokenized=True, cleaned=True, lemmatized=True, rem_stopwords=True):
    '''Preprocessing pipeline:
    Case normalization
    Sent + word tokenization
    Cleaning
    POS tagging, Lemmatization
    Stopword removal'''
    # print(text[:300])
    if case_normalized:
        text = case_normalization(text)
    if tokenized:
        text = tokenize(text)
    if cleaned:
        text = clean(text)
    if lemmatized:
        text = lemmatize(text)
    if rem_stopwords:
        text = remove_stopwords(text)

    # text = " ".join([word for sent in text for word in sent])
    # print(text[:300])
    o = list()
    [o.extend(s) for s in text]
    return o


def main():
    with open(sys.argv[1], "r") as story_file:
        print(preprocess(story_file.read()))


if __name__ == "__main__":
    main()
