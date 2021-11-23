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
    text = [_lemmatize(word, tag, wn_lemmatizer) for sent in tagged_text for (word, tag) in sent]
    return text

def remove_stopwords(text):
    '''Stop word removal with NLTK list'''
    stop_words = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stop_words]
    return text


def preprocess(text):
    '''Preprocessing pipeline:
    Case normalization
    Sent + word tokenization
    Cleaning
    POS tagging, Lemmatization
    Stopword removal'''
    # print(text[:300])
    text = case_normalization(text)
    text = tokenize(text)
    text = clean(text)
    text = lemmatize(text)
    text = remove_stopwords(text)
    text = " ".join(text)

    # print(text[:300])
    return text


def main():
    with open(sys.argv[1], "r") as story_file:
        print(preprocess(story_file.read()))


if __name__ == "__main__":
    main()
