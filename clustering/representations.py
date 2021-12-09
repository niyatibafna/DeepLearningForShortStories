#!usr/bin/env python3

import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

class Representation:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained("bert-base-uncased")

    def preprocess(self, story_text):
        sentences = sent_tokenize(story_text)
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        return sentences

    def get_last_bert_layer(self, sentences):
        '''Returns last BERT layer'''
        inputs = self.tokenizer(sentences, padding = True, truncation = True, return_tensors = "pt")
        outputs = self.bertmodel(**inputs)
        return outputs[0]

    def get_word_embeddings(self, last_layer):
        '''Returns BERT contextual word embedding'''
        return last_layer[:, 1:-1, :]

    def get_sentence_embedding(self, last_layer):
        '''Returns BERT sentence embedding'''
        return last_layer[:, 0, :]


    def global_mean():
        '''Returns mean of all vectors depending on how story is divided.'''
        raise NotImplementedError

    def k_part_mean(self, k=3):
        '''Returns k-dimensional mean assuming k parts to the story'''
        raise NotImplementedError

    def get_word_based_representation(self, story_text):
        '''Operations are done on word embeddings'''
        print("word")
        sentences = self.preprocess(story_text)
        last_layer = self.get_last_bert_layer(sentences)
        word_embeddings = last_layer[:, 1:-1, :]
        number_sentences = word_embeddings.size()[0]
        number_words = word_embeddings.size()[1]
        #Take the mean
        word_embeddings = torch.reshape(word_embeddings, (number_sentences*number_words, -1))
        story_representation = torch.mean(word_embeddings, axis = 0)
        return story_representation


    def get_sentence_based_representation(self, story_text):
        '''Operations are done on sentence embeddings'''
        print("sentence")
        sentences = self.preprocess(story_text)
        last_layer = self.get_last_bert_layer(sentences)
        sentence_embeddings = last_layer[:, 0, :]
        #Take the mean
        story_representation = torch.mean(sentence_embeddings, axis = 0)
        return story_representation


if __name__ == "__main__":
    rep = Representation()
    story = "There was a cat. The cat liked food. The cat drank milk."
    # a = rep.get_sentence_based_representation(story)
    b = rep.get_word_based_representation(story)
    # print(a.size())
    print(b.size())
