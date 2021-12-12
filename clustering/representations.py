#!usr/bin/env python3
"""Constuct representaiton object for accessing sentence- or word-level representations.

We consider two types of BERT embeddings:
    1. sentence-level: Representation of [CLS] token with tahn(dense(h)). See reference 1.
    2. word-level    : Representations form all tokens except for [CLS] and [SEP]

The model runs one mini-batch with shape (num_sentences, max_seq_length) for one story.

Reference:
    1. BERT's outputs: https://github.com/huggingface/transformers/issues/7540#issuecomment-704155218
"""
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


class Representation:

    # to_device("GPU")
    def __init__(self, bert_model):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bertmodel = BertModel.from_pretrained(bert_model)

    def preprocess(self, story_text):
        sentences = sent_tokenize(story_text)
        return sentences[:50]

    def get_last_bert_layer(self, sentences):
        '''Returns last BERT layer'''
        inputs = self.tokenizer(sentences, padding = True, truncation = True, return_tensors = "pt")
        outputs = self.bertmodel(**inputs)
        # print(outputs)
        return outputs

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
        # print("word")
        sentences = self.preprocess(story_text)
        # `last_hidden_states`: (num_sentences, max_length, dims)
        last_layer = self.get_last_bert_layer(sentences)["last_hidden_state"]
        # (num_sentences, max_length-2, dims)
        word_embeddings = last_layer[:, 1:-1, :]
        # Take the mean over axis 0 and 1.
        story_representation = word_embeddings.mean(word_embeddings, axis=[0,1])
        return story_representation

    def get_sentence_based_representation(self, story_text):
        '''Operations are done on sentence embeddings'''
        # print("sentence")
        sentences = self.preprocess(story_text)
        print("Length of story in sentences: ", len(sentences))
        last_layer = self.get_last_bert_layer(sentences)
        # See reference 1
        sentence_embeddings = last_layer["pooler_output"]
        # Take the mean
        story_representation = torch.mean(sentence_embeddings, axis = 0)
        return story_representation
        
    def save_representations(self, rep_file, story_reps):
        '''Save np array to file'''
        with open(rep_file, "wb") as rf:
            np.save(rf, story_reps)

    def load_representations(self, rep_file):
        '''Load story representation file'''
        with open(rep_file, "rb") as rf:
            story_reps = np.load(rf)

        return story_reps

if __name__ == "__main__":
    rep = Representation("google/bert_uncased_L-2_H-128_A-2")
    story = "There was a cat. The cat liked food. The cat drank milk."
    a = rep.get_sentence_based_representation(s)
    # b = rep.get_word_based_representation(s)
    print(a.size())
    # print(b.size())
