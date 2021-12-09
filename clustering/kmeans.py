#! usr/bin/env python3
import os
import sys
sys.path.append("../")
from representations import Representation
from utils.stories import Stories
from sklearn.cluster import KMeans
import numpy as np
import pickle


def get_story_representations(rep_unit: str, story_limit = None):
    '''Returns all representations as a numpy array
    of dimensions [num_stories, dims_representation]'''

    stories = Stories(REL_STORY_PATH = "../data/raw/")
    rep = Representation()

    if rep_unit == "sentence":
        rep_method = rep.get_sentence_based_representation
        # story_reps.append(rep.get_sentence_based_representation(story))
    elif rep_unit == "word":
        rep_method = rep.get_word_based_representation
        # story_reps.append(rep.get_word_based_representation(story))
    else:
        raise ValueError("Invalid rep_unit: must be either 'sentence' or 'word'.")

    story_reps = list()
    idx = 0
    for story in stories.read_all_stories():
        story_reps.append(rep_method(story))

        idx += 1
        if idx == story_limit:
            break

    return np.asarray(story_reps)

def kmeans(story_reps):
    '''Computes KMeans clustering and returns estimator'''
    kmeans_obj = KMeans(n_clusters = 2)
    labels = kmeans_obj.fit_predict(story_reps)
    print(labels)
    return kmeans_obj

def save_model(model_file, kmeans_obj):
    '''Save model'''
    with open(model_file, "wb") as f:
        pickle.dump(kmeans_obj, f)

def load_model(model_file):
    '''Load model'''
    with open(model_file, "rb") as f:
        kmeans_obj = pickle.load(f)
    return kmeans_obj




# a = get_story_representations('sentence', story_limit= 3)
# story_reps = np.random.randint(10, size = (5,5))
# kmeans_obj = kmeans(story_reps)
# print(kmeans_obj.predict([[1,2,3,4,5]]))
