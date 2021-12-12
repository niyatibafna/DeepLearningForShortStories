#! usr/bin/env python3
import os
import sys
sys.path.append("../")
from representations import Representation
from utils.stories import Stories
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
import pickle
import argparse

def get_story_representations(rep_object, rep_unit: str, story_limit = None):
    '''Returns all representations as a numpy array
    of dimensions [num_stories, dims_representation]'''

    stories = Stories(REL_STORY_PATH = "../data/raw/")

    if rep_unit == "sentence":
        rep_method = rep_object.get_sentence_based_representation
        # story_reps.append(rep.get_sentence_based_representation(story))
    elif rep_unit == "word":
        rep_method = rep_object.get_word_based_representation
        # story_reps.append(rep.get_word_based_representation(story))
    else:
        raise ValueError("Invalid rep_unit: must be either 'sentence' or 'word'.")

    story_reps = list()
    idx = 0
    total = len(stories) if story_limit is None else story_limit
    for story in tqdm(stories.read_all_stories(), total=total):
        story_rep = rep_method(story).detach().cpu().numpy()
        story_reps.append(story_rep)
        
        # print("Got rep!")
        if idx == story_limit:
            break
        idx+=1
    return np.asarray(story_reps)

def kmeans(story_reps, n_clusters = 10):
    '''Computes KMeans clustering and returns estimator'''
    kmeans_obj = KMeans(n_clusters = n_clusters)
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

def main(bert_model,
         existing_rep_file_path=None,
         rep_unit="sentence",
         rep_file_path=None,
         num_clusters=10,
         story_limit=None,
         model_file_path=None,
         gpu_id=None):
    rep_object = Representation(bert_model, gpu_id=gpu_id)
    if existing_rep_file_path:
        story_reps = rep_object.load_representations(existing_rep_file_path)
    else:
        story_reps = get_story_representations(rep_object, rep_unit, story_limit)
        if rep_file_path:
            rep_object.save_representations(rep_file_path, story_reps)

    # print(story_reps.shape)
    # fake_reps = np.random.rand(15,5)
    # kmeans_obj = kmeans(fake_reps, num_clusters)
    kmeans_obj = kmeans(story_reps, num_clusters)
    if model_file_path:
        save_model(model_file_path, kmeans_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans clustering on story representations")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help = "BERT model name")
    parser.add_argument("--existing_rep_file_path", type=str, default=None, help = "File path to existing representations")
    parser.add_argument("--rep_unit", type=str, default="sentence", help = "Must be ``sentence'' or ``word''.")
    parser.add_argument("--rep_file_path", type=str, default="../outputs/clustering/kmeans_model.pkl", help = "File path for saving extracted representations")
    parser.add_argument("--num_clusters", type=int, default=10, help = "Number of clusters for KMeans")
    parser.add_argument("--story_limit", type=int, default=None, help = "Number of stories to be clustered, from the top")
    parser.add_argument("--model_file_path", type=str, default="../outputs/clustering/kmeans.pkl", help = "File path for saving trained KMeans model")
    parser.add_argument("--gpu_id", type=int, default=None, help="Specify which GPU to use.")    
    args = parser.parse_args()
    
    # Create rep
    rep_file_dir = "/".join(args.rep_file_path.split("/")[:-1])
    if not os.path.exists(rep_file_dir):
        os.makedirs(rep_file_dir)
    
    main(bert_model=args.bert_model,
         existing_rep_file_path=args.existing_rep_file_path,
         rep_unit=args.rep_unit,
         rep_file_path=args.rep_file_path,
         num_clusters=args.num_clusters,
         story_limit=args.story_limit,
         model_file_path=args.model_file_path,
         gpu_id=args.gpu_id)
