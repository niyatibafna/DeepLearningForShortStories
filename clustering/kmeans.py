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

def kmeans(story_reps, n_clusters = 10):
    '''Computes KMeans clustering and returns estimator'''
    kmeans_obj = KMeans(n_clusters = n_clusters)
    labels = kmeans_obj.fit_predict(story_reps)
    # print(labels)
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

def main(rep_file_path=None,
         num_clusters=10,
         output_dir=None):

    if rep_file_path:
        story_reps = Representation.load_representations(rep_file_path)
    else:
        raise ValueError("`rep_file_path` must be provided.")

    # Create rep
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # print(story_reps.shape)
    # fake_reps = np.random.rand(15,5)
    # kmeans_obj = kmeans(fake_reps, num_clusters)
    kmeans_obj = kmeans(story_reps, num_clusters)
    if output_dir:
        if "kmeans.pkl" not in output_dir:
            save_file = os.path.join(output_dir, "kmeans.pkl")
        else:
            save_file = output_dir
        save_model(save_file, kmeans_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans clustering on story representations")
    parser.add_argument("--rep_file_path", type=str, default=None, help = "File path for saving extracted representations")
    parser.add_argument("--num_clusters", type=int, default=10, help = "Number of clusters for KMeans")
    parser.add_argument("--output_dir", type=str, default=None, help = "File path for saving trained KMeans model")
    args = parser.parse_args()

    main(rep_file_path=args.rep_file_path,
         num_clusters=args.num_clusters,
         output_dir=args.output_dir)
