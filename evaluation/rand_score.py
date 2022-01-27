#!/usr/bin/env python3
'''Takes a KMeans object over all stories, and computes Rand score,
assuming genre buckets as gold labels'''

import pickle as pkl
import json
import argparse
import sys
sys.path.append("../")
from utils.stories import Stories
from collections import defaultdict
from math import comb
import collections
from itertools import combinations

def get_predicted_cluster_labels(kmeans_model_file_path, story_ids):
    '''Use story idx to return predicted label from kmeans object
    Return: dict[story_id]:cluster_id'''
    with open(kmeans_model_file_path, "rb") as f:
        kmeans_obj = pkl.load(f)
    cluster_ids = {idx:kmeans_obj.labels_[idx] for idx in story_ids}
    return cluster_ids

def get_labelled_story2idx(gold_clustering, REL_STORY_PATH="../data/raw/"):
    '''Get original idx for each labelled story.'''

    stories = Stories(REL_STORY_PATH = REL_STORY_PATH)
    titles = stories.get_all_titles()
    labelled = {t for g, stories in gold_clustering.items() for t in stories}
    labelled_story2idx = {t:titles.index(t) for t in labelled}
    #CHECK FOR UNIQUENESS
    assert len(labelled_story2idx)==len(list(labelled_story2idx.values()))
    return labelled_story2idx


def get_genre2idx(gold_clustering):
    '''Get all genres and return genre2id mapping'''
    genre2idx = {genre:id for genre, id in zip(list(gold_clustering.keys()), range(len(gold_clustering)))}
    return genre2idx


def get_gold_clustering(gold_clusters_file_path):
    '''Read gold label file.
    Return: dict[genre]:{stories}'''
    with open(gold_clusters_file_path, "r") as f:
        gold_clustering = json.load(f)
    return gold_clustering

def gold_judgment(s1, s2, gold_clustering):
    '''Returns correct judgment of s1, s2'''
    for genre, story_set in gold_clustering.items():
        if s1 in story_set and s2 in story_set:
            return True
    return False


def multilabel_rand_score(gold_clustering, cluster_ids):
    '''Calculates multilabel Rand Score
    gold_clustering: dict[genre_id]:{story_ids},
    cluster_ids: dict[story_id]:cluster_id,
    Returns multilabel Rand Score
    '''
    real_positives = 0
    labelled_story_ids = list(cluster_ids.keys())
    all_gold_judgments = {s_id:dict() for s_id in labelled_story_ids}
    for s1, s2 in combinations(labelled_story_ids, 2):
         judgment = gold_judgment(s1, s2, gold_clustering)
         all_gold_judgments[s1][s2] = judgment
         all_gold_judgments[s2][s1] = judgment
         if judgment:
             real_positives += 1

    true_positives = 0
    seen_pairs = defaultdict(lambda: set())
    clusters = defaultdict(lambda: set())
    for story, cluster in cluster_ids.items():
        clusters[cluster].add(story)
    for cluster, stories in clusters.items():
        for s1, s2 in combinations(stories, 2):
            if s1 in seen_pairs[s2] or s2 in seen_pairs[s1]:
                continue
            if all_gold_judgments[s1][s2]:
                true_positives += 1
            seen_pairs[s1].add(s2)


    total_cluster_count = collections.Counter(list(cluster_ids.values()))
    judged_positives = sum([comb(same_cluster_count, 2) for same_cluster_count in total_cluster_count.values()])

    # real_positives = true_positives+false_negatives
    precision = float(true_positives/judged_positives)
    recall = float(true_positives/real_positives)

    return precision, recall
    # total = sum(list(total_cluster_count.values()))
    # negatives = total - positives
    #
    # false_negatives = 0
    # for genre, stories in gold_clustering.items():
    #     pred_cluster_ids = [cluster_ids[story_id] for story_id in list(stories)]
    #     cluster_count = collections.Counter(pred_cluster_ids)
    #     cluster_counts = list(cluster_count.values())
    #     assert sum(cluster_counts) == len(stories)
    #     n = sum(cluster_counts)
    #     cluster_counts_accumulative = list()
    #     for idx in range(len(cluster_counts)):
    #         n -= cluster_counts[idx]
    #         cluster_counts_accumulative.append(n)
    #
    #     for idx in range(len(cluster_counts)):
    #         false_negatives += cluster_counts[idx]*cluster_counts_accumulative[idx]
    #
    # true_negatives = negatives - false_negatives



def main(gold_clusters_file_path,
            REL_STORY_PATH, kmeans_model_file_path):

    gold_clustering_raw = get_gold_clustering(gold_clusters_file_path)
    genre2idx = get_genre2idx(gold_clustering_raw)
    labelled_story2idx = get_labelled_story2idx(gold_clustering_raw, REL_STORY_PATH)
    gold_clustering = {genre2idx[genre]:{labelled_story2idx[story] for story in stories} \
                        for genre, stories in gold_clustering_raw.items()}
    labelled_story_ids = list(labelled_story2idx.values())
    cluster_ids = get_predicted_cluster_labels(kmeans_model_file_path, labelled_story_ids)
    precision, recall = multilabel_rand_score(gold_clustering, cluster_ids)
    print(f"PRECISION: {precision}, \t RECALL: {recall}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Calculate Multilabel Rand Score")
    parser.add_argument("--gold_clusters_file_path", type = str, default = "../build_eval/genre_buckets.json",
                        help = "Path to JSON file containing gold genre buckets")
    parser.add_argument("--REL_STORY_PATH", type = str, default = "../data/raw/",
                        help = "Path to folder containing stories")
    parser.add_argument("--kmeans_model_file_path", type = str, default = None,
                        help = "Path to KMeans model object, must be previously fit")

    args = parser.parse_args()
    main(args.gold_clusters_file_path, args.REL_STORY_PATH, args.kmeans_model_file_path)
