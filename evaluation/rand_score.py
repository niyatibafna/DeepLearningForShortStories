#!/usr/bin/env python3
'''Takes a KMeans object over all stories, and computes Rand score,
assuming genre buckets as gold labels'''

import pickle as pkl
import json
import argparse

def get_predicted_cluster_labels(story_ids, kmeans_model_file_path):
    '''Use story idx to return predicted label from kmeans object
    Return: dict[story_id]:cluster_id'''
    with open(kmeans_model_file_path, "rb") as f:
        kmeans_obj = pkl.load(f)
    cluster_ids = {idx:kmeans_obj.labels_[idx] for idx in story_ids}
    return cluster_ids

def get_labelled_story2idx(gold_clustering, REL_STORY_PATH="../data/raw/"):
    '''Get original idx for each labelled story.'''

    stories = Stories(REL_STORY_PATH)
    titles = [title for title in stories.read_all_stories()]
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


def multilabel_rand_score(gold_clustering, cluster_ids):
    '''Calculates multilabel Rand Score
    gold_clustering: dict[genre_id]:{story_ids},
    cluster_ids: dict[story_id]:cluster_id,
    labelled_story_ids: list[story_ids]
    Returns multilabel Rand Score
    '''
    true_positives = 0
    for genre, stories in gold_clustering.items():
        pred_cluster_ids = [cluster_ids[story_id] for story_id in list(stories)]
        cluster_count = defaultdict(lambda: 0)
        for cidx in pred_cluster_ids:
            cluster_count[cidx] += 1
        true_positives += sum([len(combinations(same_cluster_count, 2)) for same_cluster_count in cluster_count.values()])

    positives = 0
    total_cluster_count = defaultdict(lambda: 0)
    for story, cluster_id in cluster_ids:
        total_cluster_count[cluster_id] += 1
    positives += sum([len(combinations(same_cluster_count, 2)) for same_cluster_count in total_cluster_count.values()])

    false_positives = positives - true_positives

    total = sum(list(total_cluster_count.values()))
    negatives = total - positives

    false_negatives = 0
    for genre, stories in gold_clustering.items():
        pred_cluster_ids = [cluster_ids[story_id] for story_id in list(stories)]
        cluster_count = defaultdict(lambda: 0)
        for cidx in pred_cluster_ids:
            cluster_count[cidx] += 1
        cluster_counts = list(cluster_count.values())
        assert sum(cluster_counts) == len(stories)
        n = sum(cluster_counts)
        cluster_counts_accumulative = list()
        for idx in range(len(cluster_counts)):
            n -= cluster_counts[idx]
            cluster_counts_accumulative.append(n)

        for idx in range(len(cluster_counts)):
            false_negatives += cluster_counts[idx]*cluster_counts_accumulative[idx]

    true_negatives = negatives - false_negatives

    precision = float(true_positives/positives)
    recall = float(true_positives/(true_positives+false_negatives))

    return precision, recall


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
    print("PRECISION: %s \t RECALL: %s", precision, recall)


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