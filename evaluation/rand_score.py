#!/usr/bin/env python3
'''Takes a KMeans object over all stories, and computes Rand score,
assuming genre buckets as gold labels'''
import pickle as pkl
import json

def get_predicted_cluster_labels(self, story_ids, kmeans_obj_file_path):
    '''Use story idx to return predicted label from kmeans object
    Return: dict[story_id]:cluster_id'''
    with open(kmeans_obj_file_path, "rb") as f:
        kmeans_obj = pkl.load(f)
    cluster_ids = {idx:kmeans_obj.labels_[idx] for idx in story_ids}
    return cluster_ids

def get_labelled_story2idx(self, gold_clustering, REL_STORY_PATH="../data/raw/"):
    '''Get original idx for each labelled story.'''

    stories = Stories(REL_STORY_PATH)
    titles = [title for title in stories.read_all_stories()]
    labelled = {t for g, stories in gold_clustering.items() for t in stories}
    labelled_story2idx = {t:titles.index(t) for t in labelled}
    #CHECK FOR UNIQUENESS
    assert len(labelled_story2idx)==len(list(labelled_story2idx.values()))
    return labelled_story2idx


def get_genre2idx(self, gold_clustering):
    '''Get all genres and return genre2id mapping'''
    genre2idx = {genre:id for genre, id in zip(list(gold_clustering.keys()), range(len(gold_clustering)))}
    return genre2idx


def get_gold_clustering(self, genre_bucket_file_path):
    '''Read gold label file.
    Return: dict[genre_id]:{story_ids}'''
    with open(genre_bucket_file_path, "r") as f:
        gold_clustering = json.load(f)
    return gold_clustering


def multilabel_rand_score(self, gold_clustering, cluster_ids, labelled_story_ids):
    '''Calculates multilabel Rand Score'''
    true_positives = 0
    for genre, stories in gold_clustering.items():
        pred_cluster_ids = [cluster_ids[story_id] for story_id in list(stories)]
        cluster_count = defaultdict(lambda: 0)
        for cidx in pred_cluster_ids:
            cluster_count[cidx] += 1
        true_positives += sum([len(combinations(same_cluster_count, 2)) for same_cluster_count in cluster_count.values()])

    positives = 0
    total_cluster_count = defaultdict(lambda: 0)
    for sidx in labelled_story_ids:
        total_cluster_count[cluster_ids[sidx]] += 1
    positives = sum([len(combinations(same_cluster_count, 2)) for same_cluster_count in total_cluster_count.values()])

    false_positives = positives - true_positives

    precision = float(true_positives/positives)

    return precision





def main(self, genre_bucket_file_path,
            REL_STORY_PATH, kmeans_obj_file_path):

    gold_clustering_raw = self.get_gold_clustering(genre_bucket_file_path)
    genre2idx = self.get_genre2idx(gold_clustering_raw)
    labelled_story2idx = self.get_labelled_story2idx(gold_clustering_raw, REL_STORY_PATH)
    gold_clustering = {genre2idx[genre]:{labelled_story2idx[story] for story in stories} \
                        for genre, stories in gold_clustering_raw.items()}
    labelled_story_ids = list(labelled_story2idx.values())
    cluster_ids = self.get_predicted_cluster_labels(kmeans_obj_file_path, labelled_story_ids)
    precision = self.multilabel_rand_score(gold_clustering, cluster_ids, labelled_story_ids)
    print(precision)
