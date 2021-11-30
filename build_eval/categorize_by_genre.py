#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
sys.path.append("../preprocessing/")

import wikipediaapi
import requests
import os
from genres import *
from preprocess import *

def _preprocess(text):
    '''Returns cleaned text (not lemmatized since genres are not lemmatized to avoid noise)'''
    return preprocess(text, lemmatized=False)

def get_wiki_text(title):
    '''Returns uncleaned text from Wiki page'''

    PARAMS = {
        "action": "opensearch",
        "namespace": "0",
        "search": title + " short story",
        "limit": "5",
        "format": "json"
    }
    open_results = S.get(url=URL, params=PARAMS)
    wiki_titles = open_results.json()[1]

    if len(wiki_titles)==0:
        return None

    for w_title in wiki_titles:
        if w_title.lower() != title and "short story" not in w_title:
            continue
        story_page = wiki.page(w_title)
        if story_page.exists():
            return _preprocess(story_page.text)
    return None

def get_genre(text):
    '''Returns set of all genre words in text'''
    genres = {g_word for g_word in genre_words if g_word in text.split(" ")}
    return genres


def get_story_list(CLEAN_DATA_PATH="../data/clean/"):
    '''Returns list of story titles'''
    stories = os.listdir(CLEAN_DATA_PATH)
    return stories

def stories_by_genre(stories):
    '''Categorize stories by genre from story titles'''

    wiki_found = 0
    story_count = 0

    genre_buckets = defaultdict(lambda:set())

    for fname in stories:
        title = " ".join(fname.split("_")[1:])[:-4] #Remove extension .txt
        story_id = fname.split("_")[0]
        text = get_wiki_text(title)
        if text is None:
            continue
        genres = get_genre(text)
        if not genres:
            continue
        print(fname, genres)
        for g in genres:
            genre_buckets[g].add(fname)

    return genre_buckets

def save_to_file(genre_buckets, OUTFILE):
    '''Saves genre_buckets in JSON format'''
    genre_buckets = {k:list(v) for k,v in genre_buckets.items()}
    with open(OUTFILE, "w") as outfile:
        json.dump(genre_buckets, outfile, indent=4)


def main(STORY_PATH):
    '''Returns genre buckets of story titles from files in STORY_PATH'''
    global S, wiki, URL, genre_words
    S = requests.Session()
    wiki = wikipediaapi.Wikipedia('en')
    URL = "https://en.wikipedia.org/w/api.php"
    genre_words = get_genres()

    stories = get_story_list(STORY_PATH)
    genre_buckets = stories_by_genre(stories)
    print(genre_buckets)
    save_to_file(genre_buckets, OUTFILE="genre_buckets.json")

if __name__=="__main__":
    main("../data/clean/")
