
import string
import sys
sys.path.append("../preprocessing/")
from preprocess import *

def is_genre_word(word):
    '''Judges words'''
    non_genres = {"girl", "male", "story", "stories", "body", "tale", "soft"}
    if len(word)<4:
        return False
    if word in non_genres:
        return False

    return True

def _preprocess(word):
    return preprocess(word.strip(), lemmatized=False, rem_stopwords=False)

def get_literary_movements():
    '''Returns additional "genres"'''
    return {"realis", "modernis", "romanticis", "naturalis", "postmodernis"}

def get_genres():
    genre_words = set()
    genres_list = open("genres.txt")
    for line in genres_list:
        line_words = {_preprocess(w) for w in line.strip().split(" ")}
        genre_words = genre_words.union(line_words)

    genre_words = {w for w in genre_words if is_genre_word(w)}
    return genre_words.union(get_literary_movements())

if __name__=="__main__":
    genre_words = get_genres()
    print(genre_words)
