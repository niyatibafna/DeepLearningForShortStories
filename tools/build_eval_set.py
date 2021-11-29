
import string

def is_genre_word(word):
    '''Judges words'''
    non_genres = {"girl", "male", "story", "soft"}
    if len(word)<4:
        return False
    if word in non_genres:
        return False

    return True

def clean(word):
    word = [ch for ch in word.strip() if ch not in string.punctuation]
    word = "".join(word)
    return word.lower()

def get_literary_movements():
    '''Returns additional "genres"'''
    return {"realis", "modernis", "romanticis", "naturalis", "postmodernis"}

def get_genres():
    genre_words = set()
    genres_list = open("genres1.txt")
    for line in genres_list:
        line_words = {clean(w) for w in line.strip().split(" ")}
        genre_words = genre_words.union(line_words)

    genre_words = {w for w in genre_words if is_genre_word(w)}
    return genre_words.union(get_literary_movements)

genre_words = get_genres()
print(genre_words)
