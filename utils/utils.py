
import os
import re
import pickle
import pathlib

def load_object_from_pkl(fname):
    """Loading object from pkl file."""
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def preprocess_text(text):
    """Preprocssing text."""
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text

def collect_sentence(file_lst, sentence_length=512):
    """Collecting sentences
    
    Args:
      sentence_length: maximum length of the tokens
    
    Returns:
      sentences: List of sentences.
    """
    sentences = list()
    
    for file in file_lst:
        with open(file, "r", encoding="utf-8") as f:
            tokens = f.read().split()
            token_len = len(tokens)
            if token_len < sentence_length :
                sentence_length = token_len
            sentences.append(' '.join(tokens[:sentence_length]))

    return sentences


def convert_value2str(arr, round_float=True):
    """Covert the elements of list into string type.

    Args:
      arr: List of element(s) that can be converted into string.
    Returns:
      o: List contains string object.
    """
    # 2D list.
    if isinstance(arr[0], list):
        print("here")
        o = list()
        for row in arr:
            if round_float is True:
                o.append([str(round(i, 3)) for i in row])
            else:
                o.append([i for i in row])
        return o
            
    # 1D list 
    if round_float:
        o = [str(round(i, 3)) for i in arr]
    else:
        o = [str(i) for i in arr]
    return o


def write_to_file(write_file, text):
    with open(write_file, "w") as f:
        f.write(text)
    
    
def build_dirs(output_dir, logger):
    """Build hierarchical directories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Create folder for output directory: {output_dir}")


def get_output_dir(output_dir, file):
    """Joint path for output directory."""
    return pathlib.Path(output_dir,file)

