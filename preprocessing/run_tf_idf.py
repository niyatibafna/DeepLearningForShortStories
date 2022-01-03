"""Compute word importance by tf-idf and relative frequency.

The script computes the tf-idf for each story and export as nested dicitonary.
In addition, a subowrd vocabulary file with frequency and relative frequency will also be saved. 

Export:
    1 `tf_idf_dict.pkl`: Nested dictionary with the text id as primary key linking to the inner tf-idf dict.
    2 `bert_tokenizer.vocab`: vocabulary file with the frequency and relative frequency.
"""
import sys
sys.path.append("../")
from utils.stories import Stories
from utils.utils import get_output_dir
from collections import Counter
import argparse
from transformers import BertTokenizer
from tqdm import tqdm
from functools import partial
import pickle
import numpy as np


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="../data/bert_tokenizer")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def compute_document_frequency(token, doc_lst):
    """Return number of documents that contains the token.
    
    Args:
      token: str. One token in the document.
    Returns:
      document frequency: Int. Number of documents containing token. 
    """
    return sum([1 if token in doc else 0 for doc in doc_lst] )


def compute_tf_idf_np(tokens, doc_lst, num_stories):
    """Compute tf-idf based on numpy.
    
    Args:
      tokens: List of tokens.
      doc_lst: List of set. Each set is the set of tokens for one story.
      num_stories: Int, Number of story.

    Returns:
      unique_token: Numpy array of tokens.
      tf-idf: Numpy array of tf-idf scores.
    """
    unique_token, counts = np.unique(np.array(tokens), return_counts=True) 
    num_tokens = np.sum(counts)
    # Making partial function
    doc_freq_fn = partial(compute_document_frequency, doc_lst=doc_lst)
    doc_freq_arr = np.array(list(map(doc_freq_fn, unique_token)))
    # Computing tf and idf
    tf = counts / num_tokens
    idf = np.log((num_stories+1)/(doc_freq_arr+1))
    return unique_token, tf*idf


def main():
    args = get_args()

    stories = Stories(REL_STORY_PATH=args.data_dir)
    num_stories = len(stories)

    print(f"Loading files from: {args.data_dir}")
    # Collect story as a set of tokens
    story_lst = [ set(story.split()) for story in stories.read_all_stories() ]
    token_freq_cnt = Counter()
    tf_idf_dict = dict()

    # Add tf-idf score to dictioanry
    for fname, story in tqdm(stories.read_all_stories(return_file_name=True), total=num_stories):            
        text_id = fname.split("_")[0]
        tokens = story.split()
        # term frequency
        token_freq_cnt.update(tokens)
        # Tf-idf
        unique_token, tf_idf_arr = compute_tf_idf_np(tokens, story_lst, num_stories)
        # Adding to dict
        tf_idf_dict[text_id] = dict()
        for tok, score in zip(unique_token, tf_idf_arr):    
            tf_idf_dict[text_id][tok] = score
    
    # Export nested dict
    fname = get_output_dir(args.output_dir, "tfidf_dict.pkl")
    print(f"Saving file: {fname}")
    with open(fname, "wb") as pfile:
        pickle.dump(tf_idf_dict, pfile, pickle.HIGHEST_PROTOCOL)

    # Export voacb
    total_token = sum(token_freq_cnt.values())
    fname = get_output_dir(args.output_dir, "bert_tokenizer.vocab")
    print(f"Saving file: {fname}")
    with open(fname, "w") as f:
        for tok, freq in token_freq_cnt.most_common():
            relative_freq = freq / total_token
            f.write(f"{tok}\t{freq}\t{relative_freq}\n")


if __name__ == "__main__":
    main()
