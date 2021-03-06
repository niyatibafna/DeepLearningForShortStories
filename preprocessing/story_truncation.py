"""Truncate story around maximum length by selecting sentences using tf-idf.


Note: Since, BERT can handle raw sentences, we export the truncated story from 
      original text in lowercase.
      If one wants to score sentence using the tf-idf dict computed on lemmatized text.
      Please use `lemmatized_setneces` for scoring the sentence's tf-idf.
      Otherwise, no need to lemmatize the sentences.
"""
import sys
sys.path.append("../")

import os
import argparse
import numpy as np

from preprocess import lemmatize
from utils.stories import Stories
from utils.utils import (
        load_object_from_pkl, 
        write_to_file,
        get_output_dir)

from tqdm import tqdm
from functools import partial
from nltk.tokenize import sent_tokenize
from collections import defaultdict

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max_sent_len", type=int, default=128,
                        help="Max length of story.")
    parser.add_argument("--tfidf_file", type=str, default="tfidf_dict.pkl")
    parser.add_argument("--score_strategy", type=str, default="mean")
    parser.add_argument("--data_dir", type=str, default="../data/bert_tokenizer")
    parser.add_argument("--output_dir", type=str, default="../data/truncation_128_tfidf_mean")
    return parser.parse_args()


def compute_scores(sentences, score_fn):
    """Compute the score for list of sentences.
    Args:
      sentences: List of sentences.
      score_fn: callable function to score sentence. 
    Returns:
      List of the averaging tf-idf.
    """
    scores = [score_fn(s) for s in sentences]
    return scores

def compute_sentence(sentence, tok2score, return_mean=True):
    """Compute score for one sentence.
    
    Note: If you use the raw text and tf-idf dict based on 
          lemmatized text with stopwords, punctuations 
          removal. These tokens are `missing` in tf-idf
          dict. The score will not take these missing tokens
          into account.

    Args:
      sentence: Sequence of tokens that can be separated by space.
      tok2score: Dict. Mapping token into value (int or float).
    Returns:
      out: Float. The tf-idf score of the sentence. 
    """
    tokens = sentence.split()
    # Stopwords and punctuation may not in dict
    out = [tok2score[tok] for tok in tokens if tok in tok2score]

    if len(out) == 0:
        return 0

    if return_mean:
        return sum(out)/len(out)
    # print(sum([tok2score[tok] for tok in tokens]))
    return sum(out)


def select_sentence(sentences, indices, max_len=128):
    """Select sentence according the indices.
    
    Args:
      sentences: List of sentences.
      indices: List of indices in descending order.
    Return:
      truncated_sentence: One story concatenated by whitespace.
    """
    current_len = 0
    sent_indices = list()
    for sent_idx in indices:
        sent = sentences[sent_idx]
        current_len += len(sent.split())
        sent_indices.append(sent_idx)
        if current_len >= max_len:
            break
    # Select sequences
    sent_lst =  [ sent for idx, sent in enumerate(sentences) if idx in set(sent_indices)]
    truncated_sentence = " ".join(sent_lst)
    return truncated_sentence


def main():
    args = get_args()
    if args.score_strategy.lower() not in {"mean", "sum"}:
        raise ValueError("score strategy must be `mean` or `sum`")
    use_mean = True if args.score_strategy == "mean" else False
    output_dir = args.output_dir
    max_len = args.max_sent_len
    tf_idf_file  = args.tfidf_file

    ### If using the tf-idf dict .###
    stories = Stories(REL_STORY_PATH=args.data_dir)
    num_stories = len(stories)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Create folder for output directory: {output_dir}")
   
    print(f"loading tf-idf dictionary from: {tf_idf_file}")
    tf_idf_dict = load_object_from_pkl(tf_idf_file)
    # print(len(tf_idf_dict))
    
    for fname, story in tqdm(stories.read_all_stories(return_file_name=True), total=num_stories):
        text_id = fname.split("_")[0]

        # List of sentences
        sentences = sent_tokenize(story)[:10]
        
        ### Lemmatize the sentences: Do this if loading stories without lemmatization  ###
        ### and using lemmatized tf-idf. ###
        # Convert to a list of sentence in string
        sentences_nested_lst = [sent.split() for sent in sentences ] 
        lemmatized_sentences = [" ".join(sent) for sent in lemmatize(sentences_nested_lst)]
        ### Lemmatize the sentecnes ###

        tok2score = tf_idf_dict[text_id]
        tf_idf_fn = partial(compute_sentence, tok2score=tok2score, return_mean=use_mean)
        scores = compute_scores(sentences=lemmatized_sentences,
                                score_fn=tf_idf_fn)
        
        # index of scores in descending order
        largest_indices = np.argsort(np.array(scores))[::-1]
        
        # print(largest_indices)
        # for idx, s in enumerate(sentences):
        #    print(f"#sent {idx}#: {s}")
        
        # Get the short version by selecting high tf-idf-scored sentences
        short_story = select_sentence(sentences=sentences,
                                      indices=largest_indices,
                                      max_len=max_len) 

        # Export short story
        short_story_fname = get_output_dir(output_dir, fname)
        write_to_file(short_story_fname, short_story)


if __name__ == "__main__":
    main()
