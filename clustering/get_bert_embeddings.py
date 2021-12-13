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

def get_story_representations(rep_object, rep_unit: str, story_limit = None):
    '''Returns all representations as a numpy array
    of dimensions [num_stories, dims_representation]'''

    stories = Stories(REL_STORY_PATH = "../data/raw/")

    if rep_unit == "sentence":
        rep_method = rep_object.get_sentence_based_representation
        # story_reps.append(rep.get_sentence_based_representation(story))
    elif rep_unit == "word":
        rep_method = rep_object.get_word_based_representation
        # story_reps.append(rep.get_word_based_representation(story))
    else:
        raise ValueError("Invalid rep_unit: must be either 'sentence' or 'word'.")

    story_reps = list()
    idx = 0
    total = len(stories) if story_limit is None else story_limit
    for story in tqdm(stories.read_all_stories(), total=total):
        story_rep = rep_method(story).detach().cpu().numpy()
        story_reps.append(story_rep)
        
        # print("Got rep!")
        if idx == story_limit:
            break
        idx+=1
    return np.asarray(story_reps)


def main(bert_model,
         rep_unit="sentence",
         num_sentences=150,
         max_sent_len=128
         output_dir=None
         story_limit=None,
         gpu_id=None):
    
    # Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    rep_object = Representation(bert_model, gpu_id=gpu_id)
    story_reps = get_story_representations(rep_object, rep_unit, story_limit)
    
    if output_dir:
        if "rep_model.npy" not in output_dir:
            save_file = os.path.join(output_dir, "rep_model.npy")
        else:
            save_file = output_dir
        rep_object.save_representations(save_file, story_reps)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save story representations")   
    # hparam for representation
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help = "BERT model name")
    parser.add_argument("--rep_unit", type=str, default="sentence", help = "Must be ``sentence'' or ``word''.")
    parser.add_argument("--num_sentence", type=int, default=150, help="number of sentences for stroy.")
    parser.add_argument("--max_sent_length", type=int, default=128, help="Maximum length of sentence.")

    parser.add_argument("--output_dir", type=str, default="../outputs/clustering", help = "File path for saving extracted representations")
    parser.add_argument("--story_limit", type=int, default=None, help = "Number of stories to be clustered, from the top")
    parser.add_argument("--gpu_id", type=int, default=None, help="Specify which GPU to use.")    
    args = parser.parse_args()
    
     main(bert_model=args.bert_model,
          rep_unit=args.rep_unit,
          num_sentences=aegs.num_sentences,
          max_sent_len=args.max_sent_length,
          output_dir=args.output_dir,
          story_limit=args.story_limit,
          gpu_id=args.gpu_id)
