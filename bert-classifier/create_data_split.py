"""Conduct data augmentation and split data.

Data augmentation
    1. One story, one sample: Select first 512 tokens for each story.
    2. One story, k smaples : Select multiple segmentation to create serveal story-genre pair (x,y) 
    3. Other approaches
"""
import sys
sys.path.append("../")
from utils.stories import Stories
from utils.utils import get_output_dir

import os
import json
import random
import argparse
from functools import partial
from sklearn.model_selection import train_test_split


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default="../data/source/4000-Stories-with-sentiment-analysis.xlsx")
    parser.add_argument("--output_dir", type=str, default="splits/tmp")
    parser.add_argument("--sentence_len", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--test_mode", type=bool, default=False)

    return parser.parse_args()


def load_genre_buckets(file):
    with open(file, "r") as f:
        data = json.loads(f.read())
    return data


def get_idx_genre_pair(genre_dict, threshold=3):
    """Retrun list of index-genre pairs.

    Args:
      genre_dict: Dict. Object in `genre_buckets.json`.
      threshold: Int. Filter out the genre which the number of the 
                 correspoidng stories is less than theshold. 

    Returns:
      data_pairs: List of index-genre pairs.

    Example:
    >> data_pairs = get_idx_genre_pair(genre_dict)
    >> print(data_pairs)
    [(2616, horror), (3209, horror), ...]
    """
    data_pairs = list()
    
    for gen in genre_dict:
        # if gen not in ["fiction", "american"]:
        #     continue
        if len(genre_dict[gen]) >= threshold:
            # Adding list of (index, genre) pair
            data_pairs += [(stores.split("_")[0], gen) for stores in genre_dict[gen]]
    
    return data_pairs



def contruct_data_pair(examples, idx2text_fn, sentence_len=512):
    """Construct textual input 

    This function does:
        1. Construct pair by different strategy (TO-DO)
        2. Raw text up to `sentence_len` by index.

    Args:
        examples: .
        idx2text_fn: Callback function. The method in `Stories` class to get stroy by index. 
        sentence_len: Int. maximum length of sentence.
    
    Returns:
        pairs: List of textual segment and label pairs.
    """
    def _process_pair(pair, sentence_len):
        """Do 2 and 3."""
        story_idx, genre = pair

        ### Segment up to sentence length ###
        # Story in String
        text = idx2text_fn(int(pair[0])).lower()
        tokens = text.replace("\n","").split() 
        token_len = len(tokens)
        # Maximum length of tokens
        max_len = token_len if (token_len < sentence_len) else sentence_len
        segment = " ".join(tokens[:max_len])

        return (segment, pair[1])
    
    # Do 2 and 3.
    process_fn = partial(_process_pair, sentence_len=sentence_len) 
    pairs = [process_fn(pair=p) for p in examples]
    return pairs


def write_split(texts, labels, file):
    """Write split to file."""
    with open(file, "w") as f:
        for (text, label) in zip(texts, labels):
            f.write(f"{label}\t{text}\n")


if __name__ == "__main__":
    args = get_args()
    SEED = 123

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    f = "../build_eval/genre_buckets.json"
    genre_buckets = load_genre_buckets(f)

    data = args.data_dir
    stories = Stories(data)

    # List of index-genre pairs
    idx_genre_pairs = get_idx_genre_pair(genre_buckets,
                                         threshold=args.threshold)
    
    # Label2idx. {"label_1": 0, "label_2":1, ..}
    labels = list(set([pair[1] for pair in idx_genre_pairs]))
    label2idx = { label: idx for idx, label in enumerate(labels)}
    print(label2idx)

    # Map index into raw text
    idx2text_fn = partial(stories.get_story_from_id)
    example_pairs = contruct_data_pair(examples=idx_genre_pairs,
                                       idx2text_fn=idx2text_fn,
                                       sentence_len=args.sentence_len)

    texts = [p[0] for p in example_pairs]
    labels = [p[1] for p in example_pairs]

    ### Split ###
    # Train split
    train_texts, small_texts, train_labels, small_labels = train_test_split(texts,
                                                                            labels,
                                                                            test_size=.2,
                                                                            random_state=SEED)
    # Dev and test splits
    dev_texts, test_texts, dev_labels, test_labels = train_test_split(small_texts,
                                                                      small_labels,
                                                                      test_size=.5,
                                                                      random_state=SEED)

    # Write file
    train_path = os.path.join(args.output_dir, "split.train")
    dev_path = os.path.join(args.output_dir, "split.dev")
    test_path = os.path.join(args.output_dir, "split.test")
    write_split(train_texts, train_labels, train_path)
    write_split(dev_texts, dev_labels, dev_path)
    write_split(test_texts, test_labels, test_path)

    # Write label file
    label_path = get_output_dir(args.output_dir, "labels.txt")
    with open(label_path, "w") as f:
        for k,v in label2idx.items():
            f.write(f"{k}\t{v}\n")
    

    


