"""Export stories into the separated text."""
import sys
sys.path.append("../")
import os
import re
import argparse
import pandas as pd

from tqdm import tqdm
from utils.utils import load_object_from_pkl, write_to_file
from transformers import BertTokenizer

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, default="../data/source/4000_content_only.xlsx")
    parser.add_argument("--output_dir", type=str, default="../data/tmp")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    return parser.parse_args()
    

def main():
    # Arguments
    args = get_args()

    data_dir = args.output_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load tokenizer
    tokenizer_name = args.tokenizer_name
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name) if tokenizer_name else None

    print("Loading dataframe")
    df = pd.read_excel(args.data)
    
    num_row = df.shape[0]
    for row in tqdm(df.itertuples(index=True), total=num_row):    
        story_no = str(row.Index)
        
        if tokenizer_name:
            story = " ".join(tokenizer.tokenize(row.story))
        else:
            story = row.story

        # Do lowercasing and remove all except 0..9, a-z and A..Z.
        title = re.sub(r"\s+", " ", row.title.lower()).replace(" ", "_")
        title = re.sub('[^A-Za-z0-9_]+', '', title)
        title = re.sub(r"_+", "_", title)

        output_file = os.path.join(args.output_dir, f"{story_no}_{title}.txt")
        write_to_file(output_file, story)

if __name__ == "__main__":
    main()
