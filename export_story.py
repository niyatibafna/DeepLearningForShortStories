# Export stories into the separated text.
import os
import re
import argparse
import pandas as pd

from utils.utils import write_to_file

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, default="4000-Stories-with-sentiment-analysis.xlsx")
    parser.add_argument("--output_dir", type=str, default="")
    
    return parser.parse_args()
    

def main():
    # Arguments
    args = get_args()

    data_dir = "data/raw/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df = pd.read_excel(args.data)

    for row in df.itertuples(index=True):    
        story_no = str(row.Index)
        story = row.story
        # Do lowercasing and remove all except 0..9, a-z and A..Z.
        title = re.sub(r"\s+", " ", row.title.lower()).replace(" ", "_")
        title = re.sub('[^A-Za-z0-9_]+', '', title)
        title = re.sub(r"_+", "_", title)

        output_file = os.path.join(args.output_dir, f"{data_dir}/{story_no}_{title}.txt")
        write_to_file(output_file, story)

if __name__ == "__main__":
    main()
