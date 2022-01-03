# Find top-k stories
from itertools import combinations

from utils.utils import convert_value2str
from utils.similarity import compute_cosine_similarity

import argparse 
import numpy as np


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--embedding_file", type=str, default="embs.npy")
    parser.add_argument("--output_file", type=str, default="top_k_stories.txt")
    parser.add_argument("--test_mode", type=bool, default=False)
    return parser.parse_args()


def main():
    # arguments
    args = get_args()

    # (4071, 300). Selecting 10 sotries if `test_mode
    if args.test_mode:
        embs = np.load(args.embedding_file)[:10,:]
    else:
        embs = np.load(args.embedding_file)
    
    n_story = embs.shape[0]
    
    # Initialze a matrix (n_sotry, n_story)
    similarity_mat = np.zeros((n_story, n_story))

    # Iterate all possible combinations of
    # two doc indices. (1,2) is equalvalent to (2,1)
    print(f"Computing cosine similarity from {n_story} stories.")
    for c in combinations(range(n_story), 2):
        i,j = c
        cos_sim = compute_cosine_similarity(embs[i], embs[j])
        # Assign score to the entry
        similarity_mat[i][j] = cos_sim
        similarity_mat[j][i] = cos_sim

    # print(similarity_mat)

    with open(args.output_file, "w") as f:
        indices_str = ",".join([str(i) for i in list(range(1,args.k+1))]) 
        f.write(f"text_no,{indices_str},scores\n")
        for i in range(n_story):
            # (1, n_story)
            cosine_sims = similarity_mat[i,:]

            # Returing indices
            topk_indices = convert_value2str(np.argsort(cosine_sims)[-(args.k):][::-1])
        
            # Returning similarities 
            topk_cos_sims = convert_value2str(np.take(cosine_sims, topk_indices))
        
            # Write
            topk_indices_str = ",".join(topk_indices)
            scores_str = "\t".join(topk_cos_sims)
            f.write(f"{i},{topk_indices_str},{scores_str},\n")




if __name__ == "__main__":
    main()
