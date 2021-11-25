# Compute the similarty between  and sentence embeddings from BERT mdoel.
import os
import re
import pandas as pd
import argparse

from utils.similarity import compute_consine_similarity_from_matrix
from utils.utils import (
    preprocess_text,
    collect_sentence,
    convert_value2str
)
from utils.stories import Stories

import torch
import torch.nn as nn

from transformers import BertModel, BertConfig, BertTokenizer
import sklearn.metrics as metrics


def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--data_dir", type=str, default="data/clean/")
    parser.add_argument("--dim", type=str, default=128)
    parser.add_argument("--batch_size", type=str, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="outputs/top_k_stories_bert_tiny.txt")
    parser.add_argument("--test_mode", type=bool, default=False)

    return parser.parse_args()


def get_bert_embeddings(sentences, num_data, dim, batch_size, model, tokenizer):
    """Get the sentence representations.

    Looping the sentences (stories) in batch and obtain the
    repsentations by callable model.

    Args:
      sentences: List of sentences in string.
      num_data: Int, number of sentences.
      dim: Int, dimensions of representation.
      batch_size: Int.
      model: Callable model.
      tokenizer: transformers.BertTokenizer, tokenize one or several
                 sequence(s).

    Tokenizer handles a batch of sentences:
    >>> sentences = ["Hello I'm a single sentence", "And another sentence", "And the very very last one"]
    >>> input_encodings = tokenizer(inputs)

    Returns:
      bert_sent_embeddings: Tensor, a matrix stores sentence embeddings
                            with shape (num_data, dim)
    """
    # Initalize zero matrix with shape (num_data, dim)
    bert_sent_embeddings = torch.zeros(num_data, dim)

    # Loop 4071 stories in batch and set embedding to matrix
    for i in range(0, num_data, batch_size):
        end = i+batch_size
        if end > num_data:
            end = num_data
        # Dict object contains `input_ids`,`token_type_ids`, `attention_mask`
        sent_encodings = tokenizer(sentences[i:end], padding=True, truncation=True,return_tensors="pt")
        # `pooldr_output` has shape (batch_size, dim)
        out = model(**sent_encodings)["pooler_output"]
        # Set embeddings to zeros matrix
        bert_sent_embeddings[i:end,:] = out
    return bert_sent_embeddings


def main():
    # Arguments
    args = get_args()

    # Test data
    query_data = ["love, romance, flowers", "cars, chase, guns", "magic, enchant, witch", "animals", "Christmas, December, Christ"]

    # Training arguments
    data_dir = args.data_dir
    bert_dim = args.dim
    batch_size = args.batch_size

    # Truncate sentence into short seuqence for test mode
    if args.test_mode:
        sentence_length = 5
    else:
        sentence_length = 512

    # List of sentences
    file_lst = [os.path.join(args.data_dir, f) for f in os.listdir(data_dir) if not f.startswith(".")]
    sentences = collect_sentence(file_lst=file_lst, sentence_length=sentence_length)
    num_data = len(sentences)

    # Load tokenizer and bert model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained(args.model)

    # Shape (num_data, bert_dim)
    bert_sent_embeddings = get_bert_embeddings(sentences=sentences,
                                               num_data=num_data,
                                               dim=bert_dim,
                                               batch_size=batch_size,
                                               model=model,
                                               tokenizer=tokenizer)
    # print(bert_sent_embeddings)
    # Shape (num_query, bert_dim)
    query_embeddings = get_bert_embeddings(sentences=query_data,
                                           num_data=len(query_data),
                                           dim=bert_dim,
                                           batch_size=1,
                                           model=model,
                                           tokenizer=tokenizer)

    # Shape (num_query, num_data)
    sim_scores = compute_consine_similarity_from_matrix(query_embeddings, bert_sent_embeddings)
    # print(sim_scores)

    # Return tuple (values, indices)
    topk_values_indices = torch.topk(sim_scores, args.k)
    # `_pt` pytorch tensor. Both has shape (num_query, num_data)
    values_pt, indices_pt = topk_values_indices[0], topk_values_indices[1]

    # Convert into string type
    topk_cos_sims = convert_value2str(values_pt.tolist())
    # topk_indices = convert_value2str(indices_pt.tolist())

    topk_indices = indices_pt.tolist()

    # print(topk_cos_sims)
    # print(topk_cos_sims)

    # Write
    # with open(args.output_file, "w") as f:
    #     indices_str = ",".join([str(i) for i in list(range(1,args.k+1))])
    #     f.write(f"query_data,{indices_str},scores\n")
    #     # Iterate row-wise (4071)
    #     for idx, cos_sim_lst in enumerate(topk_cos_sims):
    #         # Select by index of row
    #         topk_indices_str = ",".join(topk_indices[idx])
    #
    #         # Story index in .xlsx
    #         scores_str = "\t".join(cos_sim_lst)
    #         f.write(f"{query_data[idx]},{topk_indices_str},{scores_str},\n")


    with open(args.output_file, "w") as f:
        stories = Stories('data/source/4000-Stories-with-sentiment-analysis.xlsx')
        indices_str = "\t\t".join([str(i) for i in list(range(1,args.k+1))])
        f.write(f"query_data\t\t{indices_str}\t\tscores\n")
        # Iterate row-wise (4071)
        for idx, cos_sim_lst in enumerate(topk_cos_sims):
            # Select by index of row
            topk_indices_str = "\t\t".join([str(stories.get_title_from_id(sidx)) for sidx in topk_indices[idx]])

            # Story index in .xlsx
            scores_str = "\t\t".join(cos_sim_lst)
            f.write(f"{query_data[idx]}\t\t{topk_indices_str}\t\t{scores_str}\n")



if __name__ == "__main__":
    main()
