"""Fine-tuning BERT form bert-base ."""
import sys
sys.path.append("../")

import os
import json
import time
import random
import logging
import pathlib
import argparse
from functools import partial
from utils.utils import build_dirs, get_output_dir
from utils.stories import StoriesDataset

import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import (
        TensorDataset,
        DataLoader,
        RandomSampler)

from transformers import BertTokenizer
from models.bert import BertForClassification



def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")

    # Model
    parser.add_argument("--bert_model", type=str, default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="splits/tmp",
                        help="Directory contrains splits and label file.")
    parser.add_argument("--output_dir", type=str, default="results/transfer",
                        help="Directory for saving checkpoint and log file.")

    # Training 
    parser.add_argument("--data", type=str, default="dataset.json")
    parser.add_argument("--epochs", type=int, default=30) 
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--logging_steps", type=int, default=5)
    # parser.add_argument("--num_domain", type=int, default=50)
    parser.add_argument("--gpu_id", type=int, default=0)

    # Zero shot
    parser.add_argument("--zero_shot", type=bool, default=False)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    # Test mode
    parser.add_argument("--test_mode", type=bool, default=False,
                        help="If true, using small batch and check output.")

    return parser.parse_args()


def load_splits_from_dir(split_dir):
    """Return train, dev and test split."""
    split_dict = dict()

    for file_extension in ["train", "dev", "test"]:
        texts = list() 
        labels = list()

        path = get_output_dir(split_dir, f"split.{file_extension}")
        with open(path) as f:
            for line in f.readlines():
                row = line.split("\t")
                
                labels.append(row[0])
                texts.append(row[1].strip()) ####
        split_dict[file_extension] = (texts, labels)
    return split_dict



def load_label2idx_from_file(file):
    """Load label2idx (dict) from file."""
    label2idx = dict()
    with open(file) as f:
        for line in f.readlines():
            split_line = line.strip().split("\t")
            label, idx = split_line[0], int(split_line[1])
            label2idx[label] = idx
    return label2idx

if __name__ == "__main__":
    args = get_args()
    SEED = 123


    # output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)
    build_dirs(pathlib.Path(output_dir, "ckpt"), logger)
    
    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(msecs)d %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    
    # Add console to logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console) 

    logger.info(args)

    # Load tokenizer and bert model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    label_path = get_output_dir(args.data_dir, "labels.txt")
    label2idx = load_label2idx_from_file(label_path)
    #print(label2idx)

    ### Loading split from dir ###
    # Dictionary of raw text and label
    #   {"train": (["text", "text"], ["label", "label"]),
    #    "dev":   ([...], [...])}
    split_dict = load_splits_from_dir(args.data_dir)
    
    # Train, dev, test 
    train_texts, train_labels = split_dict["train"]
    dev_texts, dev_labels = split_dict["dev"]
    test_split, test_labels = split_dict["test"]

    # Convert label to index
    train_labels = [label2idx[l] for l in train_labels]
    dev_labels = [label2idx[l] for l in dev_labels]
    test_labels = [label2idx[l] for l in test_labels]

    # Encodings by tokenizer
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)    
    dev_encodings = tokenizer(train_texts, truncation=True, padding=True)    
    test_encodings = tokenizer(train_texts, truncation=True, padding=True)    


    # Create Dataset for DataLoader
    train_dataset = StoriesDataset(train_encodings, train_labels)
    dev_dataset = StoriesDataset(dev_encodings, dev_labels)
    test_dataset = StoriesDataset(test_encodings, test_labels)

    ### DataLoader ###
    dataLoader_fn = partial(DataLoader,
                            batch_size=args.batch_size,
                            num_workers=2)

    train_loader = dataLoader_fn(train_dataset, shuffle=True)
    dev_loader = dataLoader_fn(dev_dataset, shuffle=True)
    test_loader = dataLoader_fn(test_dataset, shuffle=False)


    ### 
    args.num_labels = len(label2idx)
    print(f"Number of classes {args.num_labels}")
    ### Model ###
    model = BertForClassification(args)


    # ### Training, evalulating and test ###
    for epoch in range(1, args.epochs+1):    
        epoch_start_time = time.time()
        model.train(train_loader)
        acc_val = model.evaluate(dev_loader)

        logger.info("\n{}\n| end of epoch {:3d} | time: {:5.2f}s | "
                    "valid accuracy {:8.3f} \n{}\n\n".format("-"*59,
                                                             epoch,
                                                             time.time() - epoch_start_time,
                                                             acc_val,
                                                             "-"*59))
        
        # Evaluate on test set
        acc_test = model.evaluate(test_loader)
        logger.info("test accuracy {:8.3f}".format(acc_test))

        # Save model
        pt_file = get_output_dir(args.output_dir, f"ckpt/transfer.epoch-{epoch}.pt")
        torch.save(model, pt_file)
        logger.info(f"Saving checkpoint to {pt_file}")



