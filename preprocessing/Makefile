#!/bin/zsh
DATA_DIR = ../data/raw/
OUT_DIR = ../data/clean/
preprocess:
	mkdir -p $(OUT_DIR)
	for f in $(shell ls $(DATA_DIR) | grep txt); do \
		./preprocess.py $(DATA_DIR)$$f > $(OUT_DIR)$$f;\
	done
