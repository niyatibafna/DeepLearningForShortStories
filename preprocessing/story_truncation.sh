#!/bin/sh
# Run `story_truncation.py` script`

for max_sent_len in 128 256 512;
do
	echo "Execute story_truncation.py using ${max_sent_len} sentence length."
	python story_sent_len.py \
		--max_sent_len $max_sent_len \
		--output_dir ../data/truncat_${max_sent_len}_tfidf
done
