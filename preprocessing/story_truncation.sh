#!/bin/sh
# Run `story_truncation.py` script`

s="mean sum"
for max_sent_len in 128 256 384 512;
do
	for scoreStrategy in $s;
	do
		echo "Execute story_truncation.py using ${max_sent_len} sentence length."
		echo "Scoring sentence by ${scoreStrategy} of tf-idf."
		python story_truncation.py \
			--max_sent_len $max_sent_len \
			--tfidf_file tfidf_dict_all.pkl \
			--score_strategy $scoreStrategy \
			--output_dir ../data/truncat_${max_sent_len}_tfidf_${scoreStrategy}_all
	done
done
