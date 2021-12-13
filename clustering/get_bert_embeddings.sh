#!/bin/sh
# Run `get_bert_embeddings.py` script`

sent_type="sentence"
max_sent_length=128
for num_sentence in 50 100 200 300 400;
do
	echo "Execute get_bert_embedding.py using ${sent_type}-based repreesntations."
	echo "num_sentence=${num_sentence}, max_sent_length=${max_sent_length}"
	python get_bert_embeddings.py \
		--rep_unit $sent_type \
		--num_sentence $num_sentence \
		--max_sent_length 128 \
		--output_dir ../outputs/clustering/base_sent_ns-$num_sentence.ml-$max_sent_length \
		--gpu_id 1
done
