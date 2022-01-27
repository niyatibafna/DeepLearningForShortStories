#!/bin/sh
# Run `get_bert_embeddings.py` script on truncated story

data_dir="../data/"
sent_type="sentence"

for data in $(ls $data_dir)
do
	if [[ ${data} == *"truncat"* ]]; then
	#echo $data
	echo "Execute get_bert_embedding.py on ${data}"	
	IFS=_ read var1 seq_len var3 var4 <<< $data
	python get_bert_embeddings.py \
		--rep_unit $sent_type \
		--max_sent_length $seq_len \
		--data_dir ${data_dir}/$data \
		--output_dir ../outputs/clustering/base_sent_$data \
		--gpu_id 1
	fi
done
