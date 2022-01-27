#!/bin/bash
# Evaluate all models in outputs/clustering for comparison

clustering="../outputs/clustering/"
for model_name in $(ls ../outputs/clustering)
do
    echo "Evaluating ${model_name}"
    python3 rand_score.py --kmeans_model_file_path $clustering$model_name/models/kmeans.pkl
done
#  $model_name$
