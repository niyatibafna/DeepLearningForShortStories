#!/bin/bash
clustering_dir="../outputs/clustering/"

for model_dir in $(ls $clustering_dir)
do
    echo "Running KMeans for representations: ${model_dir}: "
    mkdir $clustering_dir$model_dir/models/
    python3 kmeans.py --rep_file_path $clustering_dir$model_dir/rep_model.npy --output_dir $clustering_dir$model_dir/models/
done
