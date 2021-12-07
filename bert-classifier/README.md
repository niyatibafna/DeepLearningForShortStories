# Introduction

This Repo conducts tasks: training classifier, evalaution and inference on raw data 


# Data preparation

To run model on classifcation task, one need to prepare the data splits beforehand in one folder.
We select the labeled data according to `../build_eval/genre_buckets.json`. Each pair contains 
story (`x`) and genre (`y`) stored in `split.train`, `split.dev` and `split.test`. 
the lable and text are separated by tab.

```
python create_data_split.py \
    --data_dir ../data/source/4000-Stories-with-sentiment-analysis.xlsx \
    --outout_dir splits/tmp \
    --threshold 3
```

For further testing various preprocessing and data augmentation, we will modify the script.


# Train

To fine-tune BERT model, run the script:

```
python run_finetuning.py \
    --data_dir splits/tmp
```



