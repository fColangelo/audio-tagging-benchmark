# Audio Progressive Resize
Repo to collect code and results for experiments on progressive resize applied to audio event classification


# Dataset structure

baseline_dataset houses the abstract audio dataset

Each dataset gets the file list on its own and read a csv file containing the labels(s) for each file.

Normalization stats are calculated inside init

Each dataset has a change_resolution implemented. Depending on the type of dataset, this function either changes the melspectrogram transform, changes the resize transforms or both. 

# TODO

- Instantiate transforms in base dataset
- Length normalizer in load audio

# WIP

- Figuring out how to best configure progressive runs, specifically audio cfg objects. Should be something easily   pluggable in external code. It should adjust learning rate and dataset resolution based on epochs and transitions made in config. Options:
    - Callback? 
    - 

# Configs

- details for the progressive resize are kept in the datamodule config.