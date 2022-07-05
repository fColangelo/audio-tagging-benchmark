# Audio Progressive Resize
Repo to benchmark different audio tagging models and techniques, to support reproducibility and reliable results.


# Dataset structure

baseline_dataset houses the abstract audio dataset

Each dataset gets the file list on its own and read a csv file containing the labels(s) for each file.

Normalization stats are calculated inside init

Each dataset has a change_resolution implemented. Depending on the type of dataset, this function either changes the melspectrogram transform, changes the resize transforms or both. 

# TODO

- Instantiate transforms in base dataset
- Length normalizer in load audio

# WIP

- 
