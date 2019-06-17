# Overview

> This project was originally made as a university assignment and is being used for experimentation with Tensorflow and Python. It is not intended for production use under any circumstances.

This is a Simple Audio Recognition system made using Python and Tensorflow and specifically built on top of [Tensorflow for Poets 2](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2) Google CodeLab repository.

It uses Tensorflow to train a neural network into recognizing digits 0-9 spoken in English. It has been trained using [a dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) containing 2000 recordings of 4 different speakers. Then the user can record a sentence consisting of 4 to 10 words. The script will split the sentence into .wav files for each word and try to recognize each based on its training. Final output will indicate a list of possible answers and a match percentage, for each word. 

## Setup

> Before you can run this a minimal setup is required.

Fetch the recordings dataset
```
cd sar
```
```
git clone https://github.com/Jakobovski/free-spoken-digit-dataset
```
Generate spectrograms for the recordings
```
cd ..
```
```
python ./sar/spectrograms.py
```
Retrain Tensorflow
```
python -m scripts.retrain --bottleneck_dir=tf_files/bottlenecks --how_many_training_steps=2000 --model_dir=tf_files/models --summaries_dir=tf_files/training_summaries/mobilenet_1.0_224 --output_graph=tf_files/retrained_graph.pb --output_labels=tf_files/retrained_labels.txt --architecture="mobilenet_1.0_224" --image_dir=sar/spectrograms
```
Now you can run the project!
```
python ./sar/record.py
```
