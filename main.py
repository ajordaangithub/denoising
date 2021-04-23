import numpy as np
import re
from scipy.io import wavfile
from scipy import signal
import subprocess
import matplotlib.pyplot as plt
from IPython.display import Audio
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import functions

from os import listdir  # to read files
from os.path import isfile, join  # to read files


def read_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]


def main():
    # load files
    feats = read_files("data/perturbed/")
    labels = read_files("data/clean/")

    sr1, x1 = wavfile.read("data/clean/fn000033_5.wav")
    sr2, x2 = wavfile.read("data/perturbed/fn000033_5_aug.wav")

    # preprocessing
    lenx = np.minimum(len(x1), len(x2))
    y = np.array(x1[:lenx], dtype='int16').reshape(-1, 1)
    X = np.array(x2[:lenx], dtype='int16').reshape(-1, 1)

    # train and test DT model
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X, y)
    prediction = tree_model.predict(X)

    # write wav file from prediction
    functions.writeWav(prediction, "output.wav")


main()
