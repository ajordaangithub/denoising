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
    files = read_files("data/clean/")

    feats, labels = list(), list()

    for f in files:
        # get sample rate and vector
        sr1, x1 = wavfile.read("data/clean/" + f)
        sr2, x2 = wavfile.read("data/perturbed/" + f[:-4] + "_aug.wav")

        # preprocessing
        lenx = np.minimum(len(x1), len(x2))
        x1p = np.array(x1[:lenx], dtype='int16').reshape(-1, 1)
        x2p = np.array(x2[:lenx], dtype='int16').reshape(-1, 1)

        # add vectors to list
        labels.append(x1p)
        feats.append(x2p)

    # # train and test DT model
    # tree_model = DecisionTreeRegressor()
    # tree_model.fit(feats, labels)
    # prediction = tree_model.predict(feats[0])
    #
    # # write wav file from prediction
    # functions.writeWav(prediction, "data/prediction/output.wav")


main()
