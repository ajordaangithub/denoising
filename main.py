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
import noisereduce as nr
import random
import functions
import sys


def main():
    # load files
    files = functions.read_files("data/clean/")

    feats, labels = list(), list()

    for f in files:
        # get sample rate and vector
        sr1, x1 = wavfile.read("data/clean/" + f)
        sr2, x2 = wavfile.read("data/perturbed/" + f[:-4] + "_aug.wav")

        # preprocessing
        lenx = np.minimum(len(x1), len(x2))
        clean = x1[:lenx].astype(np.float32)
        perturbed = x2[:lenx].astype(np.float32)
        
        # perform denoising
        result = functions.denoise(perturbed, clean)

        # write wav file
        functions.writeWav(result.astype(np.int16), "data/output/output.wav")
        functions.writeWav(x2, "data/output/original.wav") # for comparing
        sys.exit()
  

main()
