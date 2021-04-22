import numpy as np
import re
from scipy.io import wavfile
from scipy import signal
import subprocess
import matplotlib.pyplot as plt
from IPython.display import Audio
import functions

def main():
    sr1, x1 = wavfile.read("data/clean/fn001166_167.wav")
    sr2, x2 = wavfile.read("data/perturbed/fn001166_167_aug.wav")
main()