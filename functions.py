import numpy as np
import re
from scipy.io import wavfile
from scipy import signal
import subprocess
import matplotlib.pyplot as plt
from IPython.display import Audio
from os import listdir  # to read files
from os.path import isfile, join  # to read files
import noisereduce as nr


def denoise(perturbed, clean):
    reduced_noise = nr.reduce_noise(audio_clip=clean, noise_clip=perturbed, verbose=True)
    return denoise


def specSimilarity(cleanFilePath, augFilePath):
    '''
    Computes the similarity between two audio files.
    Make sure the sample rate of the two files are equal (16 kHz)

    Input:
    - cleanFilePath: path to clean wav audio file
    - augFilePath: path to augmented wav audio file

    Output: similarity of the (square-root) of the two corresponding
    spectrograms, computed by taking the average of the values obtained
    by dividing the minimum value by the maximum value for each pixel
    '''

    # Get audio vectors
    sr1, x1 = wavfile.read(cleanFilePath)
    sr2, x2 = wavfile.read(augFilePath)
    if sr1 != sr2:
        print('Sample rates are unequal')
        return

    lenx = np.minimum(len(x1), len(x2))
    x1 = np.array(x1[:lenx], dtype='int64')
    x2 = np.array(x2[:lenx], dtype='int64')

    # Normalize volume of the augmented track
    rms_1 = np.sqrt(np.mean(x1 ** 2))
    rms_2 = np.sqrt(np.mean(x2 ** 2))
    if rms_2 > 0:
        x2 = rms_1 / rms_2 * x2

    # Compute similarity scores
    f, t, specComplex1 = signal.stft(x1, fs=sr1, nperseg=2048)
    f, t, specComplex2 = signal.stft(x2, fs=sr2, nperseg=2048)
    minSpec = np.minimum(np.abs(specComplex1), np.abs(specComplex2))
    maxSpec = np.maximum(np.abs(specComplex1), np.abs(specComplex2))

    return np.sum(minSpec ** .5) / np.sum(maxSpec ** .5)


def denoisedScore(cleanFilePath, augFilePath, denoisedFilePath):
    '''
    Computes the cleanness score of the denoised file.

    Input:
    - cleanFilePath: path to clean wav audio file
    - augFilePath: path to augmented wav audio file
    - denoisedFilePath: path to denoised wav audio file
    '''

    sim_clean_aug = specSimilarity(cleanFilePath, augFilePath)
    sim_clean_den = specSimilarity(cleanFilePath, denoisedFilePath)
    return (sim_clean_den - sim_clean_aug) / (1.0 - sim_clean_aug)


def showWaveform(audioFilePath):
    '''
    Show the waveform representation of an audio file
    '''

    sr, x = wavfile.read(audioFilePath)
    t = np.linspace(0, len(x) / sr, len(x), endpoint=False)
    fig = plt.figure(figsize=(20, 5))
    plt.plot(t, x)
    plt.show()


def showSpec(audioFilePath, expo=.5, maxFreq=5000, cutFactorAmp=.5):
    '''
    Show the spectrogram representation of an audio file
    '''

    plt.figure(figsize=(20, 5))
    sr, x = wavfile.read(audioFilePath)
    f, t, specComplex = signal.stft(x, fs=sr, nperseg=2048)  # spectrogram
    spec = np.abs(specComplex)
    ZxxMax = np.max(spec ** expo)
    plt.pcolormesh(t, f, spec ** expo, vmin=0, vmax=cutFactorAmp * ZxxMax, shading='gouraud')
    plt.ylim([0, maxFreq])
    plt.title('STFT Magnitude')
    plt.ylabel('STFT Magnitude')
    plt.xlabel('Time [sec]')
    plt.yscale('linear')
    plt.show()


def mp3ToWav(audioFilePath):
    '''
    Converts non-wav files to 16kHz wav files and stores them
    in the same directory as the original file path.

    Requires to have ffmpeg installed.
    '''

    newPath = re.sub('\.[a-zA-Z0-9]+$', '.wav', audioFilePath)
    command = 'ffmpeg -i  "' + audioFilePath + '" -ac 1 -ar 16000 "' + newPath + '"'
    print(command)
    subprocess.call(command)

def writeWav(Array, filename):
    wavfile.write(filename, 16000, Array)

def read_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]
