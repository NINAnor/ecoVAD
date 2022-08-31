import os
import soundfile
import numpy as np
import torch

from torch.utils.data import DataLoader
from math import floor
from shutil import rmtree

from VAD_algorithms.ecovad._utils.audiodataset import AudioDataset

def generate_sound(number_of_samples, save_dir):
    # Generate 3s at 44100kHz seconds of dummy audio for the sake of example
    for i in range(number_of_samples):
        samples = np.random.uniform(low=-0.2, high=0.2, size=(44100*3,)).astype(np.float32)
        soundfile.write(os.sep.join([save_dir, "test_segment_{}.wav".format(i)]), samples, 44100)

def populate_temp_folders():

    temp_folders = ["./tmp/speech", "./tmp/no_speech"]

    for folder in temp_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
        generate_sound(200, folder)

def test_audiodataset():

    # Parameters
    n_fft = 1024
    hop_length = 376
    n_mels = 128
    batch_size = 1
    num_workers = 4
    sr = 44100
    n_seconds = 3
    data_path = "./tmp"

    populate_temp_folders()

    dataset = AudioDataset(data_path, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, sr = sr)
    dloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers= num_workers, pin_memory=False)

    for tensor, label in dloader:
        # test the shape of the tensor
        assert tensor.shape == torch.Size([1, 1, n_mels, floor((sr*n_seconds) / hop_length) + 1])
        # test the type of the label
        assert label.dtype == torch.int64

    # End of test, remove the temp dir
    rmtree("./tmp/")