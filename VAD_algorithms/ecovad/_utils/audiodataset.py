"""
Define a class AudioDataset which take the folder of the training / test data as input
and make normalized mel-spectrograms out of it

Labels are also one hot encoded
"""

from torch.utils.data import Dataset
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder

import numpy as np
import librosa
import torch
import os

from utils.process_audio import openAudioFile

class AudioDataset(Dataset):
    def __init__(self, data_root, n_fft, hop_length, n_mels, sr=16000):
        self.data_root = data_root.split(",") # If going to have multiple train / test folders (OL + Forest for ex.)
        self.samples = []
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.class_encode = LabelEncoder()
        self.sr = sr
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        audio_filepath, label = self.samples[idx]

        arr, sr = openAudioFile(audio_filepath, self.sr)

        mel = self.to_mel_spectrogram(arr)
        #mel_norm = self.normalize_row_matrix(mel) 
        # When using Librosa, this leads to NAs. With Librosa the array is already normalised
        mel_norm_tensor = torch.tensor(mel)
        mel_norm_tensor = mel_norm_tensor.unsqueeze(0)

        label_encoded = self.one_hot_sample(label)
        label_class = torch.argmax(label_encoded)

        return (mel_norm_tensor, label_class)

    def _init_dataset(self):

        folder_names = set()

        for folder_root in self.data_root:

            # Get the label (i.e. folder name / speech & no_speech) so we can one hot encode it
            for folder_label in os.listdir(folder_root):
                path_file = os.path.join(folder_root, folder_label)

                # Check if path_file is a folder containing the training data
                if os.path.isdir(path_file):
                    folder_names.add(folder_label)

                    # List all the files in the "labeled" folders. Path will be stored in samples
                    for audio_file in os.listdir(path_file):
                        if audio_file.endswith(('.wav', '.WAV', '.mp3')):
                            audio_filepath = os.path.join(path_file, audio_file)
                            self.samples.append((audio_filepath, folder_label))
                        else: 
                            continue
                else:
                    continue

        self.class_encode.fit(list(folder_names))

    def to_mel_spectrogram(self, x):

        sgram = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=self.sr, n_mels=self.n_mels)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)
        return mel_sgram

    def normalize_row_matrix(self, mat):

        mean_rows = mat.mean(axis=1)
        std_rows = mat.std(axis=1)
        normalized_array = (mat - mean_rows[:, np.newaxis]) / std_rows[:, np.newaxis]
        return normalized_array

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]

    def one_hot_sample(self, label):
        t_label = self.to_one_hot(self.class_encode, [label])
        return t_label