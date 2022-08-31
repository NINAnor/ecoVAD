#!/usr/bin/env python3

import numpy as np
import argparse
import librosa
import torch
import json
import glob
import os
import yaml

from yaml import FullLoader
from functools import reduce
from torch.utils.data import DataLoader
from pydub import AudioSegment

from VAD_algorithms.ecovad.VGG11 import VGG11

class ecoVADpredict():

    def __init__(self, input, output, model_path, threshold, use_gpu=False):

        # arguments
        self.input = input
        self.model_path = model_path
        self.output = output

        # related to the hardware
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # related to the neural network
        self.model = self.initModel()
        self.batch_size = 128

        # related to audio
        self.soundscape = self.initSoundscape()
        self.soundscapeSegments = self.soundscape2segments()

        # Threshold for being counted as detection
        self.threshold = threshold

        # metrics
        self.METRICS_SIZE = 1
        self.METRICS_LABELS_NDX = 0

    def merge_detected(self, partial, current):
        """Util function that squash the dictionnary of detections"""
        if not partial:
            return [current]
        previous = partial[-1]
        if previous['end'] == current['start']:
            partial[-1] = {'start': previous['start'], 'end': current['end']}
        else:
            partial.append(current)
        return partial

    def to_mel_spectrogram(self, x):

        sgram = librosa.stft(x, n_fft=1024, hop_length=376)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=16000, n_mels=128)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)
        return mel_sgram

    def normalize_row_matrix(self, mat):

        mean_rows = mat.mean(axis=1)
        std_rows = mat.std(axis=1)
        normalized_array = (mat - mean_rows[:, np.newaxis]) / std_rows[:, np.newaxis]
        return normalized_array

    def initModel(self):

        model = VGG11()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.double().to(self.device)
        model.eval()

        return model

    def initSoundscape(self):

        soundscape = AudioSegment.from_file(self.input)
        soundscape = soundscape.set_frame_rate(16000)
        return soundscape

    def soundscape2segments(self):

        arr = np.arange(0, len(self.soundscape), 1000)
        segments = []

        for i in range(len(arr) - 3):
            segment = self.soundscape[arr[i]:arr[i + 3]]
            segments.append(segment)

        return segments

    def initDataset(self):

        list_segments_tensor = [np.array(segment.get_array_of_samples(), dtype=float) for segment in
                                self.soundscapeSegments]
        list_segments_mel = [self.to_mel_spectrogram(segment) for segment in list_segments_tensor]
        list_segment_mel_norm = [self.normalize_row_matrix(segment) for segment in list_segments_mel]
        list_segment_mel_norm = torch.tensor(list_segment_mel_norm)
        list_segment_mel_norm = list_segment_mel_norm.unsqueeze(1)

        # Put all the tensors in a dataloader
        predLoader = DataLoader(list_segment_mel_norm,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=self.use_gpu
                                )

        return predLoader

    def do_prediction(self, dl, metrics_mat):

        with torch.no_grad():
            for batch_ndx, batch_mel in enumerate(dl):
                inputs = batch_mel.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze(1)

                start_ndx = batch_ndx * self.batch_size
                end_ndx = start_ndx + inputs.size(0)

                metrics_mat[self.METRICS_LABELS_NDX, start_ndx:end_ndx] = outputs.detach()

    def write_json(self, prob_array, threshold):

        detections = []

        det_ixs = np.where((prob_array > threshold))[0]

        for det in range(len(det_ixs) - 1):
            dic = {'start': int(det_ixs[det]), 'end': int(det_ixs[det] + 1)}
            detections.append(dic)

        merged = reduce(self.merge_detected, detections, [])

        data = {'ecoVAD': "Timeline", 'content': merged}

        with open(self.output + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def main(self):

        pred_dl = self.initDataset()
        pred_metrics = torch.zeros(self.METRICS_SIZE, len(pred_dl.dataset), device=self.device)

        self.do_prediction(pred_dl, pred_metrics)

        # Write dictionary - keys = seconds, values = probability of speech
        pred_array = np.array(pred_metrics.cpu()).squeeze(0)

        # Write JSON file
        self.write_json(pred_array, self.threshold)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="./config_inference.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # List the folder with files that needs predictions
    # List the folder with files that needs predictions
    types = ('/**/*.WAV', '/**/*.wav', '/**/*.mp3') # the tuple of file types
    audiofiles= []
    for files in types:
        audiofiles.extend(glob.glob(cfg["PATH_INPUT_DATA"] + files, recursive=True))
    print("Found {} files to analyze".format(len(audiofiles)))

    # Make the prediction
    for audiofile in audiofiles:
        out_name = audiofile.split('/')[-1].split('.')[0]
        out_path = os.sep.join([cfg["PATH_JSON_DETECTIONS"],  out_name])
 
        ecoVADpredict(audiofile, 
                    out_path,
                    cfg["ECOVAD_WEIGHTS_PATH"].model_path,
                    cfg["THRESHOLD"], 
                    use_gpu=cfg["USE_GPU"]).main()