#!/usr/bin/env python3

import argparse
import numpy as np
import soundfile
import glob
import yaml
import os

from yaml import FullLoader
from audiomentations import Compose, AddBackgroundNoise, Shift, Gain

from utils.process_audio import openAudioFile, splitSignal

RANDOM = np.random.RandomState(42)

def make_training_val_dir(path_to_save):

    # Check if a folder "speech" exists and create one if not
    if not os.path.exists(os.sep.join([path_to_save, 'speech'])):
        os.makedirs(os.sep.join([path_to_save, 'speech']))

    # Check if a folder "no_speak" exists and create one if not
    if not os.path.exists(os.sep.join([path_to_save, 'no_speech'])):
        os.makedirs(os.sep.join([path_to_save, 'no_speech']))

def preprocess_with_one_sound(arr, sr, directory, p_sound):

    preprocess = Compose([
        AddBackgroundNoise(sounds_path=directory, min_absolute_rms_in_db=-56.16, max_absolute_rms_in_db=-8.3, p=p_sound, noise_rms = "absolute",
                            noise_transform=Shift(min_fraction=0.5,max_fraction=0.5, p=0.5, fade=True, rollover=True)),
    ])

    processed_segment = preprocess(samples=arr, sample_rate=sr)

    return processed_segment

def preprocess_with_two_sounds(arr, sr, beta, directory_first_sound, directory_second_sound, p_sound_1, p_sound_2):

    preprocess = Compose([
        AddBackgroundNoise(sounds_path=directory_first_sound, min_absolute_rms_in_db=-56.16, max_absolute_rms_in_db=-8.3, p=p_sound_1, noise_rms = "absolute", 
                            noise_transform=Compose([
                                Gain(min_gain_in_db=beta, max_gain_in_db=beta, p=1),
                                Shift(min_fraction=0.5,max_fraction=0.5, p=0.5, fade=True, rollover=True)
                            ])
                            ),
        AddBackgroundNoise(sounds_path=directory_second_sound, min_absolute_rms_in_db=-56.16, max_absolute_rms_in_db=-8.3, p=p_sound_2, noise_rms = "absolute", 
                            noise_transform=Compose([
                                Gain(min_gain_in_db=(1-beta), max_gain_in_db=(1-beta), p=1),
                                Shift(min_fraction=0.5,max_fraction=0.5, p=0.5, fade=True, rollover=True)
                            ])
                            )
    ])

    processed_segment = preprocess(samples=arr, sample_rate=sr)

    return processed_segment

def mix_audio(arr, sr, speech_dir, noise_dir, proba_speech, proba_noise_speech, proba_noise_nospeech, return_label=False):

    binom_speech = np.random.binomial(n=1, p=proba_speech, size =1)

    # Human speech added
    if binom_speech==1:
        binom_bg_noise = np.random.binomial(n=1, p=proba_noise_speech, size =1)

        if binom_bg_noise==1:
            # Human noise (p=1) + background noise (p=proba_speech_noise)
            beta = np.random.uniform(low=0.1, high=0.9, size=1)
            sound = preprocess_with_two_sounds(arr, sr, beta, speech_dir, noise_dir, p_sound_1=1, p_sound_2=proba_noise_speech)
            label = "speech+noise"

        else:
            # Human noise only (p=1)
            sound = preprocess_with_one_sound(arr, sr, speech_dir, p_sound=1)
            label = "speech"

    # No human speech added
    else:
        binom_bg_noise = np.random.binomial(n=1, p=proba_noise_nospeech, size =1)

        if binom_bg_noise==1:
            # Background noise added (p=proba_noise_no_speech)
            sound = preprocess_with_one_sound(arr, sr, noise_dir, p_sound=proba_noise_nospeech)
            label = "nospeech+noise"

        else:
            # No addition of any noise
            sound = arr
            label = "nospeech"

    return sound, label

def preprocess_file(audio_path, length_segments, overlap, min_length,
                    speech_dir, noise_dir, proba_speech, proba_noise_speech, proba_noise_nospeech):
 
    processed_arrays = []

    arr, sr = openAudioFile(audio_path)
    split_sig = splitSignal(arr, sr, length_segments, overlap, min_length)

    for arr in split_sig:
        processed_array, label = mix_audio(arr, sr, speech_dir, noise_dir, proba_speech, proba_noise_speech, proba_noise_nospeech)
        processed_arrays.append((processed_array, label))

    return processed_arrays, sr

def save_processed_arrays(audio_path, save_dir, list_processed_arrays, sr):

    save_name = audio_path.split("/")[-1].split(".")[0]

    for i, arr_label in enumerate(list_processed_arrays):

        processed_arr, label = arr_label

        if label.split("+")[0] == "speech":
            soundfile.write(os.sep.join([save_dir, "speech", "{}_{}.wav".format(save_name, i)]), processed_arr, sr)
        else:
            soundfile.write(os.sep.join([save_dir, "no_speech", "{}_{}.wav".format(save_name, i)]), processed_arr, sr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="./config_training.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    make_training_val_dir(cfg["AUDIO_OUT_DIR"])

    # List the folder with files that needs predictions
    types = ('/**/*.WAV', '/**/*.wav', '/**/*.mp3') # the tuple of file types
    audiofiles= []
    for files in types:
        audiofiles.extend(glob.glob(cfg["AUDIO_PATH"] + files, recursive=True))
    print("Found {} files to analyze".format(len(audiofiles)))

    for file in audiofiles:
        processed_arr, sr = preprocess_file(file, 
                        cfg["LENGTH_SEGMENTS"], 
                        overlap = 0, 
                        min_length = cfg["LENGTH_SEGMENTS"],
                        speech_dir=cfg["SPEECH_DIR"],
                        noise_dir=cfg["NOISE_DIR"],
                        proba_speech=cfg["PROBA_SPEECH"],
                        proba_noise_speech=cfg["PROBA_NOISE_WHEN_SPEECH"],
                        proba_noise_nospeech=cfg["PROBA_NOISE_WHEN_NO_SPEECH"])

        save_processed_arrays(file, cfg["AUDIO_OUT_DIR"], processed_arr, sr)