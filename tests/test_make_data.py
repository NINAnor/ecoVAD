import numpy as np
import pytest
import glob
import soundfile
import os
import shutil

from collections import Counter

from VAD_algorithms.ecovad.make_data import preprocess_file

def generate_sound(number_of_samples, save_dir):
    # Generate 3s at 44100kHz seconds of dummy audio for the sake of example
    for i in range(number_of_samples):
        samples = np.random.uniform(low=-0.2, high=0.2, size=(44100*3,)).astype(np.float32)
        soundfile.write(os.sep.join([save_dir, "test_segment_{}.wav".format(i)]), samples, 44100)

def get_proportion(labels):

    label_prop = []
    c = Counter(labels)

    for label, counts in c.items():
        prop = counts / len(labels)
        label_prop.append((label, prop))

    return label_prop

def populate_temp_folders():

    temp_folders = ["./tmp/soundscapes", "./tmp/human_speech", "./tmp/noise"]

    for folder in temp_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
        generate_sound(200, folder)

def get_ideal_proportion(proba_speech, proba_noise_when_speech, proba_noise_when_no_speech):

    ideal_prop = {}

    speech_noise = proba_speech * proba_noise_when_speech
    speech_only = proba_speech * (1-proba_noise_when_speech)

    noise_only = (1 - proba_speech) * proba_noise_when_no_speech
    no_processing = (1 - proba_speech) * (1 - proba_noise_when_no_speech)

    ideal_prop["speech"] = speech_only
    ideal_prop["speech+noise"] = speech_noise
    ideal_prop["nospeech+noise"] = noise_only
    ideal_prop["nospeech"] = no_processing

    return ideal_prop

def test_preprocess_file():
    """
    Check if the percentages of speech / no speech and noise addition is 
    coherent with the probabilities
    """
    proba_speech=0.5
    proba_noise_when_speech=0.5
    proba_noise_when_no_speech=0.90

    populate_temp_folders()

    audiofiles = glob.glob("./tmp/soundscapes/*")

    labels = []

    for file in audiofiles:
        processed_arr, _ = preprocess_file(file, 
                        length_segments=3, 
                        overlap = 0, 
                        min_length = 3,
                        speech_dir="./tmp/human_speech",
                        noise_dir="./tmp/noise",
                        proba_speech=proba_speech,
                        proba_noise_speech=proba_noise_when_speech,
                        proba_noise_nospeech=proba_noise_when_no_speech)

        for arr_label in processed_arr:
            processed_arr, label = arr_label
            labels.append(label)
            assert processed_arr.dtype == np.float32

    label_prop = get_proportion(labels)
    label_ideal_prop = get_ideal_proportion(proba_speech, proba_noise_when_speech, proba_noise_when_no_speech)

    for label, proportion in label_prop:
        if label == "speech":
            assert proportion == pytest.approx(label_ideal_prop["speech"], abs=5e-2)
        elif label == "speech+noise":
            assert proportion == pytest.approx(label_ideal_prop["speech+noise"], abs=5e-2)
        elif label == "nospeech+noise":
            assert proportion == pytest.approx(label_ideal_prop["nospeech+noise"], abs=5e-2)
        elif label == "nospeech":
            assert proportion == pytest.approx(label_ideal_prop["nospeech"], abs=2e-2)

    # End of test, remove the temp dir
    shutil.rmtree("./tmp/")


    
