from ast import parse
import os
import soundfile
import numpy as np
import json

from functools import reduce

from anonymise_data import audio_anonymisation, parseFolders
from utils.process_audio import openAudioFile

def merge_detected(partial, current):
        """Util function that squash the dictionnary of detections"""
        if not partial:
            return [current]
        previous = partial[-1]
        if previous['end'] == current['start']:
            partial[-1] = {'start': previous['start'], 'end': current['end']}
        else:
            partial.append(current)
        return partial

def generate_json_file():
    # Generate a detection jsonfile with random detections

    detections = []
    generate_detections = np.sort(np.random.randint(low=0, high=56, size=4))

    for det in range(len(generate_detections) - 1):
        dic = {'start': int(generate_detections[det]), 'end': int(generate_detections[det] + 1)}
        detections.append(dic)

    merged = reduce(merge_detected, detections, [])

    data = {'TEST': "Timeline", 'content': merged}

    return data

def generate_n_json_files(number_of_json_files, save_dir):

    for i in range(number_of_json_files):
        json_file = generate_json_file()
        save_path = os.sep.join([save_dir, "test_segment_{}".format(i)])
        with open(save_path + '.json', 'w') as outfile:
            json.dump(json_file, outfile)

def generate_sound(number_of_samples, save_dir):
    # Generate 3s at 44100kHz seconds of dummy audio for the sake of example
    for i in range(number_of_samples):
        samples = np.random.uniform(low=-0.2, high=0.2, size=(44100*3,)).astype(np.float32)
        soundfile.write(os.sep.join([save_dir, "test_segment_{}.wav".format(i)]), samples, 44100)

def populate_temp_folders():

    files_to_anonymise = "./tmp/files_to_anonymise"
    json_detection_files = "./tmp/json_detection_files"

    if not os.path.exists(files_to_anonymise):
        os.makedirs(files_to_anonymise)
    generate_sound(200, files_to_anonymise)

    if not os.path.exists(json_detection_files):
        os.makedirs(json_detection_files)
    generate_n_json_files(200, json_detection_files)

def test_audio_anonymisation():

    files_to_anonymise = "./tmp/files_to_anonymise"
    json_detection_files = "./tmp/json_detection_files"

    populate_temp_folders()
    parsed_folders = parseFolders(apath=files_to_anonymise, rpath=json_detection_files)

    for i in range(len(parsed_folders)):

        afile = parsed_folders[i]['audio']
        rfile = parsed_folders[i]['result']

        anonymised_arr, sr = audio_anonymisation(rfile, afile)
        arr, sr = openAudioFile(afile)

        # Compare the length of the anonymised array and the initial array
        assert anonymised_arr.dtype == np.float32
        assert (anonymised_arr.size, sr) == (arr.size, sr)