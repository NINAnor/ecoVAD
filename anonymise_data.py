#!/usr/bin/env python3

import argparse
import glob
import yaml
import os
import soundfile
import json

from yaml import FullLoader
from pyannote.audio import Pipeline

from VAD_algorithms.pyannote.pyannote_predict import PyannotePredict
from VAD_algorithms.ecovad.ecoVAD_predict import ecoVADpredict
from VAD_algorithms.webrtcvad.webrtc_predict import WebrtcvadPredict

from utils.process_audio import openAudioFile

def remove_extension(input):

    filename = input.split("/")[-1].split(".")

    if len(filename) > 2:
        filename = ".".join(filename[0:-1])
    else:
        filename = input.split("/")[-1].split(".")[0]

    return filename

def parseFolders(apath, rpath):

    audio_files = [f for f in glob.glob(apath + "/**/*", recursive = True) if os.path.isfile(f)]
    audio_no_extension = []
    for audio_file in audio_files:
        audio_file_no_extension = remove_extension(audio_file)
        audio_no_extension.append(audio_file_no_extension)


    result_files = [f for f in glob.glob(rpath + "/**/*", recursive = True) if os.path.isfile(f)]

    flist = []
    for result in result_files:
        result_no_extension = remove_extension(result)
        is_in = result_no_extension in audio_no_extension

        if is_in:
            audio_idx = audio_no_extension.index(result_no_extension)
            pair = {'audio': audio_files[audio_idx], 'result': result}
            flist.append(pair)
        else:
            continue

    print('Found {} audio files with valid result file.'.format(len(flist)))

    return flist

def audio_anonymisation(json_file, audio_file):

        # Open the JSON containing the detections
        with open(json_file) as f:
            data = json.load(f)

        # Open the audio file
        arr, sr = openAudioFile(audio_file, sample_rate=44100)

        # Start the anonymization loop
        for i in range(len(data["content"])):
            s = int(data["content"][i]["start"] * sr)
            e = int(data["content"][i]["end"] * sr)
            arr[s:e] = 0

        # return anonymised array and sr for saving
        return (arr, sr)

def infer_detections(VAD_model, audiofile, out_path, cfg):

    if VAD_model == "pyannote":
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
        PyannotePredict(pipeline, audiofile, out_path).main()

    elif VAD_model == "ecovad":
        ecoVADpredict(audiofile, 
            out_path,
            cfg["ECOVAD_WEIGHTS_PATH"],
            cfg["THRESHOLD"],
            cfg["USE_GPU"]).main()

    elif VAD_model == "webrtcvad":
        WebrtcvadPredict(audiofile, 
                        out_path,
                        cfg["FRAME_LENGTH"],
                        cfg["AGGRESSIVENESS"]).main()
    else:
        print("Please choose a correct VAD model")

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

    ######################
    # Get the detections #
    ######################
    types = ('/**/*.WAV', '/**/*.wav', '/**/*.mp3') # the tuple of file types
    audiofiles= []
    for files in types:
        audiofiles.extend(glob.glob(cfg["PATH_INPUT_DATA"] + files, recursive=True))
    print("Found {} files to analyze".format(len(audiofiles)))

    for audiofile in audiofiles:
        out_name = audiofile.split('/')[-1].split('.')[0]
        out_path = os.sep.join([cfg["PATH_JSON_DETECTIONS"], out_name])
        infer_detections(cfg["CHOSEN_VAD"], audiofile, out_path, cfg)

    #########################
    # Anonymise the dataset #
    #########################
    parsed_folders = parseFolders(cfg["PATH_INPUT_DATA"], cfg["PATH_JSON_DETECTIONS"])

    # Anonymise the files
    for i in range(len(parsed_folders)):

        afile = parsed_folders[i]['audio']
        rfile = parsed_folders[i]['result']

        audio_name = afile.split("/")[-1].split(".")[0]
        save_name = os.sep.join([cfg["PATH_ANONYMIZED_DATA"], "ANONYMISED_" + audio_name])

        anonymised_arr, sr = audio_anonymisation(rfile, afile)
        soundfile.write(save_name + ".wav", anonymised_arr, sr)