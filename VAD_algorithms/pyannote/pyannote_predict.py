#!/usr/bin/env python3

import argparse
import json
import glob
import os
import yaml

from yaml import FullLoader
from pyannote.audio import Pipeline

class PyannotePredict():
    """
    Function that take a file and returns a csv composed of the name of the file and the speech duration.
    file_path: path to the file that needs to be processed by Pyannote
    save_folder: name of the folder where to save the csv file
    save_file: name of the csv file
    """

    def __init__(self, pipeline, file_path, save_file):

        self.file_path = file_path
        self.save_file = save_file
        self.pipeline = pipeline

    def get_preds(self):

        output = self.pipeline(self.file_path)
        detections = output.for_json()

        list_detections = []

        for det in detections['content']:
            list_detections.append(det['segment'])

        return list_detections

    def main(self):

        speech = self.get_preds()

        with open(self.save_file + '.json', 'w') as outfile:
            json.dump(speech, outfile)


if __name__ == '__main__':

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

    # Load the pyannote model
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

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
        PyannotePredict(pipeline, audiofile, out_path).main()