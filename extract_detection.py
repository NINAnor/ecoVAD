#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import yaml
import glob

from yaml import FullLoader
from pydub import AudioSegment

from utils.parse_json import get_df

def get_sample_detections(base_folder, input_df, output, nb_sample_detections, random_state):

    f = pd.read_csv(input_df)
    random_sample = f.sample(n=nb_sample_detections, random_state=random_state)

    for row in range(len(random_sample)):

        # Extract the information from the row
        audio_file = random_sample.iloc[row]['json_file']
        start_detection = random_sample.iloc[row]['start'] * 1000
        end_detection = random_sample.iloc[row]['end'] * 1000

        # Get name of the segment
        extension = ['.wav','.WAV','.mp3']

        for ext in extension:
            s_name = audio_file.split(".")[0] + ext
            audio_file_path = os.path.join(base_folder, s_name)

            if os.path.exists(audio_file_path):
                # Extract the detection
                audio = AudioSegment.from_file(audio_file_path)
                detection = audio[start_detection:end_detection]

                # Save the file
                audio_file_save = audio_file.split(".")[0]
                detection.export(os.path.join(output, audio_file_save + "-{}-{}.wav".format(round(start_detection, 2),
                                                                                        round(end_detection, 2))),
                                format='wav')
            else:
                continue

if __name__=='__main__':

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

    # Get all the files in the input folder
    list_files = glob.glob(os.sep.join([cfg["PATH_JSON_DETECTIONS"], "**/*.json"]), recursive=True)

    # Get the DF as a csv
    get_df(list_files, cfg["PATH_PARSED_JSON"])

    # Sample the detections
    get_sample_detections(cfg["PATH_INPUT_DATA"],
                          cfg["PATH_PARSED_JSON"],
                          cfg["PATH_SAMPLE_DETECTIONS"],
                          cfg["NUMBER_OF_SAMPLES"],
                          cfg["RANDOM_SEED"])