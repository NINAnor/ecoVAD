#!/usr/bin/env python3

"""
Script that parse multiple .json files into a dataframe.
Each row of the dataset corresponds to 1 file and has columns for
the number of detections and all the temporal information of the file
"""

import argparse
import json
import glob
import pandas as pd
import yaml

from yaml import FullLoader

def get_list(json_file):
    """Return a list containing nb of detections & temporal information & number of detections"""

    # Get the name of the audio file
    file_name = json_file.split('/')[-1].split('.')[0] + '.' + json_file.split('.')[1]

    # Isolate the temporal info
    ymd_hms = json_file.split("/")[-1].split(".")[0].split("json")[0]
    ymdhms = ymd_hms.replace("_", "")
 
    # List the start - end of each detection
    start_end = []

    with open(json_file) as f:
        data = json.load(f)

        # Loop through the detections
        for detection in range(len(data["content"])):

            # start - end
            start = data["content"][detection]["start"]
            end = data["content"][detection]["end"]
            start_end.append([file_name, ymdhms, start, end])

    # Build the dataframe
    df = pd.DataFrame(start_end, columns = ['json_file', 'date', 'start', 'end'])

    return df

def get_df(list_files, output):
    """
    Return a dataframe containing nb of detections & temporal information & number of detections for
    a list of files
    """

    # Initiate empty DF
    small_dfs = []

    # Loop through the files
    for file in list_files:
        # Get the detections
        det = get_list(file)
        # Append to the df
        small_dfs.append(det)

    # concatenate all small df
    df = pd.concat(small_dfs)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Write the output
    df.to_csv(output, index=False)


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

    # Get all the files in the input folder
    list_files = glob.glob(cfg["PATH_JSON_DETECTIONS"] + "**/*.json", recursive=True)

    # Get the DF as a csv
    get_df(list_files, cfg["PATH_PARSED_JSON"])
