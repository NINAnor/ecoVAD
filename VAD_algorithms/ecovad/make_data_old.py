#!/usr/bin/env python3

"""
Function that mix the sound of the ecosystem with another dataset.
Data augmentation occur in this function
"""

import argparse
import numpy as np
import xlsxwriter
import os
import yaml
import glob

from yaml import FullLoader
from pydub import AudioSegment
from pydub.generators import WhiteNoise

from VAD_algorithms.ecovad._utils.augment_audio import mix_audio
from VAD_algorithms.ecovad._utils.augment_audio import match_target_amplitude
from VAD_algorithms.ecovad._utils.augment_audio import get_random_segment

#######################
# Define the function #
#######################

def mix_at_random(soundscape_path, overlay_dataset, overlay_dataset_background, length_segments, proba, path_to_save, xlsx_doc, include_noises, include_soundscape):

    if include_noises == 1:
        p_incl_bg_nospeech=0.9
        p_incl_bg_speech=0.5
    else:
        p_incl_bg_nospeech=0
        p_incl_bg_speech=0

    # Get the dataset from the specified path
    print("Processing file: {}".format(soundscape_path))
    soundscape = AudioSegment.from_file(soundscape_path)

    # If not using soundscape replace with white noise of same RMS
    if not include_soundscape == 0:
        noise = WhiteNoise().to_audio_segment(duration=len(soundscape), volume=-50)
        soundscape = noise

    # Decompose the soundscape into segments of 3 seconds
    # We set a +1 to include the last element
    ecosystem_decomp = np.arange(0, len(soundscape) + 0.01, length_segments)

    # Check if a folder "speech" exists and create one if not
    if not os.path.exists(os.sep.join([path_to_save, 'speech'])):
        os.makedirs(os.sep.join([path_to_save, 'speech']))

    # Check if a folder "no_speak" exists and create one if not
    if not os.path.exists(os.sep.join([path_to_save, 'no_speech'])):
        os.makedirs(os.sep.join([path_to_save, 'no_speech']))

    # Compute whether the segment should be mixed with a voice
    binom_seq = np.random.binomial(n=1, p=proba, size = len(ecosystem_decomp) - 1) # -1 or I have 1 elements too much

    # elements for the XLSX document
    is_speech = []
    segment_l = []
    alpha_l = []
    beta_l = []
    offset_l = []
    speech_p = []
    bg_p = []

    # Loop through each segment of the ecosystem dataset
    for i in range(len(ecosystem_decomp) - 1):

        # take the segment beginning at i and finishing at i+1
        start = ecosystem_decomp[i]
        end = ecosystem_decomp[i+1]
        segment = soundscape[start:end]

        # If the binomial list at i is 1 then we add the human voice
        if binom_seq[i] == 1:

            # Select a random file in the overlay dataset
            speech, speech_path = get_random_segment(overlay_dataset, length_segments)

            # In addition to the speech, should we include another BG sound?
            include_bg = np.random.binomial(n=1, p=p_incl_bg_speech, size = 1)

            if include_bg == 1:

                # Select a random file in the overlay dataset
                bg, bg_path = get_random_segment(overlay_dataset_background, length_segments)

                # Overlay the speech with the BG noise while changing the SNR
                speech_augmented, alpha = mix_audio(speech, bg, aug=True)
                # Speech augmented is full scale -> need to rescale it to avoid clipping
                # We set the max loudness at -8.3: 99th percentile of the max / -56.16 min rms
                beta = np.random.uniform(-8.3, -56.16)
                speech_augmented = match_target_amplitude(speech_augmented, beta)

                # Chance that the speech begin "before" or begin "after" -> offset
                chance_offset = np.random.binomial(n=1, p=0.50, size = 1)

                if chance_offset == 1: # Simulate speech that began before
                    offset = np.random.randint(1000,3000)
                    speech_augmented = speech_augmented[0:offset]
                    combined = segment.overlay(speech_augmented)

                else:
                    offset = np.random.randint(0,2000)
                    combined = segment.overlay(speech_augmented, position=offset)

            else:
                # Only change the volume of the speech and add an offset
                bg_path = "None"
                offset = np.random.randint(0,2000)
                alpha = 0
                beta = np.random.uniform(-8.3, -56.16)
                speech_augmented = match_target_amplitude(speech, beta)
                combined = segment.overlay(speech_augmented, position=offset)

            # export the segment into the "speak" folder
            is_speech.append(1)
            segment_l.append("{}_segment_{}".format(path_to_save.split("/")[-1].split(".")[0], i))
            alpha_l.append(alpha)
            beta_l.append(beta)
            offset_l.append(offset)
            bg_p.append(bg_path)
            speech_p.append(speech_path)

            combined_save_dir = os.path.join(path_to_save, 'speech')

        # Else, keep the ecosystem segment as it is (without human voice) or add some noise
        else:

            include_bg = np.random.binomial(n=1, p=p_incl_bg_nospeech, size = 1)

            if include_bg == 1:

                # Select a random file in the overlay dataset
                bg, bg_path = get_random_segment(overlay_dataset_background, length_segments)
                beta = np.random.uniform(-8.3, -56.16)
                bg_augmented = match_target_amplitude(bg, beta)

                # Chance that the speech "began before" or begin "after" -> offset
                chance_offset = np.random.binomial(n=1, p=0.50, size = 1)

                if chance_offset == 1: # Simulate speech that began before
                    offset = np.random.randint(1000,3000)
                    speech_augmented = bg_augmented[0:offset]
                    combined = segment.overlay(bg_augmented)

                else:
                    offset = np.random.randint(0,2000)
                    combined = segment.overlay(bg_augmented, position=offset)

            else:
                offset = 0
                beta = 0
                bg_path = "None"
                combined = segment

            # export the segment into the "no speach" folder
            is_speech.append(0)
            segment_l.append("{}_segment_{}".format(path_to_save.split("/")[-1].split(".")[0], i))
            beta_l.append(beta)
            alpha_l.append(0)
            offset_l.append(offset)
            bg_p.append(bg_path)
            speech_p.append("None")

            combined_save_dir = os.path.join(path_to_save, 'no_speech')

        # Save combined audio to the
        name_file = soundscape_path.split("/")[-1].split(".")[0]
        combined.export(os.path.join(combined_save_dir, '{}_segment_{}.wav'.format(name_file, i)), format="wav")

        # Write the XLSX doc
        workbook = xlsxwriter.Workbook(xlsx_doc)
        worksheet = workbook.add_worksheet()

        # Add bold to highlight cells
        bold = workbook.add_format({'bold': True})

        # Start from the first cell.
        # Rows and columns are zero indexed.
        row = 1
        n = len(segment_l) + 1

        # Write headers
        worksheet.write('A1', 'file_name', bold)
        worksheet.write('B1', 'is_speech', bold)
        worksheet.write('C1', 'alpha', bold)
        worksheet.write('D1', 'beta', bold)
        worksheet.write('E1', 'offset', bold)
        worksheet.write('F1', 'background', bold)
        worksheet.write('G1', 'speech', bold)

        # iterating through content list
        for i in range(len(segment_l)):

            # write operation perform
            worksheet.write(row, 0, segment_l[i])
            worksheet.write(row, 1, is_speech[i])
            worksheet.write(row, 2, alpha_l[i])
            worksheet.write(row, 3, beta_l[i])
            worksheet.write(row, 4, offset_l[i])
            worksheet.write(row, 5, bg_p[i])
            worksheet.write(row, 6, speech_p[i])

            # incrementing the value of row by one
            # with each iteratons.
            row += 1

        workbook.close()


####################
# Run the function #
####################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="./config_training.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    list_audio_files = glob.glob(cfg["AUDIO_PATH"] + "/*")
    print("Found {} files to split into training segments".format(len(list_audio_files)))

    for file in list_audio_files:
        mix_at_random(file,
                cfg["SPEECH_DIR"],
                cfg["NOISE_DIR"],
                cfg["LENGTH_SEGMENTS"],
                cfg["PROBA_SPEECH"],
                cfg["AUDIO_OUT_DIR"],
                cfg["METADATA_FILE"],
                cfg["INCLUDE_NOISES"],
                cfg["INCLUDE_SOUNDSCAPE"]
                )