#!/usr/bin/env python3

import argparse
import yaml
import glob

from yaml import FullLoader

from VAD_algorithms.ecovad.make_data import preprocess_file, save_processed_arrays
from VAD_algorithms.ecovad.train_model import trainingApp

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

    # Prepare the synthetic dataset
    list_audio_files = glob.glob(cfg["AUDIO_PATH"] + "/*")
    print("Found {} files to split into training segments".format(len(list_audio_files)))
    print("Generating the synthetic dataset")

    for file in list_audio_files:
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

    # Train the model
    print("Training the model")
    trainingApp(cfg["TRAIN_VAL_PATH"],
            cfg["MODEL_SAVE_PATH"],
            cfg["CKPT_SAVE_PATH"],
            cfg["BATCH_SIZE"],
            cfg["NUM_EPOCH"],
            cfg["TB_PREFIX"],
            cfg["TB_COMMENT"],
            cfg["LR"],
            cfg["MOMENTUM"],
            cfg["DECAY"],
            cfg["NUM_WORKERS"],
            cfg["USE_GPU"]
            ).main()