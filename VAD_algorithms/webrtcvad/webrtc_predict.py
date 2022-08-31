import argparse
import webrtcvad
import wave
import glob
import contextlib
import numpy as np
import json
import yaml
import os

from yaml import FullLoader
from functools import reduce

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        
class WebrtcvadPredict:
    
    def __init__(self, input, output, frame_length, agressiveness):

        self.input = input
        self.output = output
        self.frame_length = frame_length,
        self.agressiveness = agressiveness

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
        
    def read_wave(self, path):
        """Reads a .wav file.
        Takes the path, and returns (PCM audio data, sample rate).
        """
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms[0] / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def get_preds(self, path_to_file):
        """
        Returns the a list of 0 (if no speech for the frame) or 1 (if speech)
        has been detected in the frame
        """

        audio, sample_rate = self.read_wave(path_to_file)
        vad = webrtcvad.Vad(self.agressiveness)
        frames = self.frame_generator(self.frame_length, audio, sample_rate) # frames of 30ms
        frames = list(frames)

        f = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            f.append(is_speech)

        return f

    def write_json(self, is_speech_array):

        detections = []
        filtered_merged = []
        det_ixs = np.where(is_speech_array)[0]

        for det in range(len(det_ixs) - 1):

            s = (int(det_ixs[det]) * self.frame_length[0]) / 1000
            e = ((int(det_ixs[det]) + 1) * self.frame_length[0]) / 1000

            dic = {'start': s, 'end': e}
            detections.append(dic)
 
        merged = reduce(self.merge_detected, detections, [])

        for i, item in enumerate(merged):

            diff = item['end'] - item['start']
            
            if diff > 1.0: 
                filtered_merged.append(merged[i])
            else:
                continue

        data = {'webrtc': "Timeline", 'content': filtered_merged}

        with open(self.output + '.json', 'w') as outfile:
            json.dump(data, outfile)

    def main(self):

        preds = self.get_preds(self.input)
        self.write_json(preds)

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

    # List the folder with files that needs predictions
    types = ('/**/*.WAV', '/**/*.wav', '/**/*.mp3') # the tuple of file types
    audiofiles= []
    for files in types:
        audiofiles.extend(glob.glob(cli_args.input + files, recursive=True))
    print("Found {} files to analyze".format(len(audiofiles)))

    # Make the prediction
    for audiofile in audiofiles:
        out_name = audiofile.split('/')[-1].split('.')[0]
        out_path = os.sep.join([cfg["PATH_JSON_DETECTIONS"],  out_name])

        WebrtcvadPredict(audiofile, 
                        out_path,
                        cfg["FRAME_LENGTH"],
                        cfg["AGGRESSIVENESS"]).main()
