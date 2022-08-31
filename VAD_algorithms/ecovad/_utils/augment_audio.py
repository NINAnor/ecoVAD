# ! DEPRECATED - BUT GOOD TO KEEP TO HAVE LOW LEVEL FUNCTIONS

import random
import numpy as np
import os

from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    """
    Set the sound at a certain loudness, be careful when
    setting the target value as the peak may go higher than
    0 dBFS
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def change_snr(start_range, end_range, speech, background):
    """
    Changes the background to noise ratio according to the formula:
    mix = alpha * speech + (1 - alpha) * background
    """
    # Compute alpha for low SNR
    alpha = random.uniform(start_range, end_range)
            
    # Compute the gain
    gain_in_db_for_speech = 20*np.log10(alpha)
    gain_in_db_for_bg = 20*np.log10(1-alpha)
            
    # apply the gains
    speech_altered = speech.apply_gain(gain_in_db_for_speech)
    background_altered = background.apply_gain(gain_in_db_for_bg)
    
    # return the altered speech and soundscape
    return (speech_altered, background_altered, alpha)

def mix_audio(speech, soundscape, aug=False):
    """Mix a human speech with the soundscape with augmentation"""
    
    if aug==True:
        
        # Normalize both records
        speech = match_target_amplitude(speech, 0) 
        soundscape = match_target_amplitude(soundscape, 0) # -53.65 is the RMS of the audiomoth
        
        # Add a fade in / fade out effect
        speech = speech.fade_in(500).fade_out(500)
        speech_aug, soundscape_aug, alpha = change_snr(start_range=0.1, end_range=0.9, speech=speech, background=soundscape)
                
    # Combine the records with a random time offset
    combined = speech_aug.overlay(soundscape_aug)
     
    # Return the segment
    return (combined, alpha)

def get_random_segment(dataset, length_segments):
    """
    Pick a random audio segment from a dataset and 
    split it into length_segments seconds fractions
    and pick at random within this range
    """

    segment_path = pick_a_file_at_random(dataset)
    
    # First we load a random file in the dataset
    segment = AudioSegment.from_file(segment_path)

    # Make sure the sample is AT LEAST the length of the desired segment
    while len(segment) < length_segments + 1000: # to avoid empty ranges / select segments of at least 4 seconds
        segment_path = pick_a_file_at_random(dataset)
        segment = AudioSegment.from_file(pick_a_file_at_random(dataset))

    # Then we decompose the segment record into S second segments (depending)
    # of the length_segments argument
    segment_decomp = np.arange(0, len(segment), length_segments)

    # Randomly pick a location within the sequence split and its adjacent number
    s = random.randrange(len(segment_decomp) - 1)
    e = s + 1

    # Finally isolate the random sequence
    random_segment_range = segment[segment_decomp[s]: segment_decomp[e]] 
    
    return (random_segment_range, segment_path)

def pick_a_file_at_random(to_open_path):
    """
    Pick a random file in a random directory
    """

    all_f_paths = []
    for currentpath, folders, files in os.walk(to_open_path):
        for file in files:
            all_f_paths.append(os.path.join(currentpath, file))

    #all_f_paths = [f for f in all_f_paths if '.wav' in f.lower() or '.flac' in f.lower()]
    #print(all_f_paths)
    chosen_path = np.random.choice(all_f_paths)

    try:
        segment = AudioSegment.from_file(chosen_path)
    except Exception as e:
        return pick_a_file_at_random(to_open_path)

    # Return the path of the file
    return chosen_path

