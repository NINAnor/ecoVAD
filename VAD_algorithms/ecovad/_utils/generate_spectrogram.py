"""
Function to generate a spectrogram

Each column in the spectrogram is the FFT of a slice in time where the centre at this time point has a window placed with n_fft=X components.
n_fft = number of samples per fft
hop length tells us how many audio samples we need to skip over before we calculate the next FFT by default n_fft / 4
Hop length thus controls for the number of columns
    
Our sample rate is 16000Hz, if we take n_fft=1024 we do a fft every 0.064sec
"""

import librosa
import librosa.display

def generate_spectrogram(x, sr, show=False):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    if show == True:
        librosa.display.specshow(Xdb, sr=sr, cmap='viridis', x_axis='time', y_axis='hz')
    return(Xdb)


def generate_mel_spectrogram(x, sr, show=False):
    
    sgram = librosa.stft(x, n_fft=1024, hop_length=376)
    
    # Separate a complex-valued spectrogram D into its magnitude (S) and phase (P)
    sgram_mag, _ = librosa.magphase(sgram)
    
    # Compute the mel spectrogram -> convert frequency in mel-scale
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
    
    # use the decibel scale to get the final Mel Spectrogram
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram)
    
    # Display the mel spectrogram
    if show == True:
        librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        
    return(mel_sgram)

def show_waveplot(y, sr):
    """Plot a waveplot: show the shape of the wave shows the pattern of the vibration"""
    
    y_df = pd.DataFrame(y)
    y_df.columns = ["amplitude"]
    
    y_df["sampling"]=range(0,len(y))
    y_df["time"]=y_df["sampling"] / sr
    
    y_df.plot(x='time', y='amplitude')