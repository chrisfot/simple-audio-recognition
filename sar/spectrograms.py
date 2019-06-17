import os
import matplotlib.pyplot as plot
from scipy.io import wavfile


# Directories which contain the recordings (.wav files) and spectrograms
DIR_RECORDINGS = './sar/free-spoken-digit-dataset/recordings'
DIR_SPECTROGRAMS = './sar/spectrograms'

# Generates the spectrogram for the given wav file and saves it to the destination path given
def generate_spectrogram(wav, spectrogram):
    frame_rate, signal_data = wavfile.read(wav)
    plot.figure(figsize=(5,5))
    plot.subplot(111)
    plot.axis('off')
    plot.specgram(signal_data, Fs=frame_rate)
    plot.savefig(spectrogram, bbox_inches='tight')
    plot.close()

def generate_spectrograms_for_recordings():
    print('Generating spectrograms for recordings located in: ' + DIR_RECORDINGS)
    # Loop through all recordings and generate the spectrogram of each one
    for filename in os.listdir(DIR_RECORDINGS):
        if filename.endswith('.wav'):
            # The digit this recording is referring to
            digit = filename[:1]
            # The filenames of the recording and the spectrogram image
            file_wav = DIR_RECORDINGS + '/' + filename
            file_spectrogram = DIR_SPECTROGRAMS + '/' + digit + '/' + filename + '.jpg'
            # Generate spectrogram!
            generate_spectrogram(file_wav, file_spectrogram)
            print('Saved: ' + file_spectrogram)
    print('Finished')

if __name__ == '__main__':
    generate_spectrograms_for_recordings()