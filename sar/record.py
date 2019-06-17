import pyaudio
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence
import ffmpeg
from spectrograms import generate_spectrogram
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
RECORD_SECONDS = 10
RECORDINGS_DIRECTORY = './sar/recordings'
WAVE_OUTPUT_FILENAME = RECORDINGS_DIRECTORY + '/original.wav'

# Record audio stream for a few seconds (set above) and save to disk 
def record():
    # Start audio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save wav file to disk
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Split a wav file into chunks, each chunk containing one word
def split_words():
    wav = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    chunks = split_on_silence(wav, min_silence_len=500, silence_thresh=-50)
    num_of_chunks = len(chunks)
    print('Detected {} words'.format(num_of_chunks))
    # Ensure that the number of words is between 4 and 10
    if num_of_chunks < 4 or num_of_chunks > 10:
        print('Please record between 4 and 10 words only')
        exit()
    for i, chunk in enumerate(chunks):
        out_wav = RECORDINGS_DIRECTORY + '/digit_{}.wav'.format(i)
        # Save wav file containing only this word
        chunk.export(out_wav, format='wav')
        out_spec = out_wav + '.jpg'
        # Generate spectrogram for this word
        generate_spectrogram(out_wav, out_spec)
        print('Processing word {} ({})'.format(i+1, out_spec))
        # Test the word using Tensorflow
        tensorflow_test_word(out_spec)

# Test the given word (spectrogram) against Tensorflow (must be trained already!)
def tensorflow_test_word(spectrogram):
    test = 'python -m scripts.label_image --graph=tf_files/retrained_graph.pb --image="{}"'.format(spectrogram)
    os.system(test)
        
if __name__ == '__main__':
    print('Recording... ({} seconds)'.format(RECORD_SECONDS))
    record()
    print('Recording saved')
    split_words()
    print('Recognition process finished')