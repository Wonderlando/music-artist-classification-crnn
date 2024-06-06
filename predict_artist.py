import src.utility
import logging
import librosa
import os
import pickle

import tensorflow as tf
import numpy as np

from pydub import AudioSegment

if __name__ == '__main__':

    sr=16000
    n_mels=128
    n_fft=2048
    hop_length=512

    slice_length = 911

    song_path = 'test_songs/2.wav' # input song file path here
    label_encoder_path = 'label_encoder.pkl'
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
        
    ### Create mel spectrogram and convert it to the log scale
    ## load song
    print('Loading song')
    try:
        y, sr = librosa.load(song_path, sr=sr)
    except Exception as e:
        logging.error(f"Librosa error processing {song_path}: {e}")
        # Fallback: Use pydub to decode the MP3 file
        try:
            audio = AudioSegment.from_file(song_path)
            y = np.array(audio.get_array_of_samples())
            sr = audio.frame_rate
        except Exception as e:
            logging.error(f"Pydub error processing {song_path}: {e}")
    ## create mel spec

    print('Creating mel-spectrogram')
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        data = librosa.power_to_db(S, ref=1.0)
    except Exception as e:
        RuntimeError(f"Error processing {song_path}: {e}")

    ## slice song
    print('Slicing song')

    spectrograms = [] # stores spectrograms

    num_slices = int(data.shape[1] / slice_length) # number of slices required
    for j in range(num_slices - 1):
        spectrograms.append(data[:, slice_length * j:slice_length * (j + 1)])
    spectrograms = np.array(spectrograms) # convert to numpy array to add channel dimension

    spectrograms = spectrograms.reshape(spectrograms.shape + (1,)) # add channel dimension

    ### Call model
    print('Loading model')
    model = tf.keras.models.load_model('trained_models/20_911_42.keras')
    result = model.predict(spectrograms)

    ### Convert results to human-readable format
    result = np.argmax(result, axis=1, keepdims=True)
    result = label_encoder.inverse_transform(result.flatten()) # flatten turns results into 1d array
    
    print(result)