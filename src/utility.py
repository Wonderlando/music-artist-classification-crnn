import os
import dill
import random
import itertools
import logging
import pickle

import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

import librosa
import librosa.display
from pydub import AudioSegment

import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy import stats


def visualize_spectrogram(path, duration=None,
                          offset=0, sr=16000, n_mels=128, n_fft=2048,
                          hop_length=512):
    """This function creates a visualization of a spectrogram
    given the path to an audio file."""

    # Make a mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop_length)

    # Convert to log scale (dB)
    log_S = librosa.power_to_db(S, ref=1.0)

    # Render output spectrogram in the console
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


def create_dataset(artist_folder='./artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    logging.basicConfig(filename = 'error.log', level = logging.ERROR)

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(os.path.join(artist_folder, path))]

    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in artists:
        print(artist)
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = [album for album in os.listdir(artist_path)
                         if os.path.isdir(os.path.join(artist_path, album))]

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = [song for song in os.listdir(album_path)
                           if os.path.isfile(os.path.join(album_path, song))]

            for song in album_songs:
                song_path = os.path.join(album_path, song)

                try:
                    # Create mel spectrogram and convert it to the log scale
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
                        continue

                try:
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                       n_fft=n_fft,
                                                       hop_length=hop_length)
                    log_S = librosa.power_to_db(S, ref=1.0)
                    data = (artist, log_S, song)

                    # Save each song
                    save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                    with open(os.path.join(save_folder, save_name), 'wb') as fp:
                        dill.dump(data, fp)
                except Exception as e:
                    print(f"Error processing {song_path}: {e}")


def load_dataset(song_folder_name='song_data',
                 artist_folder='artists',
                 nb_classes=20, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list_path = os.path.join(os.getcwd(), song_folder_name)
    song_list = os.listdir(song_list_path)

    # Load the list of artists
    artist_list_path = os.path.join(os.getcwd(), artist_folder)
    print(artist_list_path)
    artist_list = os.listdir(artist_list_path)

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])

    return artist, spectrogram, song_name

### Won't need this for our purposes
# def load_dataset_album_split(song_folder_name='song_data', 
#                              artist_folder='./artists',
#                              nb_classes=20, random_state=42):
#     """ This function loads a dataset and splits it on an album level"""
#     song_list = os.listdir(song_folder_name)

#     # Load the list of artists
#     artist_list = os.listdir(artist_folder)

#     train_albums = []
#     test_albums = []
#     val_albums = []
#     random.seed(random_state)
#     for artist in os.listdir(artist_folder):
#         albums = os.listdir(os.path.join(artist_folder, artist))
#         random.shuffle(albums)
#         test_albums.append(artist + '_%%-%%_' + albums.pop(0))
#         val_albums.append(artist + '_%%-%%_' + albums.pop(0))
#         train_albums.extend([artist + '_%%-%%_' + album for album in albums])

#     # select the appropriate number of classes
#     prng = RandomState(random_state)
#     artists = prng.choice(artist_list, size=nb_classes, replace=False)

#     # Create empty lists
#     Y_train, Y_test, Y_val = [], [], []
#     X_train, X_test, X_val = [], [], []
#     S_train, S_test, S_val = [], [], []

#     # Load each song into memory if the artist is included and return
#     for song in song_list:
#         with open(os.path.join(song_folder_name, song), 'rb') as fp:
#             loaded_song = dill.load(fp)
#         artist, album, song_name = song.split('_%%-%%_')
#         artist_album = artist + '_%%-%%_' + album

#         if loaded_song[0] in artists:
#             if artist_album in train_albums:
#                 Y_train.append(loaded_song[0])
#                 X_train.append(loaded_song[1])
#                 S_train.append(loaded_song[2])
#             elif artist_album in test_albums:
#                 Y_test.append(loaded_song[0])
#                 X_test.append(loaded_song[1])
#                 S_test.append(loaded_song[2])
#             elif artist_album in val_albums:
#                 Y_val.append(loaded_song[0])
#                 X_val.append(loaded_song[1])
#                 S_val.append(loaded_song[2])

#     return Y_train, X_train, S_train, \
#            Y_test, X_test, S_test, \
#            Y_val, X_val, S_val


def load_dataset_song_split(song_folder_name='song_data',
                            artist_folder='artists',
                            nb_classes=20,
                            test_split_size=0.1,
                            random_state=42):
    Y, X, song_name = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(
        X, Y, song_name, test_size=test_split_size, stratify=Y,
        random_state=random_state)

    # Create a validation to be used to track progress
    # X_train, X_val, Y_train, Y_val, S_train, S_val = train_test_split(
    #     X_train, Y_train, S_train, test_size=validation_split_size,
    #     shuffle=True, stratify=Y_train, random_state=random_state)

    return Y_train, X_train, S_train, \
           Y_test, X_test, S_test


def slice_songs(X, Y, S, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(Y[i])
            song_name.append(S[i])

    return np.array(spectrogram), np.array(artist), np.array(song_name)


def create_spectrogram_plots(artist_folder='./artists', sr=16000, n_mels=128,
                             n_fft=2048, hop_length=512):
    """Create a spectrogram from a randomly selected song
     for each artist and plot"""

    # get list of all artists
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(os.path.join(artist_folder, path))]

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 12), sharex=True,
                           sharey=True)

    row = 0
    col = 0

    # iterate through artists, randomly select an album,
    # randomly select a song, and plot a spectrogram on a grid
    for artist in artists:
        # Randomly select album and song
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = [path for path in os.listdir(artist_path) if
                        os.path.isdir(os.path.join(artist_path, path))]
        album = random.choice(artist_albums)
        album_path = os.path.join(artist_path, album)
        album_songs = os.listdir(album_path)
        song = random.choice(album_songs)
        print(song)
        song_path = os.path.join(album_path, song)

        # Create mel spectrogram
        y, sr = librosa.load(song_path, sr=sr, offset=60, duration=3)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.power_to_db(S, ref=1.0)

        # Plot on grid
        plt.axes(ax[row, col])
        librosa.display.specshow(log_S, sr=sr)
        plt.title(artist + ' , ' + song)
        col += 1
        if col == 5:
            row += 1
            col = 0

    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_history(history, title="model accuracy"):
    """
    This function plots the training and validation accuracy
     per epoch of training
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    return


def predict_artist(model, X, Y, S,
                   le, class_names,
                   slices=None, verbose=False,
                   ml_mode=False):
    """
    This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    print("Test results when pooling slices by song and voting:")
    # Obtain the list of songs
    songs = np.unique(S)

    prediction_list = []
    actual_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        X_song = X[S == song]
        Y_song = Y[S == song]

        # If not using full song, shuffle and take up to a number of slices
        if slices and slices <= X_song.shape[0]:
            X_song, Y_song = shuffle(X_song, Y_song)
            X_song = X_song[:slices]

        # Get probabilities of each class
        predictions = model.predict(X_song, verbose=0)

        if not ml_mode:
            # Get list of highest probability classes and their probability
            class_prediction = np.argmax(predictions, axis=1)
            class_probability = np.max(predictions, axis=1)

            # keep only predictions confident about;
            prediction_summary_trim = class_prediction[class_probability > 0.5]

            # deal with edge case where there is no confident class
            if len(prediction_summary_trim) == 0:
                prediction_summary_trim = class_prediction
        else:
            prediction_summary_trim = predictions

        # get most frequent class
        prediction = stats.mode(prediction_summary_trim)[0][0]
        actual = stats.mode(np.argmax(Y_song))[0][0]

        # Keeping track of overall song classification accuracy
        prediction_list.append(prediction)
        actual_list.append(actual)

        # Print out prediction
        if verbose:
            print(song)
            print("Predicted:", le.inverse_transform([prediction])[0], "\nActual:",
                  le.inverse_transform([actual])[0])
            print('\n')

    # Print overall song accuracy
    actual_array = np.array(actual_list)
    prediction_array = np.array(prediction_list)
    cm = confusion_matrix(actual_array, prediction_array)
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Confusion matrix for pooled results' +
                                ' with normalization')
    class_report = classification_report(actual_array, prediction_array,
                                         target_names=class_names)
    print(class_report)

    class_report_dict = classification_report(actual_array, prediction_array,
                                              target_names=class_names,
                                              output_dict=True)
    return (class_report, class_report_dict)


def encode_labels(Y,label_encoder=None):
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    # N = Y.shape[0]

    # Encode the labels
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        Y_integer_encoded = label_encoder.fit_transform(Y)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f) # save encoder for decoding in the future
    else:
        Y_integer_encoded = label_encoder.transform(Y)
        print(Y_integer_encoded)

    # convert into one hot encoding
    Y_enc = keras.utils.to_categorical(Y_integer_encoded)
    print(label_encoder.classes_)

    # return encoders to re-use on other data
    return Y_enc, label_encoder


if __name__ == '__main__':

    # configuration options
    # create_data = True
    # create_visuals = True
    # save_visuals = False

    # if create_data:
    #     create_dataset(artist_folder='artists', save_folder='song_data',
    #                    sr=16000, n_mels=128, n_fft=2048,
    #                    hop_length=512)

    # if create_visuals:
    #     # Create spectrogram for a specific song
    #     visualize_spectrogram(
    #         'artists/metro_boomin/heroes_&_villians/' +
    #         '0.wav',
    #         offset=60, duration=29.12)

    #     # Create spectrogram subplots
    #     create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
    #                              n_fft=2048, hop_length=512)
    #     if save_visuals:
    #         plt.savefig(os.path.join('spectrograms.png'),
    #                     bbox_inches="tight")


    artist_folder='artists'
    song_folder='song_data'
    nb_classes=20
    slice_length=911

    Y_train, X_train, S_train, Y_test, X_test, S_test = \
        load_dataset_song_split(song_folder_name=song_folder,
                                        artist_folder=artist_folder,
                                        nb_classes=nb_classes,
                                        random_state=42)

    print("Loaded and split dataset. Slicing songs...")

    # Create slices out of the songs
    X_train, Y_train, S_train = slice_songs(X_train, Y_train, S_train,
                                                    length=slice_length)
    X_test, Y_test, S_test = slice_songs(X_test, Y_test, S_test,
                                                 length=slice_length)

    # print("Training set label counts:", np.unique(Y_train, return_counts=True))

    # Encode the target vectors into one-hot encoded vectors
    Y_train = encode_labels(Y_train)
    Y_test = encode_labels(Y_test)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))

    # print('Testing:')
    # print(type(X_test))
    # print(X_test.shape)
    # print(Y_test)
    # print(type(Y_test))
    # print(Y_test.shape)

    # print('Training:')
    # print(type(X_train))
    # print(X_train.shape)
    # print(type(Y_train))
    # print(Y_train.shape)