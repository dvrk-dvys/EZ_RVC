import os
import re
from collections import Counter

import librosa  # Optional. Use any library you like to read audio files.
import numpy as np
import soundfile  # Optional. Use any library you like to write audio files.
import soundfile as sf
import torch

# import faiss
from pyannote.audio import Pipeline
from scipy.io.wavfile import read

from app.utils.slicer import Slicer

# contentvec
# !wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
hubert_base = "./EZ_RVC/model_dir/checkpoint_best_legacy_500.pt"
# Alternatively, you can manually download and place it in the hubert directory


# https://github.com/pyannote/pyannote-audio
# this ensures that the current MacOS version is at least 12.3+
if torch.backends.mps.is_available():
    print(torch.backends.mps.is_built())
    device = torch.device("mps")
    print(torch.backends.mps.is_available())
else:
    print("MPS device not available. Fallback to CPU.")
    device = torch.device("cpu")


class Preprocess:

    def __init__(
        self,
        target_sample_rate,
        min_duration=1.0,
        top_db=30,
        max_length=10.0,
        frame_seconds=0.5,
        hop_seconds=0.1,
    ):
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.top_db = top_db
        self.max_length = max_length
        self.frame_seconds = frame_seconds
        self.hop_seconds = hop_seconds
        self.device = device


def slice(wav_path, file_name, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    audio, sr = librosa.load(
        wav_path, sr=None, mono=False
    )  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=5000,
        min_interval=300,
        hop_size=10,
        max_sil_kept=500,
    )
    chunks = slicer.slice(audio)
    flipped_chunks = []
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        # soundfile.write(f'/preprocess/output/sza_slices_22050/{file_name}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.
        soundfile.write(
            f"{output_path}{file_name}_{i}.wav", chunk, sr
        )  # Save sliced audio files with soundfile.

        flipped_chunks.append(chunk)
    return flipped_chunks


def resample(file_path, target_sample_rate, min_duration=1.0, top_db=30):
    """
    Load an audio file, resample it to the target sample rate, adjust its volume, and trim silence.

    Parameters
    ----------
    file_path : str
        The path to the audio file.

    Returns
    -------
    y : ndarray or None
        The resampled audio as a 1D numpy array, or None if the audio was too short.
    sr : int
        The sample rate of the resampled audio.
    """

    # Load the audio file and resample it
    y, sr = librosa.load(file_path, sr=target_sample_rate)

    # Check duration
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < min_duration:
        print(f"Skipping {file_path} because it is too short.")
        return None, sr

    # Adjust volume
    y = y / np.max(np.abs(y))

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Check duration after trimming
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < min_duration:
        print(f"Skipping {file_path} because it is too short after trimming.")
        return None, sr
    # Reshape if mono
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    return y, sr


def split(y, sr, max_length=10.0, top_db=30, frame_seconds=0.5, hop_seconds=0.1):
    """
    Split an audio signal into non-silent segments.

    Parameters
    ----------
    y : ndarray
        The audio signal as a 1D numpy array.
    sr : int
        The sample rate of the audio signal.

    Returns
    -------
    segments : list of ndarray
        A list of 1D numpy arrays, each representing a non-silent segment of the audio signal.
    """
    # Skip processing if the audio is None (indicating it was too short)
    if y is None:
        return []

    # Find non-silent intervals
    # top_db is the threshold for silence in decibels. frame_length and hop_length control
    # the length of the frames used to compute the root-mean-square (RMS) value,
    # which is used to determine whether a frame is silent or not.
    # The function returns an array of start and end indices for non-silent intervals.
    intervals = librosa.effects.split(
        y,
        top_db=top_db,
        frame_length=int(sr * frame_seconds),
        hop_length=int(sr * hop_seconds),
    )

    segments = []
    for start, end in intervals:
        for sub_start in range(start, end, int(sr * max_length)):
            sub_end = min(sub_start + int(sr * max_length), end)
            audio_cut = y[sub_start:sub_end]
            segments.append(audio_cut)

    return segments


def save_split(
    input_dir,
    output_dir,
    sr,
    max_length=10.0,
    top_db=30,
    frame_seconds=0.5,
    hop_seconds=0.1,
):
    """
    Splitting can also be useful for removing silences or non-speech segments from the audio,
     which could improve the performance of a voice conversion model.
     However, it's important to note that excessive segmentation might remove important context
      from the audio, which could negatively impact the performance of the model.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                split(
                    file_path,
                    output_dir,
                    sr,
                    max_length,
                    top_db,
                    frame_seconds,
                    hop_seconds,
                )


#

# Speaker diarization is the process of determining "who spoke when"
# in an audio or video recording that involves more than one speaker.
# In other words, it's the task of segmenting the input based on speaker identity.
# This can be particularly useful in tasks like transcription services,
# meeting summarization, and voice conversion in a multi-speaker environment.


def speaker_diarization(
    y_path, sr, min_speakers=1, max_speakers=1, huggingface_token=None
):
    """
    Perform speaker diarization using a pre-trained model from Hugging Face.

    Parameters
    ----------
    y : ndarray
        The audio data as a 1D numpy array.
    sr : int
        The sample rate of the audio data.
    min_speakers : int, optional
        The minimum number of speakers to identify in the audio. Default is 1.
    max_speakers : int, optional
        The maximum number of speakers to identify in the audio. Default is 1.
    huggingface_token : str, optional
        The Hugging Face token to use when downloading the pre-trained model. Default is None.

    Returns
    -------
    diarization : pyannote.core.Annotation
        The result of the speaker diarization. This is an object that contains the start and end times of each speaker segment, along with the identified speaker labels.
    """
    # Load the pre-trained model from Hugging Face
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=huggingface_token
    )

    # device = torch.device('metal') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = pipeline.to(device)  # switch to gpu

    # apply the pipeline to an audio file Perform speaker diarization
    diarization = pipeline(
        {"audio": y_path}, min_speakers=min_speakers, max_speakers=max_speakers
    )
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    return diarization


# 3. **Feature Extraction**: The next step is to extract features from the audio segments. The type of features extracted can vary depending on the specific model being used. Common types of features used in voice conversion model_dir include:
#
#    - **Spectrogram**: This is a 2D representation of the audio signal that shows how the frequencies present in the signal change over time. It can be calculated using the Fourier Transform.
#
#    - **Mel-frequency cepstral coefficients (MFCCs)**: These are a type of feature that represents the short-term power spectrum of a sound. They are based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.
#
#    - **F0 (Fundamental Frequency)**: This represents the pitch of the speech and is often used in voice conversion model_dir.
#
#    - **Vocal Tract Length Perturbation (VTLP)**: It's a technique to augment the audio data by simulating speakers with different vocal tract lengths.
#
#    - **Phoneme or text information**: Some model_dir also use phoneme or text information as additional features.
#
def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def lead_diairized_splits(rttm_path):
    # Path to the RTTM file
    # rttm_path = "/mnt/data/2_SZA_Doves_In_The_Wind_(feat._Kendrick_Lamar)_(Filtered_Acapella)_(Vocals)_audio.rttm"

    # Read the RTTM file and analyze the speaker labels
    speaker_labels = []

    with open(rttm_path, "r") as file:
        for line in file:
            # Typical RTTM format: SPEAKER filename 1 start_duration duration <NA> <NA> speaker_id <NA> <NA>
            parts = line.strip().split()
            if len(parts) >= 8:  # Check if the line has enough parts
                speaker_id = parts[7]
                speaker_labels.append(speaker_id)

    # Find the most frequent speaker label (assumed to be the main singer)
    most_common_label, _ = Counter(speaker_labels).most_common(1)[0]
    return most_common_label


def extract_main_singer_segments(wav_path, rttm_path):
    # Read the RTTM file and analyze the speaker labels
    annotations = []
    speaker_labels = []

    with open(rttm_path, "r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 8:
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    annotations.append(
                        {"start": start, "duration": duration, "speaker": speaker_id}
                    )
                    speaker_labels.append(speaker_id)
                except:
                    print()

    try:

        # Find the most frequent speaker label
        most_common_label, _ = Counter(speaker_labels).most_common(1)[0]
    except:
        print()
    # Extract corresponding audio segments
    audio_data, sr = sf.read(wav_path)
    segments = []
    for ann in annotations:
        if ann["speaker"] == most_common_label:
            start_sample = int(ann["start"] * sr)
            end_sample = int((ann["start"] + ann["duration"]) * sr)
            segment = audio_data[start_sample:end_sample]
            segments.append(segment)
    try:
        return np.concatenate(segments, axis=0)
    except:
        print()


if __name__ == "__main__":

    print("Current Working Directory: ", os.getcwd())

    # input_dir = './input/SZA_CTRL/'  # Replace with your actual input directory
    # input_dir = './dataset/44k/scripts/'
    input_dir = "./dataset_raw/eric_adams/"

    # input_dir = './dataset_raw/SZA_SOS/' # Replace with your actual input directory
    output_dir = "./preprocess/output/"
    # output_dir ='./dataset_raw/Scripts/'

    # speaker diairization test
    # file = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/input/SZA_CTRL/3_SZA - Love Galore (feat. Travis Scott) (Filtered Acapella)_(Vocals).wav'

    target_sample_rate = 44100  # 44100 is industry standard for cd's  # Replace with your desired sample rate
    # A common sample rate for music is 44.1 kHz.
    #  16 kHz is commonly used because it captures most of the important information in human speech while reducing the computational resources required compared to higher sampling rates.
    min_duration = 1.0  # Minimum duration in seconds
    top_db = 30

    # 0.
    # https://github.com/openvpi/audio-slicer
    # Slice!!!!
    slice_dir = "eric_adams_slices"

    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
    else:
        print(f"Searching in: {input_dir}")

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".wav"):

                    _file_name = file.split("/")[-1]
                    _file_name = _file_name.split(".wav")[0]
                    file_name = re.sub("[^\w\s]", "", _file_name)
                    file_name = re.sub(" ", "_", file_name)
                    print(file_name)

                    slices = slice(
                        root + file, file_name, f"preprocess/output/{slice_dir}/"
                    )
    # // Slice!!

    # # Resample!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    input_dir = (
        f"preprocess/output/{slice_dir}"  # Replace with your actual input directory
    )
    resampled = "eric_adams"
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                # Resample!!
                file_name = file.split("/")[-1]

                # Resample the audio file
                y, sr = resample(root + "/" + file, target_sample_rate)
                output_path = os.path.join(
                    output_dir + resampled + "_resample_44k/", file_name
                )

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Write the preprocessed audio to the output file
                # If data is 1D, make it 2D
                if y is None:
                    continue
                try:
                    sf.write(output_path, y, target_sample_rate)
                except:
                    print()
    # #// Resample!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # diairization local !!!!!!!!!!!
    # resample_dir = './EZ_RVC/output/preprocess/sza_resample_new/'
    #
    # # slices = slice(wav_path)
    # wav_file_paths = []
    # for filename in os.listdir(resample_dir):
    #     if re.match(r".*\.wav$", filename):
    #         file_path = os.path.join(resample_dir, filename)
    #         wav_file_paths.append(file_path)
    #
    # diairized_dir = './EZ_RVC/output/preprocess/sza_diairized_timestamps/'
    #
    # rttm_file_paths = []
    # for filename in os.listdir(diairized_dir):
    #     if re.match(r".*\.rttm$", filename):
    #         file_path = os.path.join(diairized_dir, filename)
    #         rttm_file_paths.append(file_path)
    #
    # # Pair up the paths to the .wav and .rttm files
    # data_pairs = list(zip(wav_file_paths, rttm_file_paths))
    #
    # # Extract main singer segments for each pair and create a new list of data
    # data_segments = [extract_main_singer_segments(wav, rttm) for wav, rttm in data_pairs]
    #
    # # Split the data into training and temporary sets
    # train_segments, temp_segments = train_test_split(data_segments, train_size=0.6, test_size=0.4)
    #
    # # Split the temporary data into validation and test sets
    # eval_segments, test_segments = train_test_split(temp_segments, train_size=0.5, test_size=0.5)
    # print()


# https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth
# https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth
