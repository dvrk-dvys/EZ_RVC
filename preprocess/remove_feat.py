import librosa
from pydub import AudioSegment
from collections import Counter
import soundfile as sf
import numpy as np
from numpy import sqrt

from scipy.io import wavfile
import os
from collections import defaultdict



# Function to read RTTM file
def read_rttm(rttm_file):
    segments = []
    with open(rttm_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            start_time = float(parts[3]) * 1000  # Convert to milliseconds
            duration = float(parts[4]) * 1000  # Convert to milliseconds
            segments.append((start_time, start_time + duration))
    return segments

def is_silent(segment, threshold=0.01):
    # Calculate RMS value
    rms = sqrt(np.mean(segment**2))
    return rms < threshold

def apply_fade(segment, fade_length_samples):
    fade_in = np.linspace(0, 1, fade_length_samples)
    fade_out = np.linspace(1, 0, fade_length_samples)

    # Apply fade-in and fade-out to both channels
    segment[:fade_length_samples, 0] *= fade_in
    segment[-fade_length_samples:, 0] *= fade_out
    segment[:fade_length_samples, 1] *= fade_in
    segment[-fade_length_samples:, 1] *= fade_out
    return segment
def extract_main_singer_segments(wav_path, rttm_path):
    # Read the RTTM file and analyze the speaker labels
    annotations = []
    speaker_durations = []

    with open(rttm_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split()
            # if len(parts) >= 8:
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            annotations.append({'start': start, 'duration': duration, 'speaker': speaker_id})
            speaker_durations.append((speaker_id, duration))

        # Find the most frequent speaker label
        speaker_sums = defaultdict(float)
        for speaker_id, value in speaker_durations:
            speaker_sums[speaker_id] += value
        collapsed_list = list(speaker_sums.items())
        MAIN_SPEAKER = sorted(collapsed_list, key=lambda t: t[1], reverse=True)[0][0]
        speaker_labels = [t[0] for t in speaker_durations]
        most_common_label, _ = Counter(speaker_durations).most_common(1)[0][0]

    # Extract corresponding audio segments
    audio_data, sr = sf.read(wav_path)
    # audio = AudioSegment.from_wav(wav_path)

    output_path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset_raw/SZA_CTRL/'
    # output_file_name = "2_SZA-Doves_In_The_Wind.wav"
    # output_file_name = "3_SZA-Love Galore.wav"
    # output_file_name = "8_SZA-Wavy.wav"
    output_file_name = "9_SZA-Pretty Little Birds.wav"
    # output_file_name = "SZA_billboard_raw_interview.wav"

    output_path = os.path.join(output_path, output_file_name)

    # segments = {}
    segments = {key: [] for key in speaker_labels}

    for ann in annotations:
        # speaker_labels
        # if ann['speaker'] == MAIN_SPEAKER:
        start_sample = int(ann['start'] * sr)
        end_sample = int((ann['start'] + ann['duration']) * sr)
        segment = audio_data[start_sample:end_sample]
        segments[ann['speaker']].append(segment)
        # output_path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/preprocess/original_file_' + str(ann['start']) + '.wav'
        # sf.write(output_path, segment, sr)
    # filtered_audio_data = audio_data * mask

    # output_path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/preprocess/original_file.wav'
    # sf.write(output_path, segments[4], sr)
    print()
    # Convert the defaultdict to a list of tuples
    filtered_segments = [segment for segment in segments[MAIN_SPEAKER] if not is_silent(segment)]

    # Fade length in milliseconds; adjust as needed
    fade_length_ms = 15
    # Convert fade length to samples
    fade_length_samples = int(fade_length_ms * sr / 1000)

    # Apply fade-in and fade-out to each segment
    for segment in filtered_segments:
        apply_fade(segment, fade_length_samples)


    concatenated_segments = np.concatenate(filtered_segments, axis=0)
    # concatenated_segments = remove_silence(concatenated_segments, threshold=0.01, min_silence_length=4410)
    sf.write(output_path, concatenated_segments, sr)

    # output__path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/preprocess/filtered_audio.wav'
    # sf.write(output__path, filtered_audio_data, sr)

    # audio_segments = [AudioSegment(data=segment.tobytes(), sample_width=2, frame_rate=sr, channels=1) for segment in segments]
    # concatenated_audio = sum(audio_segments, AudioSegment.silent(duration=0))
    # concatenated_audio.export(output_path, format='wav')
    return most_common_label, concatenated_segments, sr


if __name__ == '__main__':


    # Load RTTM file
    # rttm_file = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_diairized/2_SZA_Doves_In_The_Wind_(feat._Kendrick_Lamar)_(Filtered_Acapella)_(Vocals)_audio.rttm'
    # rttm_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_diairized/3_SZA_Love_Galore_(feat._Travis_Scott)_(Filtered_Acapella)_(Vocals)_audio.rttm"
    # rttm_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_diairized/8_SZA_Wavy_(Interlude)_(feat._James_Fauntleroy)_(Filtered_Acapella)_(Vocals)_audio.rttm"
    rttm_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_diairized/9_SZA_Pretty_Little_Birds_(feat._Isaiah_Rashad)_(Filtered_Acapella)_(Vocals)_audio.rttm"
    # rttm_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_diairized/SZA_Billboard_Interview_Unedited_Raw_audio.rttm"

    # segments = read_rttm(rttm_file)

    # Load WAV file
    # wav_file = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset_raw/SZA_CTRL_ALL/2_SZA - Doves In The Wind (feat. Kendrick Lamar) (Filtered Acapella)_(Vocals).wav'
    # wav_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset_raw/SZA_CTRL_ALL/3_SZA - Love Galore (feat. Travis Scott) (Filtered Acapella)_(Vocals).wav"
    # wav_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset_raw/SZA_CTRL_ALL/8_SZA - Wavy (Interlude) (feat. James Fauntleroy) (Filtered Acapella)_(Vocals).wav"
    wav_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/dataset_raw/SZA_CTRL_ALL/9_SZA - Pretty Little Birds (feat. Isaiah Rashad) (Filtered Acapella)_(Vocals).wav"
    # wav_file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/output/preprocess/sza_resample/SZA Billboard Interview (Unedited Raw).wav"
    # audio_data, sr = sf.read(wav_file)
    # output_path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/preprocess/original_file.wav'
    # sf.write(output_path, audio_data, sr)



    # audio = AudioSegment.from_wav(wav_file)
    #
    most_common_label, concatenated_segments, sr = extract_main_singer_segments(wav_file, rttm_file)
    # main_segments.export(f"2_SZA - Doves In The Wind.wav", format="wav")
    # Iterate through segments and cut out the segments
    # for i, (start_time, end_time) in enumerate(segments):
    #     segment_audio = audio[start_time:end_time]
    #
    #     # Export the segment to a new WAV file
    #     segment_audio.export(f"segment_{i}.wav", format="wav")
