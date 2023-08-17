import os
import librosa
import soundfile as sf
import numpy as np
import zipfile


# :Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample:


# zip_path = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/input/SZA_CTRL_Archive.zip'  # Replace with your actual zip file path
# extract_dir = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/input/SZA_CTRL'  # Replace with your actual extract directory
input_dir = '/Users/jordanharris/Code/PycharmProjects/EZ_RVC/input/SZA_CTRL'  # Replace with your actual input directory
output_dir = '/preprocess/output/preprocess/resample'  # Replace with your actual output directory
target_sample_rate = 22050  # Replace with your desired sample rate
# A common sample rate for music is 44.1 kHz.
#  16 kHz is commonly used because it captures most of the important information in human speech while reducing the computational resources required compared to higher sampling rates.
min_duration = 1.0  # Minimum duration in seconds
top_db = 30  # Silence threshold in decibels

# # Extract the zip file
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_dir)

resample = []

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.wav'):  # Assuming files are in wav format
            file_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            # Load the audio file and resample it
            y, sr = librosa.load(file_path, sr=target_sample_rate)
            # The y output from librosa.load() is a one-dimensional numpy array, which represents the audio time series or audio signal.
            # Each element in this array corresponds to a sample of the audio signal. The value of each sample represents the amplitude of the audio signal at that particular point in time.
            # In the context of digital audio, amplitude refers to the volume or loudness of the sound. More specifically, it is a measure of the pressure of the sound waves. In digital terms, this is represented as a discrete value in the range of the data type used to store the audio data (for example, -1.0 to 1.0 for floating point audio data, which is typically what librosa.load() returns).
            # The index of each sample in the array corresponds to its position in time, relative to the sample rate. For example, if the sample rate is 22050 Hz (the default for librosa.load()), this means there are 22050 samples per second. So, the sample at index 22050 in the array corresponds to the amplitude of the audio signal 1 second into the audio file.
            # If the audio file is stereo (i.e., has separate channels for left and right), librosa.load() will automatically convert it to mono by averaging the two channels. This means the resulting time series represents the average amplitude of the left and right channels at each point in time.


            # Check duration
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < min_duration:
                print(f'Skipping {file_path} because it is too short.')
                continue

            # Adjust volume
            y = y / np.max(np.abs(y))

            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=top_db)

            # Check duration after trimming
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < min_duration:
                print(f'Skipping {file_path} because it is too short after trimming.')
                continue

            # # Ensure the output directory exists
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            #
            # # Write the preprocessed audio to the output file
            # sf.write(output_path, y, target_sample_rate)
            resample.append(y)


#// :Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample::Resample:

print(resample)

