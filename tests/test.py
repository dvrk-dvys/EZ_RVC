# # https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI


# There is currently a warning when running this Colab, similar to how other AI tools have been throttled by Google
# in the past for being too popular for non-research / non-approved use. I doubt that it is risky to currently run,
# but I am looking into why this happens. Stay tuned.
###EDIT: Warning is gone now.
#
# LINKS
# https://www.youtube.com/watch?v=hB7zFyP99CY
# https://www.youtube.com/watch?v=OsoEv9SawMo
# Easy GUI (collab page)
# https://colab.research.google.com/drive/1Gj6UTf2gicndUW_tVheVhTXIIYpFTYc7?usp=sharing&authuser=0&pli=1#scrollTo=F6zbsuNs6xZF
#
# RVC Model Archive
# https://docs.google.com/spreadsheets/d/1tAUaQrEHYgRsm1Lvrnj14HFHDwJWl0Bd9x0QePewNco/edit#gid=1977693859
# STUDIO I-ID
# https://studio.d-id.com/




# My steps,
# Data retrieval
# https://www.youtube.com/watch?v=Jq_1XiJs7ZE
# https://youtu.be/Wc4EnXwT8II
# https://youtu.be/8wgQsqjwYGs




# https://4kdownload.to/en1/youtube-wav-downloader
# ultimate vocal remover


# Removing reverb / echo
# It is necessary to remove reverb / echo from the dataset for the best results.
# Ideally you have as little there as possible in the first place,
# and isolating reverb can obviously reduce the quality of the vocal.
# But if you need to do this, under MDX-Net you can find Reverb HQ,
# which will export the reverbless audio as the ‘No Other’ option. Oftentimes, this isn’t enough.
# If that did nothing, (or just didn't do enough),
# you can try to process the vocal output through the VR Architecture model_dir in UVR to remove echo
# and reverb that remains using De-Echo-DeReverb. If that still wasn't enough, somehow,
# you can use the De-Echo normal model on the output, which is the most
# aggressive echo removal model of them all.
# https://github.com/Anjok07/ultimatevocalremovergui


# train a model on the singer you want, (with music or just vocals?) then provide the model a song that is just vocals for it to infer on

# create a gradio account for running the voice model itssels
# youtube to wav file




import librosa
librosa.show_versions()

# stable diffusion to redraw a face? if its animated?






# Steps:

#   pre-sd         Speech diarization using pyannote.audio
#   pre-split      Split audio files into multiple files


# 1. Process The Dataset
# https://github.com/voicepaw/so-vits-svc-fork/tree/main/src/so_vits_svc_fork/preprocessing
#   pre-resample   Preprocessing part 1: resample
# preprocess_classify.py: It's possible that this script is used to manually classify files. This could involve moving them into different folders based on some classification scheme or labeling them in some way. The specific classification could be related to the speaker, the type of audio (e.g., speech, music, noise), or some other characteristic of the audio.
#
# preprocess_flist_config.py: This script might be used for handling file lists (flist). In the context of audio preprocessing, a file list often refers to a list of paths to audio files that need to be processed. This script might be responsible for generating such lists, perhaps based on certain criteria or configurations.
#
# preprocess_hubert_f0.py: This script likely uses the HuBERT model, which is a self-supervised model developed by Facebook AI for speech and audio understanding tasks. It could be extracting features from the audio files, such as the fundamental frequency (F0), which is often used in voice conversion tasks.
#
# preprocess_resample.py: This script probably resamples the audio files to ensure they all have the same sample rate. This is a common preprocessing step in audio processing tasks.
#The line `y, sr = librosa.load(file_path, sr=target_sample_rate)` is using the `librosa.load()` function to load an audio file and resample it to a desired sample rate.

# Let's break down what this means:
#
# **Loading the Audio File**: Audio files are a form of digital data. When we "load" an audio file, we're converting that digital data into a form that we can manipulate with our code.
#
# **Resampling to a Desired Sample Rate**: The "sample rate" of an audio file refers to the number of samples of audio carried per second, measured in Hz or kHz. A common sample rate for music is 44.1 kHz. Resampling is the process of changing the sample rate of a discrete signal to obtain a new discrete representation of the underlying continuous signal. The `librosa.load()` function automatically resamples the audio to the given sample rate (`sr`).
#
# **Output as a Numpy Array**: The `librosa.load()` function returns two outputs. The first output, `y`, is a numpy array that contains the audio time series. The length of this array is the number of samples in the audio file. Each number in the array represents the amplitude of the audio signal at a particular sample. If the audio is stereo, `librosa.load()` will automatically convert it to mono, so each sample has a single amplitude.
#
# The second output, `sr`, is a number that represents the sample rate of the audio time series. If you provided a target sample rate to `librosa.load()`, this will be equal to your target sample rate. Otherwise, it will be the default sample rate used by `librosa`, which is 22050 Hz.
#
# **Computationally**, loading and resampling an audio file can be a somewhat intensive operation, especially for long audio files. It involves reading the digital data, converting it to a time series, and potentially resampling it, which can involve interpolation or decimation of the audio signal. However, these operations are typically quite fast on modern hardware, and can be performed in real time for many applications.



# preprocess_speaker_diarization.py: This script might be used for speaker diarization, which is the process of separating an audio stream into segments based on the speaker. This could be useful in voice conversion tasks if you want to train separate model_dir for each speaker or if you want to ensure that the training data for each speaker is balanced.
#
# preprocess_split.py: As you've already posted the code for this, it's clear that this script is used to split audio files into shorter segments. This involves loading an audio file, splitting it into segments where the sound level is below a certain threshold, and saving each segment as a separate audio file.
#
# preprocess_utils.py: This script probably contains utility functions that are used in the other scripts. These could include functions for loading and saving audio files, computing audio features, handling file paths, and so on.
#
# All these steps are important and serve different purposes. However, the most critical ones generally are resampling, feature extraction (like F0 extraction with HuBERT), and splitting the audio into manageable segments. These steps are crucial to ensure that the data fed into the model is in a uniform and useful format.
#
# For more specific details about these scripts, you would need to check the actual code or the documentation provided with the code.







# 2. Pitch Extraction

# 3. Train Model

# 4. Train Index



