import librosa
import matplotlib.pyplot as plt

file = "/Users/jordanharris/Code/PycharmProjects/EZ_RVC/input/SZA_CTRL/1_SZA - Supermodel (Filtered Acapella)_(Vocals).wav"
# Load the audio file without resampling
y_original, sr_original = librosa.load(file, sr=None)

# Load the audio file with resampling
target_sample_rate = 44100  # Replace with your desired sample rate
y_resampled, sr_resampled = librosa.load(file, sr=target_sample_rate)

# Plot the original audio
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y_original, sr=sr_original)
plt.title("Original Audio")

# Plot the resampled audio
plt.subplot(2, 1, 2)
librosa.display.waveshow(y_resampled, sr=sr_resampled)
plt.title("Resampled Audio")

plt.tight_layout()
plt.show()
