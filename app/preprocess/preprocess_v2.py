import argparse
import os
import re

import librosa
import soundfile

from app.utils.pre_utils import convert_to_wav, gen_spk_config
from app.utils.slicer import Slicer
from app.utils.utils import get_device

# import faiss


device = get_device()


class Preprocessor:
    def __init__(
        self,
        speaker_name,
        speech_encoder,
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
        self.slicer = Slicer(
            sr=target_sample_rate,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500,
        )
        self.spk_config = gen_spk_config(
            speaker_name=speaker_name, speech_encoder=speech_encoder
        )

    def process_raw_audio_dir(self, input_dir):
        # PASS 1: Convert all non-wav files to wav (in-place)
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".wav"):
                    original_path = os.path.join(root, file)
                    wav_path = original_path.rsplit(".", 1)[0] + ".wav"

                    convert_to_wav(original_path, wav_path)
                    os.remove(original_path)  # Delete original m4a/mp4

        # PASS 2: Process only .wav files
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".wav"):
                    # Your existing slice/process logic here
                    file_path = os.path.join(root, file)
                    file_name = file.replace(".wav", "")
                    file_name = re.sub("[^\w\s]", "", file_name)
                    file_name = re.sub(" ", "_", file_name)

                    # Call slice method
                    sliced_chunks = self.slice(
                        file_path, file_name, f"data/processed/sliced/"
                    )

    def slice(self, wav_path, file_name, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        audio, sr = librosa.load(
            wav_path, sr=None, mono=False
        )  # Load an audio file with librosa.
        chunks = self.slicer.slice(audio)
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


def main():
    pass


if __name__ == "__main__":
    # we are keeping files lists bec of parallelization
    #  2. Parallel processing works better with file lists
    # always pick the Audio M4A option from that downloader then convert it to a wav
    # https://www.dictationer.com/video-downloader/youtube-audio-downloader

    # todo!! Put this into docker compose
    # !wget -P pretrain/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -O checkpoint_best_legacy_500.pt
    # hubert_base = './data/weights/encoders/checkpoint_best_legacy_500.pt'

    raw_audio_dir = "/Users/jordanharris/Code/EZ_RVC/data/raw/nat_king_cole"
    speaker = "nat_king_cole"

    parser = argparse.ArgumentParser(description="EZ_RVC preprocessing")

    parser.add_argument(
        "--raw_audios",
        type=str,
        required=True,
        default=raw_audio_dir,
        help="Path to raw audio files (directory or single file).",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        required=True,
        default=speaker,
        help="Speaker name identifier (e.g., 'nat_king_cole').",
    )
    parser.add_argument(
        "--speech_encoder",
        type=str,
        default="hubertsoft",
        choices=[
            "hubertsoft",
            "vec768l12",
            "vec256l9",
            "whisper-ppg",
            "whisper-ppg-large",
            "cnhubertlarge",
            "dphubert",
            "wavlmbase+",
        ],
        help="Speech encoder type.",
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=44100,
        help="Target sample rate for final WAV chunks (e.g., 44100).",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum duration (seconds) to keep after trimming.",
    )
    parser.add_argument(
        "--top_db",
        type=float,
        default=30.0,
        help="Silence threshold in dB for trimming.",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=10.0,
        help="Maximum segment length (seconds) when splitting long non-silent regions.",
    )
    parser.add_argument(
        "--frame_seconds",
        type=float,
        default=0.5,
        help="Frame length (seconds) used for silence detection.",
    )
    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=0.1,
        help="Hop length (seconds) used for silence detection.",
    )

    args = parser.parse_args()

    Pre = Preprocessor(
        speaker_name=args.speaker_name,
        speech_encoder=args.speech_encoder,
        target_sample_rate=args.target_sample_rate,
        min_duration=args.min_duration,
        top_db=args.top_db,
        max_length=args.max_length,
        frame_seconds=args.frame_seconds,
        hop_seconds=args.hop_seconds,
    )
    Pre.process_raw_audio_dir(
        args.raw_audios
    )  # Process raw audio directory (convert to WAV + slice)
