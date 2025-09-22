import argparse
import os
import re
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import soundfile
import torch
from pyannote.audio import Pipeline

from app.utils.pre_utils import convert_to_wav, gen_spkr_config
from app.utils.slicer import Slicer
from app.utils.utils import get_device

# import faiss


device = get_device()


class Preprocessor:
    '''
    From your training code, the model expects these inputs:
    c, f0, spec, y, spkr, lengths, uv, volume = items

    Where:
    - c = Content features (from HuBERT/ContentVec encoder)
    - f0 = Fundamental frequency (pitch)
    - spec = Spectrogram
    - y = Raw audio waveform
    - spkr = Speaker ID
    - lengths = Sequence lengths
    - uv = Unvoiced/voiced flags
    - volume = Volume features

    '''

    def __init__(
        self,
        output_path,
        enable_diarize,
        speaker_name,
        speech_encoder,
        target_sample_rate,
        min_duration=1.0,
        top_db=30,
        max_length=10.0,
        frame_seconds=0.5,
        hop_seconds=0.1,
    ):
        self.output_path = output_path
        self.enable_diarize = enable_diarize
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.top_db = top_db
        self.max_length = max_length
        self.frame_seconds = frame_seconds
        self.hop_seconds = hop_seconds
        self.device = device
        self.spkr = speaker_name
        self.spkr_config = gen_spkr_config(speaker_name=speaker_name, speech_encoder=speech_encoder)

        self.slicer = Slicer(
            sr=target_sample_rate, threshold=-40, min_length=5000, min_interval=300, hop_size=10, max_sil_kept=500
        )
        self.pyannote_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    def load_audio(self, audio_path):
        """Load audio file"""
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return audio, sr

    def slice(self, audio):  # , file_name, output_path):
        chunks = self.slicer.slice(audio)
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T  # Swap axes if the audio is stereo.
            yield chunk

        #!DO WE WRITE THE CHUNKS FOR THE OTHER PREPROCESSING STEPS OR DO WE YIELD THEM?
        #            audio, sr = librosa.load(
        #        wav_path, sr=None, mono=False
        #    )  #
        #    # soundfile.write(f'/preprocess/output/sza_slices_22050/{file_name}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.
        #    soundfile.write(
        #        f"{output_path}{file_name}_{i}.wav", chunk, sr
        #    )  # Save sliced audio files with soundfile.

        #    flipped_chunks.append(chunk)
        # return flipped_chunks

    def resample_audio(self, input_path):  #:, output_path):
        """Resample audio to target sample rate"""
        audio, sr = self.load_audio(input_path)
        if sr != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
        # !DO WE WRITE THE RESAMPLED AUDIO FOR THE OTHER PREPROCESSING STEPS OR DO WE YIELD THEM?
        # soundfile.write(output_path, audio, self.target_sample_rate)
        return audio.astype(np.float32), self.target_sample_rate

    def diarize_speaker(self, waveform):  # , audio_path, output_path, speaker_id=0):
        """
        - Extract target speaker from multi-speaker audio
        - Helper for pyannote with torch tensor fallback to temp file
        """
        # target_label = f"SPEAKER_{self.self.spkr:02d}"
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)
        waveform = waveform.astype(np.float32)

        try:
            wav_sample = {"waveform": torch.tensor(waveform).unsqueeze(0), "sample_rate": self.target_sample_rate}
            diarization = self.pyannote_pipeline(wav_sample)
        except Exception:
            with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                soundfile.write(tmp.name, waveform, self.target_sample_rate)
                diarization = self.pyannote_pipeline(tmp.name)

        # choose dominant speaker by total speaking time
        totals = {}
        for turn, _, spkr in diarization.itertracks(yield_label=True):
            totals[spkr] = totals.get(spkr, 0.0) + (turn.end - turn.start)
        if not totals:
            return waveform
        target_label = max(totals, key=totals.get)

        # Extract target speaker segments
        segments = []
        for turn, _, spkr in diarization.itertracks(yield_label=True):
            if spkr == target_label:
                start = int(turn.start * self.target_sample_rate)
                end = int(turn.end * self.target_sample_rate)
                if end > start:
                    segments.append(waveform[start:end])

        return np.concatenate(segments) if segments else waveform
        # if segments:
        #    audio = np.concatenate(segments)
        # soundfile.write(output_path, audio, sr)

    def convert_audio_files_to_wav(self, input_dir):
        """Convert all non-wav files to wav format in-place"""
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".wav"):
                    original_path = os.path.join(root, file)
                    wav_path = original_path.rsplit(".", 1)[0] + ".wav"
                    convert_to_wav(original_path, wav_path)

    def process_single_wav_file(self, file_path):
        """Process single wav file through the in-memory pipeline"""
        file_name = os.path.basename(file_path).replace(".wav", "")
        file_name = re.sub("[^\w\s]", "", file_name)
        file_name = re.sub(" ", "_", file_name)

        # In-memory pipeline
        waveform, sr = self.resample_audio(file_path)

        if self.enable_diarize:
            waveform = self.diarize_speaker(waveform)

        # Write final chunks
        # !to do: make a new directory if not already created with the same name as the speaker
        # !self.spkr
        out_repo = f"{self.output_path}/{self.spkr}/"
        if not os.path.exists(out_repo):
            os.makedirs(out_repo, exist_ok=False)

        for i, chunk in enumerate(self.slice(waveform)):
            soundfile.write(f"{out_repo}/{file_name}_{i}.wav", chunk, sr)

    def process_raw_audio_dir(self, input_dir):
        """Main entry point - convert then process all wav files"""
        # Step 1: Convert everything to wav
        self.convert_audio_files_to_wav(input_dir)

        # Step 2: Process each wav file
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    self.process_single_wav_file(file_path)

    # def preview_speakers_with_samples(self, audio_path, sample_duration=3.0):
    #     """Preview speakers with audio samples for manual verification"""
    #     from pyannote.audio import Pipeline
    #     import IPython.display as ipd  # For Jupyter playback
    #
    #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    #     diarization = pipeline(audio_path)
    #
    #     audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)
    #
    #     speaker_info = {}
    #     for turn, _, speaker in diarization.itertracks(yield_label=True):
    #         if speaker not in speaker_info:
    #             speaker_info[speaker] = {'segments': [], 'total_time': 0}
    #
    #         speaker_info[speaker]['segments'].append((turn.start, turn.end))
    #         speaker_info[speaker]['total_time'] += (turn.end - turn.start)
    #
    #     # Create sample clips for each speaker
    #     samples = {}
    #     for speaker, info in speaker_info.items():
    #         # Get first segment that's long enough
    #         for start, end in info['segments']:
    #             if end - start >= sample_duration:
    #                 start_sample = int(start * sr)
    #                 end_sample = int(min(start + sample_duration, end) * sr)
    #                 samples[speaker] = audio[start_sample:end_sample]
    #                 break
    #
    #     # Display info and play samples
    #     print("\n🎵 SPEAKER PREVIEW:")
    #     for speaker, info in sorted(speaker_info.items()):
    #         print(f"\n{speaker}: {len(info['segments'])} segments, {info['total_time']:.1f}s total")
    #         if speaker in samples:
    #             # Save sample for playback
    #             sample_path = f"temp_{speaker}_sample.wav"
    #             soundfile.write(sample_path, samples[speaker], sr)
    #             print(f"Sample: {sample_path}")
    #             # In Jupyter: ipd.Audio(sample_path)
    #
    #     return speaker_info, samples

    # def merge_speakers(self, audio_path, speaker_ids_to_merge, new_label="SPEAKER_00"):
    #     """Merge multiple speakers into one target speaker"""
    #     from pyannote.audio import Pipeline
    #
    #     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    #     diarization = pipeline(audio_path)
    #
    #     audio, sr = librosa.load(audio_path, sr=self.target_sample_rate)
    #     merged_segments = []
    #
    #     for turn, _, speaker in diarization.itertracks(yield_label=True):
    #         if speaker in speaker_ids_to_merge:
    #             start = int(turn.start * sr)
    #             end = int(turn.end * sr)
    #             merged_segments.append(audio[start:end])
    #
    #     if merged_segments:
    #         merged_audio = np.concatenate(merged_segments)
    #         return merged_audio, sr
    #     return audio, sr

    # def interactive_speaker_selection(self, audio_path):
    #     """Interactive workflow for speaker selection and merging"""
    #     speaker_info, samples = self.preview_speakers_with_samples(audio_path)
    #
    #     print("\n🎯 OPTIONS:")
    #     print("1. Auto-select dominant speaker")
    #     print("2. Manually select speaker ID")
    #     print("3. Merge multiple speakers")
    #
    #     # In practice, you'd get user input here
    #     # For now, return dominant speaker
    #     dominant = max(speaker_info.keys(), key=lambda s: speaker_info[s]['total_time'])
    #     speaker_id = int(dominant.split("_")[1])
    #
    #     print(f"\n✅ Selected: {dominant} (ID: {speaker_id})")
    #     return speaker_id


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

    # fmt: off
    parser = argparse.ArgumentParser(description="EZ_RVC preprocessing")
    parser.add_argument("--output_path", type=str, default="./data/44k")
    parser.add_argument("--enable_diarize", type=bool, default=True)
    parser.add_argument("--raw_audios", type=str, default=raw_audio_dir, required=True, help="Path to raw audio files (directory or single file).")
    parser.add_argument("--speaker_name", type=str, default=speaker, required=True, help="Speaker name identifier (e.g., 'nat_king_cole').")
    parser.add_argument("--speech_encoder", type=str, default="hubertsoft",
                        choices=["hubertsoft", "vec768l12", "vec256l9", "whisper-ppg", "whisper-ppg-large", "cnhubertlarge", "dphubert", "wavlmbase+"], help="Speech encoder type.")
    parser.add_argument("--target_sample_rate", type=int, default=44100, help="Target sample rate for final WAV chunks (e.g., 44100).")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum duration (seconds) to keep after trimming.")
    parser.add_argument("--top_db", type=float, default=30.0, help="Silence threshold in dB for trimming.")
    parser.add_argument("--max_length", type=float, default=10.0, help="Maximum segment length (seconds) when splitting long non-silent regions.")
    parser.add_argument("--frame_seconds", type=float, default=0.5, help="Frame length (seconds) used for silence detection.")
    parser.add_argument("--hop_seconds", type=float, default=0.1, help="Hop length (seconds) used for silence detection.")

    args = parser.parse_args()
    # fmt: on

    Pre = Preprocessor(
        output_path=args.output_path,
        enable_diarize=args.enable_diarize,
        speaker_name=args.speaker_name,
        speech_encoder=args.speech_encoder,
        target_sample_rate=args.target_sample_rate,
        min_duration=args.min_duration,
        top_db=args.top_db,
        max_length=args.max_length,
        frame_seconds=args.frame_seconds,
        hop_seconds=args.hop_seconds,
    )

    #! smoke test
    Pre.process_raw_audio_dir(input_dir=raw_audio_dir)

    '''
      From your training code, the model expects these inputs:
      c, f0, spec, y, spkr, lengths, uv, volume = items

      Where:
      - c = Content features (from HuBERT/ContentVec encoder)
      - f0 = Fundamental frequency (pitch)
      - spec = Spectrogram
      - y = Raw audio waveform
      - spk = Speaker ID
      - lengths = Sequence lengths
      - uv = Unvoiced/voiced flags
      - volume = Volume features

      Preprocessing Pipeline Breakdown:

      1. Slicer (for all)

      - Purpose: Splits long audio into manageable chunks (removes silence)
      - Output: Multiple WAV files (chunks)
      - Maps to: y (raw audio), lengths

      2. Resampling

      - Purpose: Ensures consistent sample rate (44.1kHz)
      - Output: Resampled WAV files
      - Maps to: y (raw audio)

      3. Speaker Diarization

      - Purpose: Identifies and separates different speakers in multi-speaker audio
      - Output: Cleaned single-speaker audio segments
      - Maps to: spk (speaker ID), cleaner y

      4. HuBERT F0 Processing (preprocess_hubert_f0.py)

      - Purpose: Extracts content features + pitch
      - Outputs:
        - Content features: c (HuBERT/ContentVec embeddings)
        - F0 (pitch): f0
        - Unvoiced flags: uv
        - Spectrograms: spec
        - Volume: volume

      Missing from Current Preprocessor:

      Your current preprocess_v2.py only does slicing and format conversion. You're
      missing:

      1. F0 extraction → preprocess_hubert_f0.py
      2. File list generation → Creates train/val splits
      3. Feature extraction pipeline → Content features, spectrograms, etc.

      The complete flow should be:
      Raw Audio → Slice → Resample → Diarize → Extract Features (F0, content, spec) →
       File Lists → Training
    '''
