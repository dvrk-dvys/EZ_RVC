import json
import os
from pathlib import Path

import librosa
import soundfile as sf


def gen_spk_config(
    speaker_name, speech_encoder="hubertsoft", template_name="config_template.json"
):
    """
    Generate speaker-specific config from template.

    Args:
        speaker_name (str): Name of the speaker (e.g., "nat_king_cole")
        speech_encoder (str): Speech encoder type (default: "hubertsoft")
        template_name (str): Template file to use (default: "config_template.json")

    Returns:
        str: Path to generated config file
    """

    # Paths
    script_dir = Path(__file__).parent
    template_path = script_dir.parent / "configs" / "configs_template" / template_name
    encoder_settings_path = script_dir.parent / "configs" / "encoder_settings.json"
    output_dir = script_dir.parent / "configs" / "44k"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template and encoder settings
    with open(template_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(encoder_settings_path, "r", encoding="utf-8") as f:
        encoder_settings = json.load(f)

    # Update speaker-specific fields
    config["data"]["training_files"] = f"./data/prep/filelists/train_{speaker_name}.txt"
    config["data"]["validation_files"] = f"./data/prep/filelists/val_{speaker_name}.txt"
    config["model"]["speech_encoder"] = speech_encoder
    config["model"]["n_speakers"] = 1
    config["spk"] = {speaker_name: 0}

    # Adjust encoder-specific settings
    if speech_encoder in encoder_settings:
        settings = encoder_settings[speech_encoder]
        config["model"]["gin_channels"] = settings["gin_channels"]
        config["model"]["ssl_dim"] = settings["ssl_dim"]
        print(f"📊 Using {speech_encoder}: {settings['description']}")

    # Save config
    config_filename = f"config_colab_{speaker_name}.json"
    config_path = output_dir / config_filename

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated config: {config_path}")
    return str(config_path)


def convert_to_wav(input_path, output_path):
    """
    Convert audio file to WAV format.
    """
    audio, sr = librosa.load(input_path, sr=None)
    sf.write(output_path, audio, sr)


if __name__ == "__main__":
    # Sanity check
    print("🧪 Testing config generator...")

    speaker = "nat_king_cole"
    config_path = gen_spk_config(speaker)

    # Verify
    with open(config_path, "r") as f:
        config = json.load(f)

    assert config["spk"][speaker] == 0
    assert speaker in config["data"]["training_files"]
    assert config["model"]["speech_encoder"] == "hubertsoft"

    print("✅ Config generated and validated successfully!")
    print(f"📄 Speaker: {speaker}")
    print(f"📄 Training files: {config['data']['training_files']}")
    print(f"📄 Speech encoder: {config['model']['speech_encoder']}")
