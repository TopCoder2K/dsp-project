import librosa
import soundfile as sf


def downsample_to_48k():
    """Понижает частоту дискретизации с 96 кГц до 48 кГц."""

    filenames = [
        "data/audiocheck.net_hdsweep_1Hz_48000Hz_-3dBFS_30s.wav",
        "data/audiocheck.net_pink_96k_-3dBFS.wav",
    ]
    for fname in filenames:
        y, sr = librosa.load(fname, sr=96000)
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sf.write(
            "data/" + fname.split("_")[1] + "_downsampled_to_48kHz.wav",
            y,
            48000,
            format="wav",
        )


if __name__ == "__main__":
    downsample_to_48k()
