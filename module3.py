import fire
import librosa
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)


SR = 48000


def calc_metrics_for_given_snr(snr_db: float):
    clear_voice, _ = librosa.load("data/module1/gt.wav", sr=SR)
    mixture, _ = librosa.load(
        f"data/module3/{snr_db}_dB_mixture_DeepFilterNet2.wav", sr=SR
    )

    # Считаем SNR, SDR, SI-SDR, PESQ
    mixture_tensor = torch.tensor(mixture)
    clear_voice_tensor = torch.tensor(clear_voice)
    print("SNR:", SignalNoiseRatio()(mixture_tensor, clear_voice_tensor).item())
    print("SDR:", SignalDistortionRatio()(mixture_tensor, clear_voice_tensor).item())
    print(
        "SI-SDR:",
        ScaleInvariantSignalDistortionRatio()(mixture_tensor, clear_voice_tensor).item(),
    )
    mixture_tensor = torch.tensor(librosa.resample(mixture, orig_sr=SR, target_sr=16000))
    clear_voice_tensor = torch.tensor(
        librosa.resample(clear_voice, orig_sr=SR, target_sr=16000)
    )
    print(
        "PESQ:",
        PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")(
            mixture_tensor, clear_voice_tensor
        ).item(),
    )


if __name__ == "__main__":
    fire.Fire()
