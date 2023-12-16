import fire
import librosa
import numpy as np
import soundfile as sf
import torch
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)


SR = 48000


def mixer(original: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Смешивает заданный оригинальный сигнал с заданным шумом так, что SNR получается
    равным заданному значению.
    """
    assert len(original) == len(
        noise
    ), "Ошибка! Длины оригинального сигнала и шума не совпадают!"
    # Обозначения соответствуют README.md
    k = (np.mean(original**2) / np.mean(noise**2) / 10 ** (snr_db / 10)) ** 0.5
    return original + k * noise


def calc_metrics_for_given_snr(snr_db: float):
    clear_voice, _ = librosa.load("data/module1/gt.wav", sr=SR)
    metro_noise, _ = librosa.load("data/module2/noise.wav", sr=SR)
    # Так как в начале и в конце дорожки с шумом метро шума меньше, то поровну обрежем
    # начало и конец
    length_diff = (len(metro_noise) - len(clear_voice)) // 2
    metro_noise = metro_noise[length_diff:-length_diff]
    # Получаем и сохраняем смесь
    mixture = mixer(clear_voice, metro_noise, snr_db)
    sf.write(f"data/module2/{snr_db}_dB_mixture.wav", mixture, SR, format="wav")

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
