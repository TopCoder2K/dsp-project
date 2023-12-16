import pickle

import fire
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import convolve, deconvolve


SR = 48000  # Sample rate
duration = 30  # Продолжительность свипа и шума в секундах


def get_afc(bins_cnt: int, last_pos_freq: int) -> np.ndarray:
    """Считает коэффициенты искажения звука для каждого из `bins_cnt` бинов."""

    sweep, _ = librosa.load("data/module1/hdsweep_downsampled_to_48kHz.wav", sr=SR)
    recorded_sweep, _ = librosa.load("data/module1/recorded_sweep_cropped.wav", sr=SR)
    # Добавляем нулей в начало для выравнивания длин
    recorded_sweep = np.concatenate(
        (np.zeros(len(sweep) - len(recorded_sweep)), recorded_sweep)
    )
    assert len(sweep) == len(
        recorded_sweep
    ), "Длины оригинального и записанного свипов не совпадают!"

    # Выполняем FFT
    sweep_ft = fft(sweep, norm="ortho")
    recorded_sweep_ft = fft(recorded_sweep, norm="ortho")

    # Отрисовываем АЧХ
    # Отдельно отметим, что из-за сжатия максимальная частота будет почти равна 24000
    # (если быть точным, то max_freq = duration * SR / (duration * SR + 1) / 2 * SR)
    fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    ax.set_title("БПФ исходного и записанного сигналов")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Абсолютное значение коэф-та Фурье")
    # Так как исходный сигнал действительнозначный, то берём слагаемые только
    # при неотрицательных частотах (значения при отрицательных будут симметричны)
    freqs = fftfreq(len(sweep), 1.0 / SR)[: last_pos_freq + 1]
    (recorded_sweep_line,) = ax.plot(
        freqs, np.abs(recorded_sweep_ft[: last_pos_freq + 1]), label="запись"
    )
    (sweep_line,) = ax.plot(
        freqs, np.abs(sweep_ft[: last_pos_freq + 1]), label="исходный"
    )
    ax.legend(handles=[sweep_line, recorded_sweep_line], loc="upper center")
    fig.savefig("data/module1/afc.png")
    # Отрисуем ещё вместе с левыми границами бинов
    ax.set_title(
        f"БПФ сигналов (красные линии - левые границы бинов, кол-во бинов - {bins_cnt})"
    )
    coeffs = list()
    bin_size = (last_pos_freq + 1) // bins_cnt
    for bin_left_idx in range(0, last_pos_freq + 1 - bin_size, bin_size):
        x_in_Hz = bin_left_idx * SR / len(sweep)
        ax.axvline(x=x_in_Hz, ymin=0.0, ymax=1.0, color="r")
        # Так как абсолютные значения частот сильно колеблются, лучше брать среднее
        # (я пробовал с центральным значением, получались слишком маленькие коэф-ты)
        coeffs.append(
            np.average(np.abs(recorded_sweep_ft[bin_left_idx : bin_left_idx + bin_size]))
            / np.average(np.abs(sweep_ft[bin_left_idx : bin_left_idx + bin_size]))
        )
    fig.savefig(f"data/module1/{bins_cnt}_bins_afc_with_bins.png")

    # Возвращаем коэф-ты изменения оригинального звука колонкой ноутбука
    return np.array(coeffs)


def save_corrected_noise_and_gt(bins_cnt: int = 32):
    # Получаем спектр розового шума и АЧХ колонки ноутбука. Считаем, что диапазон частот
    # свипа и шума совпадают (это выполняется для скачанных файлов).
    pink_noise, _ = librosa.load("data/module1/pink_downsampled_to_48kHz.wav", sr=SR)
    pink_noise_ft = fft(pink_noise, norm="ortho")
    # Так как исходный сигнал действительнозначный, то берём слагаемые только
    # при неотрицательных частотах (значения при отрицательных будут симметричны)
    if len(pink_noise) % 2 == 0:
        last_pos_freq = len(pink_noise) // 2 - 1
    else:
        last_pos_freq = (len(pink_noise) - 1) // 2
    afc_coeffs = get_afc(bins_cnt, last_pos_freq)

    afc_corrected_pink_noise_ft = np.zeros_like(pink_noise_ft)
    bin_size = (last_pos_freq + 1) // bins_cnt
    for i, bin_left_idx in enumerate(range(0, last_pos_freq + 1 - bin_size, bin_size)):
        afc_corrected_pink_noise_ft[bin_left_idx : bin_left_idx + bin_size] = (
            pink_noise_ft[bin_left_idx : bin_left_idx + bin_size] / afc_coeffs[i]
        )
    afc_corrected_pink_noise_ft[last_pos_freq + 1 :] = afc_corrected_pink_noise_ft[
        last_pos_freq:0:-1
    ]
    sf.write(
        f"data/module1/{bins_cnt}_bins_afc_corrected_pink_noise_48kHz.wav",
        np.abs(ifft(afc_corrected_pink_noise_ft, norm="ortho")),
        SR,
        format="wav",
    )

    # У меня есть гипотеза, что колонки достаточно сильно портят звук (судя по графику),
    # поэтому для чистоты эксперимента решил преобразовать и тестовый сигнал
    # (прастити за копипасту, некогда было на функции хорошо разносить)
    test, _ = librosa.load("data/module1/gt.wav", sr=SR)
    test_ft = fft(test, norm="ortho")
    afc_corrected_test_ft = np.zeros_like(test_ft)
    if len(test) % 2 == 0:
        last_pos_freq = len(test) // 2 - 1
    else:
        last_pos_freq = (len(test) - 1) // 2
    bin_size = (last_pos_freq + 1) // bins_cnt
    for i, bin_left_idx in enumerate(range(0, last_pos_freq + 2 - bin_size, bin_size)):
        afc_corrected_test_ft[bin_left_idx : bin_left_idx + bin_size] = (
            test_ft[bin_left_idx : bin_left_idx + bin_size] / afc_coeffs[i]
        )
    afc_corrected_test_ft[last_pos_freq + 1 :] = afc_corrected_test_ft[last_pos_freq::-1]
    sf.write(
        f"data/module1/{bins_cnt}_bins_afc_corrected_test_48kHz.wav",
        np.abs(ifft(afc_corrected_test_ft, norm="ortho")),
        SR,
        format="wav",
    )


def get_impulse_responce_and_test():
    # Считаем импульсную характеристику
    pink_noise, _ = librosa.load("data/module1/pink_downsampled_to_48kHz.wav", sr=SR)
    recorded_pink_noise, _ = librosa.load(
        "data/module1/recorded_corrected_pink_noise_cropped.wav", sr=SR
    )
    # Так как после обрезания получилось совсем чуть-чуть подлиннее (примерно на 5 мс),
    # то выровним длины
    recorded_pink_noise = recorded_pink_noise[239:]
    assert len(pink_noise) == len(
        recorded_pink_noise
    ), "Длины оригинального и записанного шумов не совпадают!"
    impulse_responce, _ = deconvolve(recorded_pink_noise, pink_noise)
    with open("data/module1/impulse_responce.p", "wb") as f:
        pickle.dump(impulse_responce, f)

    # Сворачиваем её с тестовым сигналом (посчитаем для двух вариантов:
    # для ачх-адаптированного и для исходного)
    test, _ = librosa.load("data/module1/gt.wav", sr=SR)
    predicted = convolve(impulse_responce, test)
    sf.write("data/module1/predicted_gt.wav", predicted, SR, format="wav")


if __name__ == "__main__":
    fire.Fire()
