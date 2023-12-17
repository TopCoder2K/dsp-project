from typing import List

import fire
import librosa
import numpy as np
import soundfile as sf
import torch


SR = 16000
TIMES = [4, 10, 20, 30, 40, 50, 60]
PATH_TO_DATA_DIR = "data/module4/"


def convert_source_to_target(
    src_wav_path: str, target_wav_paths: List[str]
) -> np.ndarray:
    """Переводит текст, озвученный в source, в текст, озвученный голосом из target."""

    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=True,
        trust_repo=True,
        pretrained=True,
        device="cuda",
    )
    query_seq = knn_vc.get_features(src_wav_path)
    matching_set = knn_vc.get_matching_set(target_wav_paths)
    return knn_vc.match(query_seq, matching_set, topk=4)


def downsample_then_convert(src_name: str, target_core_name: str):
    """Сначала переводит все использующиеся файлы в 16 кГц, потом запускает перенос
    голоса для заданных длин записей целевого голоса."""

    src_name = PATH_TO_DATA_DIR + src_name
    target_core_name = PATH_TO_DATA_DIR + target_core_name
    # Перевод всех записей из 48 кГц в 16 кГц
    target_fnames = list()
    for time in TIMES:
        target_fnames.append(target_core_name + f"_{time}sec.wav")
    for fname in [src_name, *target_fnames]:
        y, sr = librosa.load(fname, sr=48000)
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(
            fname.split(".")[0] + "_downsampled_to_16kHz.wav",
            y,
            16000,
            format="wav",
        )

    # Перенос голоса
    suffix = "_downsampled_to_16kHz.wav"
    src_name_without_dir = src_name.split("/")[-1].split(".")[0]
    for fname in target_fnames:
        fname_without_dir = fname.split("/")[-1].split(".")[0]
        print(
            "INPUTS:",
            src_name.split(".")[0] + suffix,
            [
                fname.split(".")[0] + suffix,
            ],
        )
        res_name = f"from_{src_name_without_dir}_to_voice_from_{fname_without_dir}.wav"
        sf.write(
            PATH_TO_DATA_DIR + res_name,
            convert_source_to_target(
                src_name.split(".")[0] + suffix,
                [
                    fname.split(".")[0] + suffix,
                ],
            ),
            16000,
            format="wav",
        )


if __name__ == "__main__":
    fire.Fire()
