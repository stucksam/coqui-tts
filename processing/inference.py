import os
import shutil

import pandas as pd
import torch

from TTS.api import TTS

CLUSTER_HOME_PATH = "cluster/home/stku"
CLUSTER_PROJECTS_PATH = "/cluster/projects/TTS-Swiss-German"
# CLUSTER_HOME_PATH = "/home/ubuntu/ma/"
OUT_PATH = "/scratch/eval"
SPEAKER_DIRECTORY = os.path.join(CLUSTER_HOME_PATH, "_speakers")
MODEL_CHECKPOINTS_PATH = os.path.join(CLUSTER_PROJECTS_PATH, "checkpoints")

CHECKPOINT_MODEL_SEARCH = "checkpoint_"
BEST_MODEL_SEARCH = "best_model_"
INFERENCE_MODEL_NAME = "model.pth"

os.makedirs(OUT_PATH, exist_ok=True)

LANG_MAP = {
    'ch_be': 'Bern',
    'ch_bs': 'Basel',
    'ch_gr': 'Graubünden',
    'ch_in': 'Innerschweiz',
    'ch_os': 'Ostschweiz',
    'ch_vs': 'Wallis',
    'ch_zh': 'Zürich',
    'de': 'Deutschland'
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}

PHON_DID_CLS = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Deutschland"}


def assert_checkpoint_folder_contains_model(checkpoint_dir: str) -> None:
    """
    As each checkpoint folder must have a "model.pth" in order for the inference to work, we need to ensure this is the
    case.
    """
    models = [file for file in os.listdir(checkpoint_dir)
              if CHECKPOINT_MODEL_SEARCH in file or BEST_MODEL_SEARCH in file or INFERENCE_MODEL_NAME in file]
    if INFERENCE_MODEL_NAME in models:
        return
    else:
        highest_step_count = 0
        longest_trained_model = ""
        for model in models:
            steps = int(model.split("_")[-1].replace(".pth", ""))
            if highest_step_count < steps:
                longest_trained_model = model
                highest_step_count = steps

        shutil.copyfile(os.path.join(checkpoint_dir, longest_trained_model), os.path.join(checkpoint_dir, "model.pth"))


def run_inference(directory: str) -> None:
    folder_name = os.path.basename(os.path.normpath(directory))
    model_path = os.path.join(OUT_PATH, folder_name)
    os.makedirs(model_path, exist_ok=True)
    save_eval_path = os.path.join(CLUSTER_PROJECTS_PATH, "generated_speech", folder_name)
    os.makedirs(model_path, exist_ok=True)

    config_path = os.path.join(model_path, "config.json")
    vocab_path = os.path.join(model_path, "vocab.json")

    shutil.copyfile(os.path.join(directory, INFERENCE_MODEL_NAME), os.path.join(model_path, INFERENCE_MODEL_NAME))
    shutil.copyfile(os.path.join(directory, "config.json"), config_path)
    shutil.copyfile(os.path.join(directory, "vocab.json"), vocab_path)

    # Init TTS
    tts = TTS(
        model_path=model_path,
        config_path=config_path,
        progress_bar=True
    ).to(device)

    generated_speech_path = os.path.join(model_path, "generated_speech")
    for speaker, wav in speaker_wavs.items():

        out_wav_path = os.path.join(generated_speech_path, speaker)
        os.makedirs(out_wav_path, exist_ok=True)

        df_speaker = []
        for dial_tag in LANG_MAP.keys():
            for tid, text in enumerate(texts):
                file_path = os.path.join(out_wav_path, f"{tid}_{LANG_MAP[dial_tag]}")
                tts.tts_to_file(text=text, speaker_wav=wav, language=dial_tag, split_sentences=False,
                                file_path=file_path)
                df_speaker.append({"tid": tid, "text": text, "dialect": LANG_MAP[dial_tag], "speaker": speaker, "file_path": file_path})

        df_speaker = pd.DataFrame(df_speaker)
        df_speaker.to_csv(os.path.join(out_wav_path, f"{speaker}.csv"), index=False, encoding="utf-8", sep=";")

    df_texts = []
    for tid, text in enumerate(texts):
        df_texts.append({"tid": tid, "text": text})
    df_texts = pd.DataFrame(df_texts)
    df_texts.to_csv(os.path.join(generated_speech_path, "texts.csv"), index=False, encoding="utf-8", sep=";")

    shutil.copytree(generated_speech_path, save_eval_path, dirs_exist_ok=True)
    # cleanup scratch
    shutil.rmtree(model_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = ["Das ist ein Beispielsatz, welcher auf Schweizerdeutsch ausgesprochen werden soll."]
    speaker_wavs = {}
    for dial_tag in LANG_MAP.keys():
        ref_path = os.path.join(SPEAKER_DIRECTORY, dial_tag, "references")
        refs = os.listdir(ref_path)

        for speaker in refs:
            wav_files = os.listdir(os.path.join(ref_path, speaker))
            speaker_wavs[speaker] = [os.path.join(ref_path, speaker, wav) for wav in wav_files]

    list_of_directories = [os.path.join(MODEL_CHECKPOINTS_PATH, model_dir) for model_dir in
                           os.listdir(MODEL_CHECKPOINTS_PATH) if "SwissGPC" in model_dir]

    for directory in list_of_directories:
        assert_checkpoint_folder_contains_model(directory)
        run_inference(directory)
