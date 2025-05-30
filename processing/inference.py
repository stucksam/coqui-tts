import os
import shutil
import tarfile
from multiprocessing import Process

import librosa
import pandas as pd
import torch
# import whisperx
# from whisperx.asr import FasterWhisperPipeline
from transformers import Wav2Vec2Processor, pipeline, Wav2Vec2ForCTC, Pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

from TTS.api import TTS

CLUSTER_HOME_PATH = "/cluster/home/stku"
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

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MODEL_PATH = os.path.join(CLUSTER_HOME_PATH, "swiss-vs-tts", "models")
MODEL_PATH_DE_CH = os.path.join(MODEL_PATH, "de_to_ch_large_2")
MODEL_PATH_DID = os.path.join(MODEL_PATH, "text_clf_3_ch_de.joblib")
MODEL_PATH_DID_CH_ONLY = os.path.join(MODEL_PATH, "text_clf_5_ch_only.joblib")

MODEL_DOWNLOAD_PATH = os.path.join(OUT_PATH, "download")
os.makedirs(MODEL_DOWNLOAD_PATH, exist_ok=True)

MODEL_AUDIO_PHONEME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_WHISPER_v3 = "openai/whisper-large-v3"
MODEL_T5_TOKENIZER = "google/t5-v1_1-large"

MISSING_TEXT = "NO_TEXT"
MISSING_PHONEME = "NO_PHONEME"
BATCH_SIZE = 32


def setup_gpu_device() -> tuple:
    train_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return train_device, dtype


def collect_speaker_condition_samples():
    wavs = {}
    for dialect in LANG_MAP.keys():
        if dialect == "de":
            continue
        ref_path = os.path.join(SPEAKER_DIRECTORY, dialect, "references")
        refs = os.listdir(ref_path)

        for speaker in refs:
            wav_files = os.listdir(os.path.join(ref_path, speaker))
            wavs[speaker] = [os.path.join(ref_path, speaker, wav) for wav in wav_files]
    return wavs


def assert_checkpoint_folder_contains_model(checkpoint_dir: str) -> None:
    """
    As each checkpoint folder must have a "model.pth" in order for the inference to work, we need to ensure this is the
    case.
    """
    print(f"Checking existence of {INFERENCE_MODEL_NAME} in {checkpoint_dir}")
    models = [file for file in os.listdir(checkpoint_dir)
              if CHECKPOINT_MODEL_SEARCH in file or BEST_MODEL_SEARCH in file or INFERENCE_MODEL_NAME == file]
    if INFERENCE_MODEL_NAME in models:
        print(f"{INFERENCE_MODEL_NAME} already exists for {checkpoint_dir}...")
        return
    else:
        highest_step_count = 0
        longest_trained_model = ""
        for model in models:
            steps = int(model.split("_")[-1].replace(".pth", ""))
            if highest_step_count < steps:
                longest_trained_model = model
                highest_step_count = steps

        print(f"Longest trained model at {highest_step_count} is {longest_trained_model} "
              f"-> will be used for inference.")
        shutil.copyfile(os.path.join(checkpoint_dir, longest_trained_model), os.path.join(checkpoint_dir, "model.pth"))


def save_to_csv(entries: list | pd.DataFrame, path: str) -> None:
    if isinstance(entries, list):
        entries = pd.DataFrame(entries)

    entries.to_csv(path, index=False, encoding="utf-8", sep=";")


def combine_all_speaker_metadata(generated_speech_path: str) -> str:
    combined_csv_path = os.path.join(generated_speech_path, "all_metadata.csv")
    all_dfs = []
    for root, dirs, files in os.walk(generated_speech_path):
        if "metadata.csv" in files:
            csv_path = os.path.join(root, "metadata.csv")
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save the combined CSV
    save_to_csv(combined_df, combined_csv_path)
    return path


def combine_dialect_metadata(generated_speech_path: str) -> None:
    for speaker, wav in speaker_wavs.items():
        out_wav_path = os.path.join(generated_speech_path, speaker)
        meta_data_paths = [os.path.join(out_wav_path, file)
                           for file in os.listdir(out_wav_path)
                           if file.endswith(".csv")]
        combined_df = pd.concat([pd.read_csv(path) for path in meta_data_paths], ignore_index=True)

        save_to_csv(combined_df, os.path.join(out_wav_path, "metadata.csv"))

        for path in meta_data_paths:  # delete dialect split metadata
            os.remove(path)


def run_inference_for_dialect(model_path: str, config_path: str, dial_tag: str) -> None:
    """
    Aimed at being run in parallel for each dialect to reduce inference time from 28h -> 3h.
    """
    print(f"Loading model for dialect {dial_tag}...")
    # Init TTS
    tts = TTS(
        model_path=model_path,
        config_path=config_path,
        progress_bar=True
    ).to(device)

    generated_speech_path = os.path.join(model_path, "generated_speech")

    print(f"Starting inference in {dial_tag} for all speakers.")
    for speaker, wav in speaker_wavs.items():

        out_wav_path = os.path.join(generated_speech_path, speaker)

        df_dialect = []
        for tid, text in enumerate(texts):
            file_path = os.path.join(out_wav_path, f"{tid}_{LANG_MAP[dial_tag]}")
            tts.tts_to_file(text=text, speaker_wav=wav, language=dial_tag, split_sentences=True,
                            file_path=file_path)

            speaker_wav_path = os.path.join(speaker, f"{tid}_{LANG_MAP[dial_tag]}.wav")
            entry = {"tid": tid, "text": text, "dialect": LANG_MAP[dial_tag], "speaker": speaker,
                     "file_path": speaker_wav_path}
            df_dialect.append(entry)

        save_to_csv(df_dialect, os.path.join(out_wav_path, f"metadata_{dial_tag}.csv"))


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

    generated_speech_path = os.path.join(model_path, "generated_speech")

    # Create outdirs before strting inference to reduce potential I/O issues
    for speaker, wav in speaker_wavs.items():
        out_wav_path = os.path.join(generated_speech_path, speaker)
        os.makedirs(out_wav_path, exist_ok=True)

    processes = [
        Process(target=run_inference_for_dialect, args=(model_path, config_path, dial_tag,))
        for dial_tag in LANG_MAP.keys()
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    combine_dialect_metadata(generated_speech_path)
    meta_data_path = combine_all_speaker_metadata(generated_speech_path)

    print("Saving texts to csv...")
    df_texts = []
    for tid, text in enumerate(texts):
        df_texts.append({"tid": tid, "text": text})
    save_to_csv(df_texts, os.path.join(generated_speech_path, "texts.csv"))

    generated_speech_zip = os.path.join(model_path, "generated_speech.zip")
    with tarfile.open(generated_speech_zip, "w:gz") as tar:
        tar.add(generated_speech_path, arcname="generated_speech")

    print("Copying generated speech to projects folder...")
    shutil.copytree(generated_speech_zip, save_eval_path, dirs_exist_ok=True)
    shutil.copyfile(meta_data_path, os.path.join(save_eval_path, "metadata.csv"))

    # cleanup scratch
    print("Deleting generated speech from scratch folder...")
    shutil.rmtree(model_path)


def load_wav_metadata(model_path):
    return pd.read_csv(os.path.join(model_path, "generated_speech", "wav_files.csv"), sep=";", encoding="utf-8")


def run_eval(model_path: str):
    transcribe_audio_to_german_and_phoneme(model_path)
    transcribe_audio_to_german_and_phoneme(model_path)


def _setup_german_transcription_model():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_WHISPER_v3, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_WHISPER_v3)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "german", "no_repeat_ngram_size": 2}
    )


def _setup_phoneme_model() -> Pipeline:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_AUDIO_PHONEME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_AUDIO_PHONEME)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )


def transcribe_audio_to_german_and_phoneme(model_path: str) -> None:
    df = load_wav_metadata(model_path)
    num_samples = len(df)

    pipe_german = _setup_german_transcription_model()
    pipe_phoneme = _setup_phoneme_model()

    german_text = []
    phoneme_text = []
    length_audio = []
    for start_idx in range(0, num_samples, BATCH_SIZE):
        # Define the batch range
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        subset = df.iloc[start_idx:end_idx]
        # Load batch of audio data
        audio_batch = []
        for path in list(subset["file_path"]):
            audio_data, _ = librosa.load(path, sr=None)
            length_audio.append(round(librosa.get_duration(y=audio_data, sr=24000), 4))
            audio_batch.append(audio_data)

        # Perform German transcription
        results_de_text = pipe_german(audio_batch, batch_size=BATCH_SIZE)
        # Run phoneme transcription
        results_phoneme = pipe_phoneme(audio_batch, batch_size=BATCH_SIZE)

        for text in results_de_text:
            german_text.append(text["text"].strip())
        for phonemes in results_phoneme:
            phonemes = phonemes["text"].strip()
            if phonemes == "":
                phonemes = MISSING_PHONEME
            phoneme_text.append(phonemes)

    assert len(df) == len(length_audio), \
        "Detected missmatch between generated number of audio lengths and number of samples"
    df["audio_length"] = length_audio

    assert len(df) == len(german_text), \
        "Detected missmatch between generated number of de texts and number of samples"
    df["de_text"] = german_text

    assert len(df) == len(phoneme_text), \
        "Detected missmatch between generated number of phoneme texts and number of samples"
    df["phoneme"] = phoneme_text

    df.to_csv(os.path.join(model_path, "generated_speech", "transcribed_metadata.csv"), sep=";", index=False,
              encoding="utf-8")


if __name__ == "__main__":
    # Todo: 1. zip all samples generated by model and move to projects
    # Todo: 2. evaluate all samples and move transcribed csv to projects as well

    device, torch_dtype = setup_gpu_device()

    texts = [f"Das ist ein Beispielsatz, welcher auf Schweizerdeutsch ausgesprochen werden soll"]
    speaker_wavs = collect_speaker_condition_samples()
    list_of_directories = [os.path.join(MODEL_CHECKPOINTS_PATH, model_dir) for model_dir in
                           os.listdir(MODEL_CHECKPOINTS_PATH) if "SwissGPC" in model_dir]

    for directory in list_of_directories:
        assert_checkpoint_folder_contains_model(directory)

    run_inference(list_of_directories[0])
