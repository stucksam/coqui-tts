import os

import pandas as pd
from datasets import Audio, Dataset

from processing.util import LANG_MAP, load_transcribed_metadata, load_reference_sentences, TEXT_TRANSCRIBED_FILE, \
    TEXT_METADATA_FILE


def save_ref_dataset():
    audio_folder = "speakers/"

    paths = []
    speaker_ids = []
    dialects = []
    sample_ids = []
    names = []
    texts = []

    test_meta = "SNF_Test_Sentences.xlsx"
    test_meta_path = f"processing/data/{test_meta}"
    df = pd.read_excel(test_meta_path, sheet_name="SNF Test Samples")

    for dial_tag in LANG_MAP.keys():
        path = audio_folder + "/" + dial_tag
        dialect = LANG_MAP[dial_tag]
        for file in os.listdir(path):
            if file.endswith(".wav"):  # You can filter other formats as needed
                file_split = file.replace(".wav", "").split("_")
                speaker_id = file_split[0]
                sample_id = file_split[1]
                text = df[(df["dialect_region"] == dialect) & (df["client_id"] == speaker_id)]["sentence"].iloc[0]
                print(text)
                file_path = f"speakers/{dial_tag}/{file}"
                paths.append(file_path)
                names.append(f"{dialect}_{sample_id}")
                speaker_ids.append(speaker_id)
                dialects.append(dialect)
                sample_ids.append(sample_id)
                texts.append(text)

    # Create a Dataset
    data = {"audio": paths, "name": names, "speaker_id": speaker_ids, "dialect": dialects, "sample_id": sample_ids,
            "text": texts}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{audio_folder}/ref_dataset")


def save_to_dataset(path, speech_folder: str = "/generated_speech"):
    # Path to your audio files
    audio_folder = path + speech_folder
    metadata = load_transcribed_metadata(path + "/" + TEXT_TRANSCRIBED_FILE)
    references = load_reference_sentences(path + "/" + TEXT_METADATA_FILE)

    # Gather all audio file paths and optionally their metadata
    paths = []
    text_ids = []
    dialects = []
    sample_id = []
    names = []
    texts = []
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):  # You can filter other formats as needed
            file_split = file_name.replace(".wav", "").split("_")
            text_id = int(file_split[0])
            dialect = file_split[1]
            file_path = os.path.join(audio_folder, file_name)

            paths.append(file_path)
            text_ids.append(text_id)
            dialects.append(dialect)

            clip = [entry for entry in metadata if entry.orig_text_id == text_id and entry.dialect == dialect][0]
            ref = [entry for entry in references if entry.sample_id == clip.orig_text_id][0]
            sample_id.append(ref.sample_id)
            texts.append(ref.text)

            name = f"{dialect}_{ref.snf_sample_id}"
            names.append(name)

    # Create a Dataset
    # important is name, should be in form of Bern_d0b2fb58b9f562292d26e82d2698ad337e5cd05a822adeaa269f05a2783f1e4d for all datasets
    data = {"audio": paths, "name": names, "text_id": text_ids, "dialect": dialects, "sample_id": sample_id,
            "text": texts}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{path}/dataset")


if __name__ == "__main__":
    save_to_dataset("generated_speech/GPT_XTTS_v2.0_Full_3_5_SNF")
    # save_ref_dataset()
