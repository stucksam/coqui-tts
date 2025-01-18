import os

import librosa
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
                filtered_df = df[(df["dialect_region"] == dialect) & (df["client_id"] == speaker_id)]
                assert len(filtered_df) == 1, f"Found multiple entries for wav file in transcription.\n{filtered_df}"
                text = filtered_df["sentence"].iloc[0]
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


def save_snf_short(path, speech_folder: str = "generated_speech"):
    # Path to your audio files
    audio_folder = path + "/" + speech_folder
    metadata = load_transcribed_metadata(path + "/" + TEXT_TRANSCRIBED_FILE)
    references = load_reference_sentences(path + "/" + TEXT_METADATA_FILE)
    test_meta = "SNF_Test_Sentences.xlsx"
    test_meta_path = f"processing/data/{test_meta}"
    df_snf_short = pd.read_excel(test_meta_path, sheet_name="SNF Test Samples")

    # Gather all audio file paths and optionally their metadata
    paths = []
    text_ids = []
    dialects = []
    sample_ids = []
    speaker_ids = []
    names = []
    texts = []
    durations = []
    genders = []
    tokens = []
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):  # You can filter other formats as needed
            file_split = file_name.replace(".wav", "").split("_")
            text_id = int(file_split[0])
            dialect = file_split[1]

            file_path = os.path.join(audio_folder, file_name)
            duration = librosa.get_duration(path=file_path, sr=24000)

            paths.append(file_path)
            text_ids.append(text_id)
            dialects.append(dialect)

            clip = [entry for entry in metadata if entry.orig_text_id == text_id and entry.dialect == dialect][0]
            ref = [entry for entry in references if entry.sample_id == clip.orig_text_id][0]
            sample_ids.append(ref.snf_sample_id)
            texts.append(ref.text)

            filtered_df = df_snf_short[
                (df_snf_short["sentence"] == ref.text) & (df_snf_short["dialect_region"] == dialect)]

            name = f"{dialect}_{ref.snf_sample_id}"
            names.append(name)
            speaker_ids.append(filtered_df["client_id"].iloc[0])
            genders.append(filtered_df["gender"].iloc[0])
            durations.append(duration)
            tokens.append(filtered_df["token_count"].iloc[0])

    # Create a Dataset
    # important is name, should be in form of Bern_d0b2fb58b9f562292d26e82d2698ad337e5cd05a822adeaa269f05a2783f1e4d for all datasets
    data = {"audio": paths, "name": names, "text_id": text_ids, "dialect": dialects, "sample_id": sample_ids,
            "speaker_id": speaker_ids,
            "text": texts, "duration": durations, "token": tokens, "gender": genders}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{path}/snf_short_dataset")


def save_snf_long(path, speech_folder: str = "generated_speech_snf_long"):
    # Path to your audio files
    audio_folder = path + "/" + speech_folder
    df_snf_long = pd.read_csv("processing/data/9_2_enriched_snf_long.csv", sep=",")

    # Gather all audio file paths and optionally their metadata
    paths = []
    text_ids = []
    dialects = []
    sample_ids = []
    speaker_ids = []
    names = []
    texts = []
    durations = []
    tokens = []
    genders = []
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):  # You can filter other formats as needed
            file_split = file_name.replace(".wav", "").split("_")
            text_id = int(file_split[0])
            dialect = file_split[1]

            file_path = os.path.join(audio_folder, file_name)
            duration = librosa.get_duration(path=file_path, sr=24000)

            paths.append(file_path)

            filtered_df = df_snf_long[(df_snf_long["ref_id"] == text_id) & (df_snf_long["dialect_region"] == dialect)]
            assert len(filtered_df) == 1, f"Found multiple entries for wav file in transcription.\n{filtered_df}"
            text_ids.append(text_id)
            dialects.append(dialect)

            sample_ids.append(filtered_df["gen_id"].iloc[0])
            texts.append(filtered_df['ref_text'].iloc[0])
            speaker_ids.append(filtered_df["speaker_id"].iloc[0])
            genders.append(filtered_df["gender"].iloc[0])

            name = f"{dialect}_{filtered_df['sentence_id'].iloc[0]}"
            names.append(name)
            durations.append(duration)
            tokens.append(filtered_df['token_count'].iloc[0])

    # Create a Dataset
    # important is name, should be in form of Bern_d0b2fb58b9f562292d26e82d2698ad337e5cd05a822adeaa269f05a2783f1e4d for all datasets
    data = {"audio": paths, "name": names, "text_id": text_ids, "dialect": dialects, "sample_id": sample_ids,
            "speaker_id": speaker_ids,
            "text": texts, "duration": durations, "token": tokens, "gender": genders}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{path}/snf_long_dataset")


def save_gpt_long(path, speech_folder: str = "generated_speech_gpt_long"):
    # Path to your audio files
    audio_folder = path + "/" + speech_folder
    df_gpt_long = pd.read_csv("processing/data/7_5_enriched_gpt_long.csv", sep=",")

    # Gather all audio file paths and optionally their metadata
    paths = []
    text_ids = []
    dialects = []
    sample_ids = []
    speaker_ids = []
    names = []
    texts = []
    durations = []
    genders = []
    tokens = []
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):  # You can filter other formats as needed
            file_split = file_name.replace(".wav", "").split("_")
            text_id = int(file_split[0])
            dialect = file_split[1]
            speaker_id = file_split[2]

            file_path = os.path.join(audio_folder, file_name)
            duration = librosa.get_duration(path=file_path, sr=24000)

            paths.append(file_path)

            filtered_df = df_gpt_long[
                (df_gpt_long["dialect_region"] == dialect) & (df_gpt_long["speaker_id"] == speaker_id) & (
                            df_gpt_long["ref_id"] == text_id)]
            assert len(filtered_df) == 1, f"Found multiple entries for wav file in transcription.\n{filtered_df}"

            text_ids.append(text_id)
            dialects.append(dialect)

            sample_ids.append(filtered_df["gen_id"].iloc[0])
            texts.append(filtered_df["ref_text"].iloc[0])
            speaker_ids.append(filtered_df["speaker_id"].iloc[0])
            genders.append(filtered_df["gender"].iloc[0])

            name = f"{dialect}_{filtered_df['sentence_id'].iloc[0]}-{filtered_df['ref_id'].iloc[0]}"
            names.append(name)
            durations.append(duration)
            tokens.append(filtered_df['token_count'].iloc[0])

    # Create a Dataset
    # important is name, should be in form of Bern_d0b2fb58b9f562292d26e82d2698ad337e5cd05a822adeaa269f05a2783f1e4d for all datasets
    data = {"audio": paths, "name": names, "text_id": text_ids, "dialect": dialects, "sample_id": sample_ids,
            "speaker_id": speaker_ids,
            "text": texts, "duration": durations, "token": tokens, "gender": genders}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{path}/gpt_long_dataset")


def save_conditioning_to_dataset():
    audio_folder = "speakers/"

    paths = []
    speaker_ids = []
    dialects = []
    sample_ids = []
    names = []
    texts = []
    genders = []
    durations = []

    test_meta = "test.tsv"
    test_meta_path = f"processing/data/{test_meta}"
    df = pd.read_csv(test_meta_path, sep="\t")

    for dial_tag in LANG_MAP.keys():
        path = f"{audio_folder}/{dial_tag}/references"
        dialect = LANG_MAP[dial_tag]
        for speaker in os.listdir(path):
            speaker_path = f"{path}/{speaker}"
            for wav in os.listdir(speaker_path):
                sample_id = wav.replace(".wav", "")
                print(f"{speaker}: {dialect} -> {sample_id}")
                filtered = df[
                    (df["dialect_region"] == dialect) & (df["client_id"] == speaker) & (df["sample_id"] == sample_id)]

                file_path = f"{speaker_path}/{wav}"
                paths.append(file_path)
                names.append(f"{dialect}_{sample_id}")
                speaker_ids.append(speaker)
                dialects.append(dialect)
                sample_ids.append(sample_id)
                texts.append(filtered["sentence"].iloc[0])
                genders.append(filtered["gender"].iloc[0])
                durations.append(filtered["duration"].iloc[0])

    # Create a Dataset
    data = {"audio": paths, "name": names, "speaker_id": speaker_ids, "dialect": dialects, "sample_id": sample_ids,
            "text": texts, "gender": genders, "duration": durations}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{audio_folder}/cond_dataset")


def save_snf_to_dataset():
    audio_folder = "speakers/clips__test"

    paths = []
    speaker_ids = []
    dialects = []
    sample_ids = []
    names = []
    texts = []
    genders = []
    durations = []

    test_meta = "test.tsv"
    test_meta_path = f"processing/data/{test_meta}"
    df_test = pd.read_csv(test_meta_path, sep="\t")

    test_meta = "SNF_Test_Sentences.xlsx"
    test_meta_path = f"processing/data/{test_meta}"
    df_xtts = pd.read_excel(test_meta_path, sheet_name="SNF Test Samples")

    speakers = set(df_xtts["client_id"])
    for speaker in speakers:
        path = f"{audio_folder}/{speaker}"
        for wav in os.listdir(path):
            sample_id = wav.replace(".mp3", "")
            filtered = df_test[
                (df_test["client_id"] == speaker) & (df_test["sample_id"] == sample_id)]
            dialect = filtered["dialect_region"].iloc[0]

            paths.append(f"{path}/{wav}")
            names.append(f"{dialect}_{sample_id}")
            speaker_ids.append(speaker)
            dialects.append(dialect)
            sample_ids.append(sample_id)
            texts.append(filtered["sentence"].iloc[0])
            genders.append(filtered["gender"].iloc[0])
            durations.append(filtered["duration"].iloc[0])

    # Create a Dataset
    data = {"audio": paths, "name": names, "speaker_id": speaker_ids, "dialect": dialects, "sample_id": sample_ids,
            "text": texts, "gender": genders, "duration": durations}
    dataset = Dataset.from_dict(data)

    # Cast the 'audio' column to the Audio feature type
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    # Inspect the dataset
    print(dataset)

    # Save the dataset to disk
    dataset.save_to_disk(f"{audio_folder}/snf_dataset")


if __name__ == "__main__":
    save_snf_short("generated_speech/GPT_XTTS_v2.0_Full_7_5")
    save_snf_long("generated_speech/GPT_XTTS_v2.0_Full_9_2_SNF")
    save_gpt_long("generated_speech/GPT_XTTS_v2.0_Full_7_5")
    # save_conditioning_to_dataset()
    # save_snf_to_dataset()
    # save_ref_dataset()
