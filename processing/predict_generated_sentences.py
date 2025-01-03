import os
import os
import shutil

import jiwer
import librosa
import pandas as pd
import seaborn as sns
import torch
from bert_score import score as bert_score
from joblib import load
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2Processor, Pipeline, Wav2Vec2ForCTC

from util import CLUSTER_HOME_PATH, TEXT_TRANSCRIBED_FILE, TEXT_METADATA_FILE, OUT_PATH, XTTS_MODEL_TRAINED, \
    DID_REF_FOLDER, GENERATED_SPEECH_FOLDER, GENERATE_SPEECH_PATH, XTTSDataPoint, phon_did_cls, ReferenceDatapoint

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MODEL_PATH = f"{CLUSTER_HOME_PATH}/swiss-phonemes-tts/src/model"
MODEL_PATH_DE_CH = f"{MODEL_PATH}/de_to_ch_large_2"
MODEL_PATH_DID = f"{MODEL_PATH}/text_clf_3_ch_de.joblib"
MODEL_AUDIO_PHONEME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_WHISPER_v3 = "openai/whisper-large-v3"
MODEL_T5_TOKENIZER = "google/t5-v1_1-large"

MISSING_TEXT = "NO_TEXT"
MISSING_PHONEME = "NO_PHONEME"
BATCH_SIZE = 32


def setup_gpu_device() -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype


def load_transcribed_metadata() -> list[XTTSDataPoint]:
    metadata = []
    # with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_TRANSCRIBED_FILE), "rt", encoding="utf-8") as f:
    with open("data/" + TEXT_TRANSCRIBED_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            metadata.append(XTTSDataPoint.load_single_datapoint(split_line))
    return metadata


def load_reference_sentences() -> list[ReferenceDatapoint]:
    references = []
    # with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE), "rt", encoding="utf-8") as f:
    with open("data/" + TEXT_METADATA_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            references.append(ReferenceDatapoint.load_single_ref_datapoint(split_line))
    return references


def load_phoneme_for_did() -> dict:
    phonemes = {}
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, DID_REF_FOLDER, TEXT_TRANSCRIBED_FILE), "r",
              encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            speaker = split_line[0].split("_")[0]
            if speaker not in phonemes:
                phonemes[speaker] = ""
            phonemes[speaker] += f"{split_line[1].replace(' ', '')} "
    return phonemes


def write_to_file(content: list[XTTSDataPoint]) -> None:
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_TRANSCRIBED_FILE), "wt", encoding="utf-8") as f:
        for line in content:
            f.write(line.to_string())


def load_orig_texts() -> list[str]:
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, "texts.txt"), "rt", encoding="utf-8") as f:
        texts = [text.split("\t")[1] for text in f]
    return texts


def get_filenames_of_wavs():
    return [file for file in os.listdir(GENERATE_SPEECH_PATH) if file.endswith(".wav")]


def copy_files_to_scratch_partition():
    shutil.copytree(
        os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, GENERATED_SPEECH_FOLDER),
        GENERATE_SPEECH_PATH,
        dirs_exist_ok=True
    )
    did_samples_path = os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, DID_REF_FOLDER)
    if os.path.exists(did_samples_path):
        shutil.copytree(
            did_samples_path,
            os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, DID_REF_FOLDER),
            dirs_exist_ok=True
        )

    shutil.copy2(
        os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE),
        os.path.join(OUT_PATH, XTTS_MODEL_TRAINED)
    )
    transcribe_path = os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED,
                                   TEXT_TRANSCRIBED_FILE)
    if os.path.exists(transcribe_path):
        shutil.copy2(
            transcribe_path,
            os.path.join(OUT_PATH, XTTS_MODEL_TRAINED)
        )


def _setup_german_transcription_model():
    device, torch_dtype = setup_gpu_device()
    # could do more finetuning under "or more control over the generation parameters, use the model + processor API directly: "
    # in https://github.com/huggingface/distil-whisper/blob/main/README.md
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
        generate_kwargs={"language": "german"}
    )


def transcribe_audio_to_german() -> None:
    # You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
    wav_files = get_filenames_of_wavs()
    num_samples = len(wav_files)

    pipe = _setup_german_transcription_model()
    generated_samples = []
    for start_idx in range(0, num_samples, BATCH_SIZE):
        # Define the batch range
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        # Load batch of audio data
        audio_batch = []
        for i in range(start_idx, end_idx):
            audio_data, _ = librosa.load(f"{GENERATE_SPEECH_PATH}/{wav_files[i]}", sr=None)
            audio_batch.append(audio_data)

        # Perform transcription
        results = pipe(audio_batch, batch_size=BATCH_SIZE, chunk_length_s=30.0)
        for idx, text in enumerate(results):
            text = text["text"].strip()
            filename = wav_files[start_idx + idx]
            text_id = int(filename.split("_")[0])
            dialect = filename.split("_")[1].replace(".wav", "")
            generated_samples.append(XTTSDataPoint(filename, text_id, dialect, text))

    write_to_file(generated_samples)


def _setup_phoneme_model() -> Pipeline:
    device, torch_dtype = setup_gpu_device()

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


def audio_to_phoneme() -> None:
    pipe = _setup_phoneme_model()
    metadata = load_transcribed_metadata()
    num_samples = len(metadata)

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        # Load batch of audio data
        audio_batch = []
        for i in range(start_idx, end_idx):
            audio_data, _ = librosa.load(f"{GENERATE_SPEECH_PATH}/{metadata[i].sample_name}.wav", sr=None)
            audio_batch.append(audio_data)

        # Run phoneme transcription
        results = pipe(audio_batch, batch_size=BATCH_SIZE)

        # Save results to collection
        for idx, result in enumerate(results):
            phoneme = result["text"].strip()
            if phoneme == "":
                phoneme = MISSING_PHONEME
            metadata[start_idx + idx].gen_phoneme = phoneme

    write_to_file(metadata)


def audio_to_phoneme_for_did() -> None:
    pipe = _setup_phoneme_model()

    samples = []
    path = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/{DID_REF_FOLDER}"
    speakers = os.listdir(path)
    for speaker in speakers:
        speaker_dir = f"{path}/{speaker}"
        if not os.path.isdir(speaker_dir):
            continue
        ref_wavs = os.listdir(speaker_dir)
        samples.extend([f"{speaker_dir}/{wav}" for wav in ref_wavs])

    num_samples = len(samples)

    save = []
    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        # Load batch of audio data
        audio_batch = []
        for i in range(start_idx, end_idx):
            audio_data, _ = librosa.load(samples[i], sr=None)
            audio_batch.append(audio_data)

        # Run phoneme transcription
        results = pipe(audio_batch, batch_size=BATCH_SIZE)

        # Save results to collection
        for idx, result in enumerate(results):
            phoneme = result["text"].strip()
            if phoneme == "":
                phoneme = MISSING_PHONEME
            sample_id = "_".join(samples[start_idx + idx].split("/")[-2:]).replace(".wav", "")
            save.append((sample_id, phoneme))
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, DID_REF_FOLDER, TEXT_TRANSCRIBED_FILE), "wt",
              encoding="utf-8") as f:
        for line in save:
            f.write(f"{line[0]}\t{line[1]}\n")
    shutil.copy2(
        os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, DID_REF_FOLDER, TEXT_TRANSCRIBED_FILE),
        os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, DID_REF_FOLDER)
    )


def dialect_identification() -> None:
    metadata = load_transcribed_metadata()
    num_samples = len(metadata)

    text_clf = load(MODEL_PATH_DID)
    text_clf['clf'].set_params(n_jobs=BATCH_SIZE)

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        phonemes = [metadata[i].gen_phoneme for i in range(start_idx, end_idx)]

        predicted = text_clf.predict(phonemes)

        for idx, result in enumerate(predicted):
            did = phon_did_cls[result]
            metadata[start_idx + idx].gen_dialect = did

    write_to_file(metadata)


def dialect_identification_multiple_samples() -> None:
    metadata = load_transcribed_metadata()
    references = load_reference_sentences()
    phoneme_per_speaker = load_phoneme_for_did()
    test_meta = "SNF_Test_Sentences.xlsx"
    test_meta_path = f"/coqui-tts/recipes/ljspeech/xtts_v2/data/{test_meta}"
    df = pd.read_excel(test_meta_path, sheet_name="SNF Test Samples")

    num_samples = len(metadata)

    text_clf = load(MODEL_PATH_DID)
    text_clf['clf'].set_params(n_jobs=BATCH_SIZE)

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        phonemes = []
        for i in range(start_idx, end_idx):
            clip = metadata[i]
            text = [ref.text for ref in references if ref.sample_id == clip.orig_text_id][0]
            speaker = df[(df["dialect_region"] == clip.dialect) & (df["sentence"] == text)]["client_id"]
            print(speaker)
            phonemes.append(phoneme_per_speaker[speaker.iloc[0]])

        predicted = text_clf.predict(phonemes)

        for idx, result in enumerate(predicted):
            did = phon_did_cls[result]
            metadata[start_idx + idx].gen_dialect = did

    write_to_file(metadata)


def calculate_scores(comparison: pd.DataFrame):
    comparison["wer"] = 0.0
    comparison["mer"] = 0.0
    comparison["wil"] = 0.0
    comparison["cer"] = 0.0
    comparison["bert_score"] = 0.0

    for idx, row in comparison.iterrows():
        hypo = row["hypothesis"]
        ref = row["reference"]
        output = jiwer.process_words(ref, hypo)
        comparison.at[idx, "wer"] = output.wer
        comparison.at[idx, "mer"] = output.mer
        comparison.at[idx, "wil"] = output.wil
        comparison.at[idx, "cer"] = jiwer.process_characters(ref, hypo).cer

        # Calculate BERTScore
        P, R, F1 = bert_score([hypo], [ref], lang="de")
        _bert_score = F1.mean().item()
        comparison.at[idx, "bert_score"] = _bert_score

    # Calculate BLEU Score
    reference_split = [ref.split(" ") for ref in comparison["reference"]]
    hypothesis_split = [hyp.split(" ") for hyp in comparison["hypothesis"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    comparison["bleu_score"] = bleu_scores

    reference_split = [ref.lower().split(" ") for ref in comparison["reference"]]
    hypothesis_split = [hyp.lower().split(" ") for hyp in comparison["hypothesis"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    comparison["bleu_score_lower"] = bleu_scores
    return pd.DataFrame(comparison)


def analyze_sentence():
    meta_data = load_transcribed_metadata()

    references = load_reference_sentences()

    comparison = []
    for clip in meta_data:
        ref = [entry for entry in references if entry.sample_id == clip.orig_text_id][0]
        comparison.append({
            "hypothesis": clip.gen_de_text,
            "reference": ref.text,
            "dialect": clip.dialect,
            "sample_id": clip.orig_text_id,
            "snf_sample_id": ref.snf_sample_id
        })
    df = pd.DataFrame(comparison)
    result = calculate_scores(df)
    result.to_excel('sentence_result.xlsx', index=False)


def analyze_did():
    meta_data = load_transcribed_metadata()

    references = [clip.dialect for clip in meta_data]
    hypothesis = [clip.gen_dialect for clip in meta_data]

    # Create a DataFrame for easy manipulation
    data = pd.DataFrame({'Reference': references, 'Hypothesis': hypothesis})

    # Count matches and mismatches
    data['Match'] = data['Reference'] == data['Hypothesis']
    match_count = data['Match'].value_counts()

    # Plot bar chart for matches and mismatches
    plt.figure(figsize=(8, 5))
    sns.barplot(x=match_count.index.map({True: 'Match', False: 'Mismatch'}), y=match_count.values, palette='coolwarm')
    plt.title("Matches vs Mismatches")
    plt.ylabel("Count")
    plt.xlabel("Comparison Result")
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(references, hypothesis, labels=list(set(references + hypothesis)))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(set(references + hypothesis)))

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    disp.plot(cmap='viridis', colorbar=True)
    plt.title("Confusion Matrix of Dialects")
    plt.show()


def analyze_predictions():
    analyze_sentence()
    analyze_did()


def main(analyze_for_did: bool = False):
    os.makedirs(GENERATE_SPEECH_PATH, exist_ok=True)
    copy_files_to_scratch_partition()
    transcribe_audio_to_german()
    audio_to_phoneme()
    if analyze_for_did:
        audio_to_phoneme_for_did()
        dialect_identification_multiple_samples()
        shutil.copy2(
            os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, DID_REF_FOLDER, TEXT_TRANSCRIBED_FILE),
            os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, DID_REF_FOLDER)
        )
    else:
        dialect_identification()

    shutil.copy2(
        os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_TRANSCRIBED_FILE),
        os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED)
    )


def add_sentence_id():
    df = pd.read_excel("data/SNF_Test_Sentences.xlsx")
    texts = []
    with open("data/texts.txt", "r", encoding="utf-8") as f:
        for line in f:
            texts.append(line.split("\t")[1].replace("\n", ""))

    with open("data/texts.txt", "wt", encoding="utf-8") as f:
        for idx, sentence in enumerate(texts):
            text_id = df[df["sentence"] == sentence]["sentence_id"].iloc[0]
            f.write(f"{idx}\t{sentence}\t{text_id}\n")


if __name__ == "__main__":
    # main(True)
    analyze_predictions()
