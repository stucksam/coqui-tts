import gc
import os
import shutil
import tarfile
from collections import defaultdict
from multiprocessing import Process, set_start_method

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
import whisperx
from bert_score import score as bert_score
from huggingface_hub import hf_hub_download
from joblib import load
from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2Processor, pipeline, Wav2Vec2ForCTC, Pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

from TTS.api import TTS

CLUSTER_HOME_PATH = "/cluster/home/stku"
CLUSTER_PROJECTS_PATH = "/cluster/projects/TTS-Swiss-German"
# CLUSTER_HOME_PATH = "/home/ubuntu/ma/"
OUT_PATH = "/scratch/eval"
# SOURCE_SPEAKER_DIRECTORY = os.path.join(CLUSTER_HOME_PATH, "_speakers")
SOURCE_SPEAKER_DIRECTORY = os.path.join(CLUSTER_PROJECTS_PATH, "snf_eval_condition_files", "speaker")
INFERENCE_SPEAKER_DIRECTORY = os.path.join(OUT_PATH, "speaker")
COQUI_TTS_PATH = os.path.join(CLUSTER_HOME_PATH, "coqui-tts")
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
PHON_DID_CLS_INV = {v: k for k, v in PHON_DID_CLS.items()}

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

MODEL_PATH = os.path.join("processing", "models")
MODEL_PATH_DID = os.path.join(MODEL_PATH, "text_clf_3_ch_de.joblib")

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
    print(f"Running on device {train_device} with dtype {dtype}")
    return train_device, dtype


def collect_speaker_condition_samples() -> tuple[dict, dict]:
    wavs = {}
    speaker_to_dialect_map = {}
    for dialect in LANG_MAP.keys():
        if dialect == "de":
            continue
        ref_path = os.path.join(INFERENCE_SPEAKER_DIRECTORY, dialect)
        refs = os.listdir(ref_path)

        for speaker in refs:
            wav_files = os.listdir(os.path.join(ref_path, speaker))
            wavs[speaker] = [os.path.join(ref_path, speaker, wav) for wav in wav_files]
            if speaker not in speaker_to_dialect_map:
                speaker_to_dialect_map[speaker] = LANG_MAP[dialect]

    return wavs, speaker_to_dialect_map


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
    combined_csv_path = os.path.join(generated_speech_path, "metadata.csv")
    all_dfs = []
    for root, dirs, files in os.walk(generated_speech_path):
        if "metadata.csv" in files:
            csv_path = os.path.join(root, "metadata.csv")
            df = pd.read_csv(csv_path, delimiter=";", encoding="utf-8")
            all_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save the combined CSV
    save_to_csv(combined_df, combined_csv_path)
    return combined_csv_path


def combine_dialect_metadata(generated_speech_path: str) -> None:
    for speaker, wav in speaker_wavs.items():
        out_wav_path = os.path.join(generated_speech_path, speaker)
        meta_data_paths = [os.path.join(out_wav_path, file)
                           for file in os.listdir(out_wav_path)
                           if file.endswith(".csv")]
        combined_df = pd.concat([pd.read_csv(path, delimiter=";", encoding="utf-8") for path in meta_data_paths],
                                ignore_index=True)

        save_to_csv(combined_df, os.path.join(out_wav_path, "metadata.csv"))

        for path in meta_data_paths:  # delete dialect split metadata
            os.remove(path)


def run_inference_for_dialect(model_path: str, config_path: str, dial_tag: str, device: str, speaker_wavs: dict,
                              speaker_to_dialect: dict, texts: list) -> None:
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
    tts.eval()
    tts.synthesizer.output_sample_rate = 24000

    generated_speech_path = os.path.join(model_path, "generated_speech")

    print(f"Starting inference in {dial_tag} for all speakers.")
    for speaker, wav in speaker_wavs.items():

        out_wav_path = os.path.join(generated_speech_path, speaker)

        df_dialect = []
        for tid, text in enumerate(texts):
            file_path = os.path.join(out_wav_path, f"{tid}_{LANG_MAP[dial_tag]}.wav")
            tts.tts_to_file(text=text, speaker_wav=wav, language=dial_tag, split_sentences=True,
                            file_path=file_path)

            speaker_wav_path = os.path.join(speaker, f"{tid}_{LANG_MAP[dial_tag]}.wav")
            entry = {"tid": tid, "text": text, "speaker": speaker, "orig_dialect": speaker_to_dialect[speaker],
                     "dialect": LANG_MAP[dial_tag], "file_path": speaker_wav_path}
            df_dialect.append(entry)

        save_to_csv(df_dialect, os.path.join(out_wav_path, f"metadata_{dial_tag}.csv"))

    del tts
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Finished inference for {dial_tag} for all speakers.")


def run_inference(model_path: str) -> None:
    if "generated_speech.tar.gz" in os.listdir(save_eval_path):
        print(f"Inference already done, skipping {folder_name}...")
        return

    config_path = os.path.join(model_path, "config.json")
    vocab_path = os.path.join(model_path, "vocab.json")

    shutil.copyfile(os.path.join(directory, INFERENCE_MODEL_NAME), os.path.join(model_path, INFERENCE_MODEL_NAME))
    shutil.copyfile(os.path.join(directory, "config.json"), config_path)
    shutil.copyfile(os.path.join(directory, "vocab.json"), vocab_path)

    generated_speech_path = os.path.join(model_path, "generated_speech")

    # Create outdirs before starting inference to reduce potential I/O issues
    for speaker, wav in speaker_wavs.items():
        out_wav_path = os.path.join(generated_speech_path, speaker)
        os.makedirs(out_wav_path, exist_ok=True)

    processes = [
        Process(target=run_inference_for_dialect, args=(model_path, config_path, dial_tag, device, speaker_wavs,
                                                        speaker_to_dialect, texts,))
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

    generated_speech_zip = os.path.join(model_path, "generated_speech.tar.gz")
    with tarfile.open(generated_speech_zip, "w:gz") as tar:
        tar.add(generated_speech_path, arcname="generated_speech")

    print("Copying generated speech to projects folder...")
    # shutil.file(generated_speech_path, save_eval_path, dirs_exist_ok=True) # uncomment if you want the raw wavs copied
    shutil.copyfile(generated_speech_zip, os.path.join(save_eval_path, "generated_speech.tar.gz"))
    shutil.copyfile(meta_data_path, os.path.join(save_eval_path, "metadata.csv"))

    del processes
    torch.cuda.empty_cache()
    gc.collect()


def load_wav_metadata(model_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(model_path, "generated_speech", "metadata.csv"), sep=";", encoding="utf-8")


def load_transcribed_metadata(model_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(model_path, "generated_speech", "transcribed_metadata.csv"), sep=";",
                       encoding="utf-8")


def run_transcription(model: str) -> None:
    transcribe_audio_to_german_and_phoneme(model)
    classify_dialect(model)
    calculate_speaker_similarity(model)


def _setup_german_transcription_model() -> Pipeline:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_WHISPER_v3, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir=MODEL_PATH
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_WHISPER_v3)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        chunk_length_s=30.0,
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


def _setup_whisperx_model():
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, language="de", compute_type="float16",
                                download_root=MODEL_PATH)
    return model


def transcribe_audio_to_german_and_phoneme(model_path: str) -> None:
    print(f"Transcribing generated samples by {model_path} into German and Phoneme.")

    if os.path.exists(os.path.join(model_path, "generated_speech", "transcribed_metadata.csv")):
        df_transcribed = load_transcribed_metadata(model_path)

        if "gen_text" in df_transcribed.columns or "phoneme" in df_transcribed.columns:
            print("Already transcribed samples, skipping step...")
            return

    df = load_wav_metadata(model_path)
    num_samples = len(df)

    pipe_german = _setup_german_transcription_model()
    # model_whisperx = _setup_whisperx_model()
    pipe_phoneme = _setup_phoneme_model()

    german_text, phoneme_text, length_audio = [], [], []

    for start_idx in range(0, num_samples, BATCH_SIZE):
        # Define the batch range
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        subset = df.iloc[start_idx:end_idx]

        # Load batch of audio data
        audio_batch = []
        for idx, row in subset.iterrows():
            audio_path = os.path.join(model_path, "generated_speech", row["speaker"],
                                      f"{row['tid']}_{row['dialect']}.wav")
            audio_data, _ = librosa.load(audio_path, sr=None)
            length_audio.append(round(librosa.get_duration(y=audio_data, sr=24000), 4))
            audio_batch.append(audio_data)

        # Perform German transcription
        results_de_text = pipe_german(audio_batch.copy(), batch_size=BATCH_SIZE)
        german_text.extend(text["text"].strip() for text in results_de_text)

        # Run batch transcription
        # results_de_text = [model_whisperx.transcribe(audio, chunk_size=15, language="de") for audio in audio_batch.copy()]
        # german_text = [res["segments"][0]["text"].strip() for res in results_de_text]

        # Run phoneme transcription
        results_phoneme = pipe_phoneme(audio_batch.copy(), batch_size=BATCH_SIZE)
        phoneme_text.extend(
            phoneme["text"].strip() if phoneme["text"].strip() else MISSING_PHONEME
            for phoneme in results_phoneme
        )

    assert len(df) == len(length_audio), "Mismatch in audio lengths"
    assert len(df) == len(german_text), "Mismatch in German text"
    assert len(df) == len(phoneme_text), "Mismatch in phoneme text"

    df["audio_length"] = length_audio
    df["gen_text"] = german_text
    df["phoneme"] = phoneme_text

    save_path = os.path.join(model_path, "generated_speech", "transcribed_metadata.csv")
    save_to_csv(df, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "transcribed_metadata.csv"))

    del pipe_german
    # del model_whisperx
    del pipe_phoneme
    torch.cuda.empty_cache()
    gc.collect()


def load_phoneme_for_did(df: pd.DataFrame) -> dict:
    phonemes = defaultdict(lambda: defaultdict(str))
    for _, line in df.iterrows():
        speaker = line["speaker"]
        dialect = line["dialect"]
        phonemes[speaker][dialect] += line["phoneme"].replace(' ', '')
    return {speaker: dict(dials) for speaker, dials in phonemes.items()}


def classify_dialect(model_path: str) -> None:
    print(f"Classifying dialect for generated samples by {model_path}.")

    df = load_transcribed_metadata(model_path)
    if "pred_dialect" in df.columns:
        print("Already classified dialect, skipping dialect classification...")
        return

    df["pred_dialect"] = ""

    phoneme_per_speaker = load_phoneme_for_did(df)

    text_clf = load(MODEL_PATH_DID)
    text_clf['clf'].set_params(n_jobs=8)

    for speaker, dialects in phoneme_per_speaker.items():
        dial_tags = list(dialects.keys())
        phonemes = list(dialects.values())
        predicted = text_clf.predict(phonemes)

        for idx, result in enumerate(predicted):
            did = PHON_DID_CLS[result]

            # Assign prediction to all matching rows in the original df
            mask = (df["speaker"] == speaker) & (df["dialect"] == dial_tags[idx])
            df.loc[mask, "pred_dialect"] = did

    save_path = os.path.join(model_path, "generated_speech", "transcribed_metadata.csv")
    save_to_csv(df, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "transcribed_metadata.csv"))


def _setup_speaker_sim_model():
    print("Loading speaker sim model ecapa2...")
    # automatically checks for cached file, optionally set `cache_dir` location
    model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=MODEL_PATH)
    ecapa2 = torch.jit.load(model_file, map_location=device)
    ecapa2.half()
    return ecapa2


def calculate_conditioning_embeddings(unique_speakers: str, ecapa2) -> tuple[dict, dict]:
    speaker_to_embedding = {}
    speaker_to_avg_similarity = {}
    for speaker in unique_speakers:
        conditioning_paths = speaker_wavs[speaker]

        ref_embeddings = []
        for cp in conditioning_paths:
            waveform, _ = torchaudio.load(cp)
            embedding = ecapa2(waveform.to(device))
            ref_embeddings.append(embedding.cpu().numpy())

        similarity_matrix = cosine_similarity(np.array(ref_embeddings).squeeze())
        div = len(similarity_matrix) * (len(similarity_matrix) - 1) / 2
        avg_similarity = np.triu(similarity_matrix, k=1).sum() / div
        speaker_to_avg_similarity[speaker] = avg_similarity
        ref_embeddings_avg = np.vstack(ref_embeddings).squeeze().mean(axis=0)

        speaker_to_embedding[speaker] = ref_embeddings_avg

    return speaker_to_embedding, speaker_to_avg_similarity


def calculate_speaker_similarity(model_path: str) -> None:
    print(f"Starting Speaker similarity calculation for {model_path}.")
    df = load_transcribed_metadata(model_path)

    if "similarity" in df.columns or "rel_sim" in df.columns:
        print("Already calculated speaker sim, skipping step...")
        return

    unique_speakers = df["speaker"].unique()

    ecapa2 = _setup_speaker_sim_model()

    print(f"Calculating speaker embeddings for conditioning samples for {model_path}.")
    speaker_to_embedding, speaker_to_avg_similarity = calculate_conditioning_embeddings(unique_speakers, ecapa2=ecapa2)

    print(f"Calculating speaker embeddings for generated samples for {model_path}.")
    similarities, rel_sims = [], []
    opath_model = os.path.join(model_path, "generated_speech")
    for sid, speaker in enumerate(unique_speakers):
        mask = df["speaker"] == speaker
        speaker_df = df[mask]

        ref_embeddings_avg = speaker_to_embedding[speaker]

        for idx, row in speaker_df.iterrows():
            audio_file = os.path.join(opath_model, row["file_path"])
            waveform, _ = torchaudio.load(audio_file)
            sample_embedding = ecapa2(waveform.to(device)).squeeze()
            similarity = float(
                torch.nn.functional.cosine_similarity(torch.tensor(ref_embeddings_avg, device=device)[None, :],
                                                      sample_embedding))
            rel_sim = similarity / speaker_to_avg_similarity[speaker]

            similarities.append(float(similarity))
            rel_sims.append(float(rel_sim))

    df["similarity"] = similarities
    df["rel_sim"] = rel_sims

    save_path = os.path.join(model_path, "generated_speech", "transcribed_metadata.csv")
    save_to_csv(df, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "transcribed_metadata.csv"))


def run_eval(model: str) -> None:
    evaluate_did(model)
    evaluate_de_text(model)
    evaluate_speaker_similarity(model)


def evaluate_did(model_path: str) -> None:
    print(f"Starting DID evaluation for {model_path}.")
    df = load_transcribed_metadata(model_path)

    references = list(df["dialect"])
    reference_classes = [PHON_DID_CLS_INV[ref] for ref in references]
    hypothesis = list(df["pred_dialect"])
    hypothesis_classes = [PHON_DID_CLS_INV[hypo] for hypo in hypothesis]

    # Create a DataFrame for easy manipulation
    data = pd.DataFrame({'Reference': references, 'Hypothesis': hypothesis})

    # Count matches and mismatches
    data['Match'] = data['Reference'] == data['Hypothesis']
    match_count = data['Match'].value_counts()

    # Compute F1 scores per dialect
    f1_scores_per_dialect = classification_report(references, hypothesis, output_dict=True)

    print("\nF1 scores per dialect:")
    dialect_region_f1 = []
    for dialect in set(references):
        if dialect in f1_scores_per_dialect:
            print(f"{dialect}: {f1_scores_per_dialect[dialect]['precision']:.4f} & "
                  f"{f1_scores_per_dialect[dialect]['recall']:.4f} & "
                  f"{f1_scores_per_dialect[dialect]['f1-score']:.4f}")
            dialect_region_f1.append({
                "dialect": dialect,
                "precision": f1_scores_per_dialect[dialect]['precision'],
                "recall": f1_scores_per_dialect[dialect]['recall'],
                "f1-score": f1_scores_per_dialect[dialect]['f1-score']
            })

    df_did_dialect_region = pd.DataFrame(dialect_region_f1)
    save_path = os.path.join(model_path, "generated_speech", "did_f1_regions.csv")
    save_to_csv(df_did_dialect_region, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "did_f1_regions.csv"))

    f1_macro = f1_score(reference_classes, hypothesis_classes, average='macro')  # Treat all classes equally
    f1_micro = f1_score(reference_classes, hypothesis_classes, average='micro')  # Aggregate globally
    f1_weighted = f1_score(reference_classes, hypothesis_classes, average='weighted')  # Weight by support

    df_did_f1_overall = {
        "macro_f1": [f1_macro],
        "micro_f1": [f1_micro],
        "weighted_f1": [f1_weighted],
        "match_true": [match_count.get(True, 0)],
        "match_false": [match_count.get(False, 0)]
    }
    df_did_overall = pd.DataFrame(df_did_f1_overall)
    save_path = os.path.join(model_path, "generated_speech", "did_f1_overall.csv")
    save_to_csv(df_did_overall, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "did_f1_overall.csv"))

    print(
        f"Macro F1: {f1_macro:.4f}",
        f"Micro F1: {f1_micro:.4f}",
        f"Weighted F1: {f1_weighted:.4f}",
        f"Match Count:\n{match_count.to_string(index=True)}"
    )

    # Confusion Matrix
    conf_matrix = confusion_matrix(references, hypothesis, labels=list(set(references + hypothesis)))

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(set(references + hypothesis))).plot(
        cmap='magma', colorbar=True)
    plt.title("Confusion Matrix of Dialects")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_eval_path, "did_confusion_matrix.png"))
    plt.show()


def calculate_scores(comparison: pd.DataFrame) -> pd.DataFrame:
    scores = {
        "wer": [],
        "wer_lower": [],
        "mer": [],
        "mer_lower": [],
        "wil": [],
        "wil_lower": [],
        "cer": [],
        "cer_lower": [],
        "bert_score": [],
    }

    print("Starting WER, MER, etc. calculations...")
    for idx, row in comparison.iterrows():
        ref, hypo = row["text"], row["gen_text"]
        ref_low, hypo_low = ref.lower(), hypo.lower()

        out = jiwer.process_words(ref, hypo)
        out_low = jiwer.process_words(ref_low, hypo_low)

        scores["wer"].append(out.wer)
        scores["wer_lower"].append(out_low.wer)
        scores["mer"].append(out.mer)
        scores["mer_lower"].append(out_low.mer)
        scores["wil"].append(out.wil)
        scores["wil_lower"].append(out_low.wil)

        scores["cer"].append(jiwer.process_characters(ref, hypo).cer)
        scores["cer_lower"].append(jiwer.process_characters(ref_low, hypo_low).cer)

    print("Starting BERTScore calculations...")
    P, R, F1 = bert_score(
        comparison["gen_text"].tolist(),
        comparison["text"].tolist(),
        lang="de",
        batch_size=64,
        verbose=False,
        device=device
    )
    scores["bert_score"] = F1.cpu().numpy().tolist()

    # Calculate BLEU Score
    print("Starting BLEU calculations...")
    reference_split = [ref.split(" ") for ref in comparison["text"]]
    hypothesis_split = [hyp.split(" ") for hyp in comparison["gen_text"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    scores["bleu_score"] = bleu_scores

    reference_split = [ref.lower().split(" ") for ref in comparison["text"]]
    hypothesis_split = [hyp.lower().split(" ") for hyp in comparison["gen_text"]]
    bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(reference_split, hypothesis_split)]
    scores["bleu_score_lower"] = bleu_scores

    # Combine original DataFrame with scores
    score_df = pd.DataFrame(scores)
    return pd.concat([comparison.reset_index(drop=True), score_df], axis=1)


def evaluate_de_text(model_path: str) -> None:
    if "de_text_calc.csv" in os.listdir(save_eval_path):
        print(f"De text eval already done, skipping {model_path}...")
        return

    print(f"Starting score calculation for DE-text for {model_path}.")
    df = load_transcribed_metadata(model_path)
    de_text_df = df[["speaker", "orig_dialect", "dialect", "tid", "text", "gen_text"]].copy()
    calc_de_text_df = calculate_scores(de_text_df)

    save_path = os.path.join(model_path, "generated_speech", "de_text_calc.csv")
    save_to_csv(calc_de_text_df, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "de_text_calc.csv"))


def evaluate_speaker_similarity(model_path: str) -> None:
    print(f"Starting average speaker similarity calculation for {model_path}")
    df = load_transcribed_metadata(model_path)

    results = []
    for dialect_1 in LANG_MAP.values():
        for dialect_2 in LANG_MAP.values():
            # only use those lines where dial_tag equals d2_tag and orig_dial_tag equals dialect_1
            dial_df = df[(df["orig_dialect"] == dialect_1) & (df["dialect"] == dialect_2)]

            if len(dial_df) == 0:
                print(f"No data for {dialect_1, dialect_2}")
                continue

            similarities = dial_df["similarity"].tolist()
            avg_similarity = sum(similarities) / len(similarities)

            rel_similarities = dial_df["rel_sim"].tolist()
            avg_rel_similarity = sum(rel_similarities) / len(rel_similarities)

            results.append({
                "orig_dialect": dialect_1,
                "dialect": dialect_2,
                "avg_similarity": avg_similarity,
                "avg_rel_similarity": avg_rel_similarity
            })

    # compute overall metrics
    similarities = df["similarity"].tolist()
    rel_similarities = df["rel_sim"].tolist()
    avg_similarity = sum(similarities) / len(similarities)
    avg_rel_similarity = sum(rel_similarities) / len(rel_similarities)

    print("Overall:")
    print(f"avg_similarity: {avg_similarity:.4f}")
    print(f"avg_rel_similarity: {avg_rel_similarity:.4f}")

    results.append({
        "orig_dialect": "Overall",
        "dialect": "Overall",
        "avg_similarity": avg_similarity,
        "avg_rel_similarity": avg_rel_similarity
    })

    sim_df = pd.DataFrame(results)
    save_path = os.path.join(model_path, "generated_speech", "speaker_similarity.csv")
    save_to_csv(sim_df, save_path)
    shutil.copyfile(save_path, os.path.join(save_eval_path, "speaker_similarity.csv"))


if __name__ == "__main__":
    device, torch_dtype = setup_gpu_device()

    # Define inference texts
    text_file = os.path.join(CLUSTER_PROJECTS_PATH, "snf_eval_condition_files", "50_inference_text_samples.csv")
    texts = list(pd.read_csv(text_file, sep=";", encoding="utf-8")["text"])
    assert len(texts) == 50, f"Loaded less than 50 samples for inference: {len(texts)}"
    # texts = [f"Das ist ein Beispielsatz, welcher auf Schweizerdeutsch ausgesprochen werden soll"] * 2

    # Copy speaker conditioning samples and setup lookup structure for inference
    shutil.copytree(SOURCE_SPEAKER_DIRECTORY, INFERENCE_SPEAKER_DIRECTORY, dirs_exist_ok=True)
    speaker_wavs, speaker_to_dialect = collect_speaker_condition_samples()

    # Get all SwissGPC checkpoints used in evaluation
    swissgpc_filter = "SwissGPC_epoch_5"
    # swissgpc_filter = "SwissGPC_epoch_1_subset_0"

    list_of_directories = [os.path.join(MODEL_CHECKPOINTS_PATH, model_dir) for model_dir in
                           os.listdir(MODEL_CHECKPOINTS_PATH) if swissgpc_filter in model_dir]

    try:
        set_start_method("spawn")  # Important due to cuda not being able to fork processes
    except RuntimeError:
        print("Experienced issue on setting start method from fork to spawn...")
        pass  # Start method already set (usually when re-running in interactive environments)

    for directory in list_of_directories:
        assert_checkpoint_folder_contains_model(directory)

        folder_name = os.path.basename(os.path.normpath(directory))
        save_eval_path = os.path.join(CLUSTER_PROJECTS_PATH, "generated_speech", folder_name)
        os.makedirs(save_eval_path, exist_ok=True)

        if "speaker_similarity.csv" in os.listdir(save_eval_path):
            print(f"Evaluation already done, skipping {folder_name}...")
            continue

        # Setup folder structure for specific checkpoint
        model_path = os.path.join(OUT_PATH, folder_name)
        os.makedirs(model_path, exist_ok=True)

        print("Starting inference")
        run_inference(model_path)

        # shutil.copyfile(os.path.join(save_eval_path, "generated_speech.tar.gz"), os.path.join(model_path, "generated_speech.tar.gz"))

        # # Extract the tar.gz file
        # with tarfile.open(os.path.join(model_path, "generated_speech.tar.gz"), "r:gz") as tar:
        #     tar.extractall(path=model_path)
        #
        # shutil.copyfile(os.path.join(save_eval_path, "metadata.csv"),
        #                 os.path.join(model_path, "generated_speech", "metadata.csv"))


        # if os.path.exists(os.path.join(save_eval_path, "transcribed_metadata.csv")):
        #     shutil.copyfile(os.path.join(save_eval_path, "transcribed_metadata.csv"), os.path.join(model_path, "generated_speech", "transcribed_metadata.csv"))

        print("Starting transcription")
        run_transcription(model_path)

        # shutil.copyfile(os.path.join(save_eval_path, "transcribed_metadata.csv"), os.path.join(model_path, "generated_speech", "transcribed_metadata.csv"))

        print("Starting evaluation")
        run_eval(model_path)

        # Cleanup scratch
        print("Deleting generated speech from scratch folder...")
        shutil.rmtree(model_path)
