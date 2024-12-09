import logging
import os

import librosa
import torch
from joblib import load
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2Processor, Pipeline, Wav2Vec2ForCTC

from recipes.ljspeech.xtts_v2.xtts_data_point import XTTSDataPoint

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
CLUSTER_HOME_PATH = "/cluster/home/stucksam"
# CLUSTER_HOME_PATH = "/home/ubuntu/ma/"
OUT_PATH = "ch_test_n"

MODEL_PATH = f"{CLUSTER_HOME_PATH}/swiss-phonemes-tts/src/model"
MODEL_PATH_DE_CH = f"{MODEL_PATH}/de_to_ch_large_2"
MODEL_PATH_DID = f"{MODEL_PATH}/text_clf_3_ch_de.joblib"
MODEL_AUDIO_PHONEME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_WHISPER_v3 = "openai/whisper-large-v3"
MODEL_T5_TOKENIZER = "google/t5-v1_1-large"

XTTS_MODEL_TRAINED = "GPT_XTTS_v2.0_LJSpeech_FT-December-04-2024_08+54PM-6202f2fe"
GENERATE_SPEECH_PATH = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/generated_speech"

MISSING_TEXT = "NO_TEXT"
MISSING_PHONEME = "NO_PHONEME"
BATCH_SIZE = 32

phon_did_cls = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Deutschland"}

logger = logging.getLogger(__name__)


def _setup_gpu_device() -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Training / Inference device is: {device}")
    return device, torch_dtype


def load_transcribed_metadata() -> list[XTTSDataPoint]:
    metadata = []
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, "transcribed.txt"), "rt", encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            metadata.append(XTTSDataPoint.load_single_datapoint(split_line))
    return metadata


def write_to_file(content: list[XTTSDataPoint]) -> None:
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, "transcribed.txt"), "wt", encoding="utf-8") as f:
        for line in content:
            f.write(line.to_string())


def load_orig_texts() -> list[str]:
    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, "texts.txt"), "rt", encoding="utf-8") as f:
        texts = [text.split("\t")[1] for text in f]
    return texts


def get_filenames_of_wavs():
    return [file for file in os.listdir(f"{OUT_PATH}/{XTTS_MODEL_TRAINED}") if file.endswith(".wav")]


def _setup_german_transcription_model():
    device, torch_dtype = _setup_gpu_device()
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
    logger.info("Transcribing to WAV to German")
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
        results = pipe(audio_batch, batch_size=BATCH_SIZE)
        for idx, text in enumerate(results):
            filename = wav_files[start_idx + idx]
            dialect = filename.split("_")[0]
            text_id = int(filename.split("_")[1].replace(".wav", ""))
            generated_samples.append(XTTSDataPoint(filename, text_id, dialect, text))

    write_to_file(generated_samples)


def _setup_phoneme_model() -> Pipeline:
    device, torch_dtype = _setup_gpu_device()

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
    logger.info("Transcribing WAV to phoneme.")

    pipe = _setup_phoneme_model()

    metadata = load_transcribed_metadata()
    num_samples = len(metadata)

    for start_idx in range(0, num_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        # Load batch of audio data
        audio_batch = []
        for i in range(start_idx, end_idx):
            audio_data, _ = librosa.load(f"{GENERATE_SPEECH_PATH}/{metadata[i].sample_name[i]}.wav", sr=None)
            audio_batch.append(audio_data)

        # Run phoneme transcription
        results = pipe(audio_batch, batch_size=BATCH_SIZE)

        # Save results to collection
        for idx, result in enumerate(results):
            phoneme = result["text"].strip()
            if phoneme == "":
                logger.error(
                    f"NO PHONEME TRANSCRIPT GENERATED FOR {metadata[start_idx + idx].sample_name}")
                phoneme = MISSING_PHONEME
            metadata[start_idx + idx].gen_phoneme = phoneme
            logger.info(f"NAME: {metadata[start_idx + idx].sample_name}, PHON: {phoneme}")

    write_to_file(metadata)


def dialect_identification_naive_bayes_majority_voting() -> None:
    logger.info("Run Dialect Identification based on phonemes")
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
            logger.info(f"NAME: {metadata[start_idx + idx].sample_name}, DID: {did}")

    write_to_file(metadata)


if __name__ == "__main__":
    pass
