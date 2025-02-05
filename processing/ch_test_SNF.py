import os
import shutil

import pandas as pd
import torch

from TTS.api import TTS
from util import SPEAKER_DIRECTORY, CLUSTER_HOME_PATH, OUT_PATH, XTTS_MODEL_TRAINED, GENERATED_SPEECH_PATH, \
    DID_REF_PATH, LANG_MAP

GENERATED_SPEECH_FOLDER = "generated_speech"
DID_REF_FOLDER = "did_speech"
TEXT_METADATA_FILE = "texts.txt"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = f"{CLUSTER_HOME_PATH}/checkpoint/{XTTS_MODEL_TRAINED}/"

config_path = model_path + "config.json"

all_texts = set()
test_meta = "SNF_Test_Sentences.xlsx"
test_meta_path = f"processing/data/{test_meta}"
# test_meta_path = f"data/{test_meta}"
df = pd.read_excel(test_meta_path, sheet_name="SNF Test Samples")
for index, row in df.iterrows():
    all_texts.add(row["sentence"])

texts = list(all_texts)

# Init TTS
tts = TTS(
    model_path=model_path,
    config_path=config_path,
    progress_bar=True
).to(device)
# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
# texts = random.sample(all_texts, k=100)


dial_tags = list(LANG_MAP.keys())

speaker_wavs = {}
for dial_tag in dial_tags:
    speaker_wavs[dial_tag] = {}
    ref_path = f"{SPEAKER_DIRECTORY}/{dial_tag}/references"
    refs = os.listdir(ref_path)
    for ref in refs:
        wav_files = os.listdir(f"{ref_path}/{ref}")
        speaker_wavs[dial_tag][ref] = [f"{ref_path}/{ref}/{wav}" for wav in wav_files]

os.makedirs(GENERATED_SPEECH_PATH, exist_ok=True)
os.makedirs(DID_REF_PATH, exist_ok=True)

print(f"Starting TTS inference for model {XTTS_MODEL_TRAINED}")
for tid, text in enumerate(texts):
    for dial_tag in dial_tags:
        filtered_df = df[(df["dialect_region"] == LANG_MAP[dial_tag]) & (df["sentence"] == text)]
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs[dial_tag][filtered_df['client_id'].iloc[0]],
                        language=dial_tag,
                        file_path=f"{GENERATED_SPEECH_PATH}/{tid}_{LANG_MAP[dial_tag]}.wav", split_sentences=False)

# needed longer segments per speaker for DID
for dial_tag in dial_tags:
    for speaker, ref_wav in speaker_wavs[dial_tag].items():
        os.makedirs(f"{DID_REF_PATH}/{speaker}", exist_ok=True)
        for tid, text in enumerate(texts):
            tts.tts_to_file(text=text, speaker_wav=ref_wav, language=dial_tag,
                            file_path=f"{DID_REF_PATH}/{speaker}/{tid}_{LANG_MAP[dial_tag]}.wav", split_sentences=False)

with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE), "wt", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        dialect_text = text.split(". ")[0] + "."
        text_id = df[df["sentence"] == text]["sentence_id"].iloc[0]
        f.write(f"{idx}\t{text}\t{text_id}\n")

shutil.copytree(GENERATED_SPEECH_PATH,
                os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, GENERATED_SPEECH_FOLDER),
                dirs_exist_ok=True)
shutil.copytree(DID_REF_PATH,
                os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, DID_REF_FOLDER),
                dirs_exist_ok=True)
shutil.copy2(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE),
             os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE))
