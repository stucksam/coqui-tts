import os
import random
import shutil

import torch

from TTS.api import TTS
from util import CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, LANG_MAP, OUT_PATH, \
    GENERATED_SPEECH_PATH_LONG, TEXT_METADATA_FILE_LONG, GENERATED_SPEECH_FOLDER_LONG

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = f"{CLUSTER_HOME_PATH}/checkpoint/{XTTS_MODEL_TRAINED}/"
config_path = model_path + "config.json"
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

all_texts = set()
test_meta = "test_sentences.csv"
test_meta_path = f"processing/data/{test_meta}"
with open(test_meta_path, 'rt', encoding='utf-8') as f:
    next(f)
    for line in f:
        sline = line.split(";")
        text = sline[0]
        all_texts.add(text)

all_texts = list(all_texts)
texts = random.sample(all_texts, k=100)

# speaker_wavs = f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1011.wav"
speaker_wavs = [
    f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1011.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1042.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1077.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1088.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1173.wav"
    # f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1088.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1173.wav"
    # speaker reference to be used in training test sentences -> condition with wav length in GPTArgs
]

dial_tags = list(LANG_MAP.keys())
os.makedirs(GENERATED_SPEECH_PATH_LONG, exist_ok=True)

for tid, text in enumerate(texts):
    for dial_tag in dial_tags:
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=dial_tag,
                        file_path=f"{GENERATED_SPEECH_PATH_LONG}/{tid}_{LANG_MAP[dial_tag]}.wav", split_sentences=False)

with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE_LONG), "wt", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        f.write(f"{idx}\t{text}\tChatGPT\n")

shutil.copytree(GENERATED_SPEECH_PATH_LONG, os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED,
                                                         GENERATED_SPEECH_FOLDER_LONG), dirs_exist_ok=True)
shutil.copy2(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE_LONG),
             os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, TEXT_METADATA_FILE_LONG))
