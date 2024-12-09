import os
import random

import torch

from TTS.api import TTS

LANG_MAP = {
    'ch_be': 'Bern',
    'ch_bs': 'Basel',
    'ch_gr': 'Graubünden',
    'ch_in': 'Innerschweiz',
    'ch_os': 'Ostschweiz',
    'ch_vs': 'Wallis',
    'ch_zh': 'Zürich',
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}

CLUSTER_HOME_PATH = "/cluster/home/stucksam"
# CLUSTER_HOME_PATH = "/home/ubuntu/ma/"
OUT_PATH = "ch_test_n"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "GPT_XTTS_v2.0_LJSpeech_FT-December-04-2024_08+54PM-6202f2fe"

model_path = f"{CLUSTER_HOME_PATH}/coqui-tts/TTS/TTS_CH/trained/{model_name}/"
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
test_meta = "test_sentences.tsv"
with open(test_meta, 'rt', encoding='utf-8') as f:
    next(f)
    for line in f:
        sline = line.split('\t')
        text = sline[0]
        all_texts.add(text)

all_texts = list(all_texts)

texts = random.sample(all_texts, k=1)

speaker_wavs = [
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1011.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1042.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1043.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1054.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1056.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1066.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1077.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1087.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1088.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1173.wav"
    # speaker reference to be used in training test sentences -> condition with wav length in GPTArgs
]

dial_tags = list(LANG_MAP.keys())
[os.makedirs(f"{OUT_PATH}/{model_name}/generated_speech", exist_ok=True) for dial_tag in dial_tags]

for tid, text in enumerate(texts):
    for dial_tag in dial_tags:
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=dial_tag,
                        file_path=f"{OUT_PATH}/{model_name}/generated_speech/{dial_tag}_{tid}.wav")

with open(os.path.join(OUT_PATH, model_name, "texts.txt"), "wt", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        f.write(f"{idx}\t{text}")
