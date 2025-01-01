import os
import random
import shutil

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
SPEAKER_DIRECTORY = f"{CLUSTER_HOME_PATH}/_speakers"
OUT_PATH = "/scratch/ch_test_n"
GENERATED_SPEECH_FOLDER = "generated_speech"
TEXT_METADATA_FILE = "texts.txt"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "GPT_XTTS_v2.0_LJSpeech_FT-December-10-2024_06+18PM-5027233c"

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
test_meta = "test_sentences.csv"
test_meta_path = f"{CLUSTER_HOME_PATH}/coqui-tts/recipes/ljspeech/xtts_v2/{test_meta}"
with open(test_meta_path, 'rt', encoding='utf-8') as f:
    next(f)
    for line in f:
        sline = line.split(";")
        text = sline[0]
        all_texts.add(text)

all_texts = list(all_texts)

texts = random.sample(all_texts, k=100)

speaker_wavs = f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1087.wav"
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1011.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1042.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1043.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1054.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1056.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1066.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/gülsha/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1077.wav",

    # f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1088.wav",
    # f"{CLUSTER_HOME_PATH}/_speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1173.wav"
    # speaker reference to be used in training test sentences -> condition with wav length in GPTArgs


speaker_wavs = {
    "ch_bs": [
        f"{SPEAKER_DIRECTORY}/ch_bs/db5567660d87860ffd470d8e04283afdd67727cd2fd2fb7ed31695d13deb341e.wav",
        f"{SPEAKER_DIRECTORY}/ch_bs/e1267f316726e5470ef8c42b1df9810b8a7853328e13f362fed2193b953c81a2.wav",
        f"{SPEAKER_DIRECTORY}/ch_bs/ed91b514080136e7ca6661531a07d9edb72b3dc17c523919f97db3db98158e49.wav"
    ],
    "ch_be": [
        f"{SPEAKER_DIRECTORY}/ch_be/9afc07ee70337a7a028d70683f01e6e38ff9125495438981ac348e830f9197bb.wav",
        f"{SPEAKER_DIRECTORY}/ch_be/9fb0575ccf3ae1b8dc5ff4b651db3b810aaa1f5c945a529acc9f141b46e2e569.wav",
        f"{SPEAKER_DIRECTORY}/ch_be/606b1bb01fb00493496a9e6e8b6d13a39396ea12c0555b8150337b427eeae95f.wav"
    ],
    "ch_gr": [
        f"{SPEAKER_DIRECTORY}/ch_gr/f7c084e9a97ba5d58ac845136a6a06b66a8574c13ab35f6019b41c30f77d30b2.wav",
        f"{SPEAKER_DIRECTORY}/ch_gr/fe746b4c0e3605e27a4bbf224efd9d5cba1dcbed457ad697eb5a54f298004358.wav",
        f"{SPEAKER_DIRECTORY}/ch_gr/db1c24a39986aec746289d0fd00c88dc464cc2897d5e8d2ac3172b0459731f19.wav"
    ],
    "ch_in": [
        f"{SPEAKER_DIRECTORY}/ch_in/e81e9125f7e540c64cbfe53eae69e04be53c366935882e8c5fe23e3b1ac403e7.wav",
        f"{SPEAKER_DIRECTORY}/ch_in/fd687162210b39a5b4fefd37dafbce039643bc90e7a1f86f344c1c675af68534.wav",
        f"{SPEAKER_DIRECTORY}/ch_in/ffa1c65d5ac160bfd30f8c65b985bb6e6b64909351aee9c6cb79e3efdd931ade.wav"
    ],
    "ch_os": [
        f"{SPEAKER_DIRECTORY}/ch_os/f074041baa63c2eb4987e1d33aae4ba3a298b5ab7ea8ba22a40a18fac724123b.wav",
        f"{SPEAKER_DIRECTORY}/ch_os/c4fab31bec5f16b1dd466a40c3873791fb27d6298030cb971c97646e0e8864a1.wav",
        f"{SPEAKER_DIRECTORY}/ch_os/9f18f947d8d6292bfc712bf0e3b7af5c5e07441093b098037c94fe796865c7e7.wav"
    ],
    "ch_vs": [
        f"{SPEAKER_DIRECTORY}/ch_vs/855b987f227db1d0141ff2ba14d095596641f4271714fd169f45f2dbed90b9bf.wav",
        f"{SPEAKER_DIRECTORY}/ch_vs/80402c841990b021efcdf9f1c1a5f1604b457afe408354dc5f57733b67c3a184.wav",
        f"{SPEAKER_DIRECTORY}/ch_vs/82f30811886565d3e007cb478068d4c5f0dab00cdf08d16968cff7b7c981987c.wav"
    ],
    "ch_zh": [
        f"{SPEAKER_DIRECTORY}/ch_zh/c85640d7841a25537605847ddc4c9ea8477ceddbbb06ad6060b529506a1567fe.wav",
        f"{SPEAKER_DIRECTORY}/ch_zh/e61779ae1c8f24ed49bf88554c1c45258e28792df591bc472cf3fad9c9df5892.wav",
        f"{SPEAKER_DIRECTORY}/ch_zh/f444d9c85fcda075d137d54152129c3508a0458509bcc5bf9bf88947ab5d59d3.wav"
    ],
    "de": [
        f"{SPEAKER_DIRECTORY}/de/f074041baa63c2eb4987e1d33aae4ba3a298b5ab7ea8ba22a40a18fac724123b.wav",
        f"{SPEAKER_DIRECTORY}/de/c4fab31bec5f16b1dd466a40c3873791fb27d6298030cb971c97646e0e8864a1.wav",
        f"{SPEAKER_DIRECTORY}/de/9f18f947d8d6292bfc712bf0e3b7af5c5e07441093b098037c94fe796865c7e7.wav"
    ]
}

dial_tags = list(LANG_MAP.keys())
os.makedirs(f"{OUT_PATH}/{model_name}/{GENERATED_SPEECH_FOLDER}", exist_ok=True)

for tid, text in enumerate(texts):
    for dial_tag in dial_tags:
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs[dial_tag], language=dial_tag,
                        file_path=f"{OUT_PATH}/{model_name}/{GENERATED_SPEECH_FOLDER}/{tid}_{LANG_MAP[dial_tag]}.wav")

with open(os.path.join(OUT_PATH, model_name, TEXT_METADATA_FILE), "wt", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        f.write(f"{idx}\t{text}\n")

shutil.copytree(os.path.join(OUT_PATH, model_name, GENERATED_SPEECH_FOLDER), os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, model_name, GENERATED_SPEECH_FOLDER), dirs_exist_ok=True)
shutil.copy2(os.path.join(OUT_PATH, model_name, TEXT_METADATA_FILE), os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, model_name, TEXT_METADATA_FILE))
