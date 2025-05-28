import os
import random
import shutil

import pandas as pd
import torch

from TTS.api import TTS
from util import CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, LANG_MAP, LANG_MAP_INV, OUT_PATH, \
    GENERATED_SPEECH_PATH_LONG, TEXT_METADATA_FILE_LONG, GENERATED_SPEECH_FOLDER_LONG, SPEAKER_DIRECTORY, \
    GENERATED_SPEECH_PATH

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

all_texts = set()
test_meta = "/cluster/home/stucksam/generated_speech/GPT_XTTS_v2.0_Full_3_5_SNF/texts_long_innerschweiz.txt"
with open(test_meta, 'rt', encoding='utf-8') as f:
    for line in f:
        sline = line.split("\t")
        text = sline[1]
        all_texts.add(text)

all_texts = list(all_texts)
texts = all_texts

speaker_wavs = [
    f"{CLUSTER_HOME_PATH}/_speakers/ch_in/references/acf67674-c912-42c0-ba3c-f85e2db965ac/3dbf17ad654cebe79a3560a922a04ad10f1880083660d854e982e10c26e15cb2.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/ch_in/references/acf67674-c912-42c0-ba3c-f85e2db965ac/4135a8bf6982d72002bd407843732d8aa302984fd30a02ca247da6189a4cf596.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/ch_in/references/acf67674-c912-42c0-ba3c-f85e2db965ac/57728cc1728807a8f0e38534f045aa052292ec52983ef9ea000d984edf981660.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/ch_in/references/acf67674-c912-42c0-ba3c-f85e2db965ac/edc6b3f75b2a0f8465220aadd50c73a438ac0b62c59f4cf3c35c35eca6a414fb.wav",
    f"{CLUSTER_HOME_PATH}/_speakers/ch_in/references/acf67674-c912-42c0-ba3c-f85e2db965ac/eef21c41f601ebf5669183efea0cd16d5ab07658f4b7f86b1ad798fafffbc515.wav"
]

# # Set to store unique shuffled lists
# unique_shuffled_lists = set()
#
# # Generate 7 unique shuffled lists
# while len(unique_shuffled_lists) < 7:
#     shuffled_list = speaker_wavs[:]
#     random.shuffle(shuffled_list)
#     unique_shuffled_lists.add(tuple(shuffled_list))  # Add as a tuple to ensure uniqueness in the set
#
# # Convert the set back to a list of lists
# final_list = [list(item) for item in unique_shuffled_lists]
#
# # Display the results
# print("7 Unique Shuffled Lists:")
# for idx, sublist in enumerate(final_list, 1):
#     print(f"List {idx}: {sublist}")

dial_tags = list(LANG_MAP.keys())

for i in range(7):
    folder = GENERATED_SPEECH_PATH_LONG + f"_{i}"
    text_file = TEXT_METADATA_FILE_LONG.replace(".txt", f"_{i}.txt")
    os.makedirs(folder, exist_ok=True)
    for tid, text in enumerate(texts):
        for dial_tag in dial_tags:
            tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=dial_tag,
                            file_path=f"{folder}/{tid}_{LANG_MAP[dial_tag]}.wav", split_sentences=False)

    with open(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, text_file), "wt", encoding="utf-8") as f:
        for idx, text in enumerate(texts):
            f.write(f"{idx}\t{text}\tacf67674-c912-42c0-ba3c-f85e2db965ac\n")

    shutil.copytree(folder, os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED,
                                                             f"{GENERATED_SPEECH_FOLDER_LONG}_{i}"), dirs_exist_ok=True)
    shutil.copy2(os.path.join(OUT_PATH, XTTS_MODEL_TRAINED, text_file),
                 os.path.join(CLUSTER_HOME_PATH, GENERATED_SPEECH_FOLDER, XTTS_MODEL_TRAINED, text_file))
