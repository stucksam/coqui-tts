import random

from pydub import AudioSegment
import torch, os, csv
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
LANG_MAP_INV = {v:k for k,v in LANG_MAP.items()}

CLUSTER_HOME_PATH = "/cluster/home/stucksam"
DATASETS_PATH = "/scratch/dialects"
OUT_PATH = f"{CLUSTER_HOME_PATH}/coqui-tts/TTS/TTS_CH/trained"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "GPT_XTTS_v2.0_LJSpeech_FT-November-11-2024_09+32PM-605fcc21"

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
#wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file

all_texts = set()
test_meta = '/02Datasets/02 Audio Processing/snf_test/test.tsv'
with open(test_meta, 'rt', encoding='utf-8') as f:
    next(f)
    for line in f:
        sline = line.split('\t')
        text = sline[2]
        all_texts.add(text)

all_texts = list(all_texts)

texts = random.sample(all_texts, k=30)

speaker_id = 'd2dee463-0eb9-47fa-b739-f1dccd8638f9'

speaker_wavs = [
    f"{CLUSTER_HOME_PATH}/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1199.wav",
    f"{CLUSTER_HOME_PATH}/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1191.wav",
    f"{CLUSTER_HOME_PATH}/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1095.wav",

    # speaker reference to be used in training test sentences
]

#speaker_wavs = [os.path.join(speaker_wavs_base, speaker_id, x) for x in audios]

dial_tags = list(LANG_MAP.keys())
[os.makedirs(f"ch_test_n/{model_name}/{dial_tag}", exist_ok=True) for dial_tag in dial_tags]

for tid, text in enumerate(texts):
    for dial_tag in dial_tags:
        tts.tts_to_file(text=text, speaker_wav=speaker_wavs, language=dial_tag, file_path=f'ch_test_n/{model_name}/{dial_tag}/{tid}.wav')


