import torch

CLUSTER_HOME_PATH = "/cluster/home/stucksam"
# CLUSTER_HOME_PATH = "/home/ubuntu/ma/"
OUT_PATH = "/scratch/ch_test_n"
SPEAKER_DIRECTORY = f"{CLUSTER_HOME_PATH}/_speakers"

GENERATED_SPEECH_FOLDER = "generated_speech"
DID_REF_FOLDER = "did_speech"
TEXT_METADATA_FILE = "texts.txt"
TEXT_TRANSCRIBED_FILE = "transcribed.txt"

XTTS_MODEL_TRAINED = "GPT_XTTS_v2.0_Full_3_5_SNF"
GENERATED_SPEECH_PATH = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/{GENERATED_SPEECH_FOLDER}"
DID_REF_PATH = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/{DID_REF_FOLDER}"

GENERATED_SPEECH_FOLDER_LONG = GENERATED_SPEECH_FOLDER + "_long"
GENERATED_SPEECH_PATH_LONG = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/{GENERATED_SPEECH_FOLDER_LONG}"
TEXT_METADATA_FILE_LONG = TEXT_METADATA_FILE.replace(".txt", "_long.txt")
TEXT_TRANSCRIBED_FILE_LONG = TEXT_TRANSCRIBED_FILE.replace(".txt", "_long.txt")

SNF_LONG_REPLACEMENT = "_snf_long"
GENERATED_SPEECH_FOLDER_SNF_LONG = GENERATED_SPEECH_FOLDER + SNF_LONG_REPLACEMENT
GENERATED_SPEECH_PATH_SNF_LONG = f"{OUT_PATH}/{XTTS_MODEL_TRAINED}/{GENERATED_SPEECH_FOLDER_SNF_LONG}"
TEXT_METADATA_FILE_SNF_LONG = TEXT_METADATA_FILE.replace(".txt", f"{SNF_LONG_REPLACEMENT}.txt")
TEXT_TRANSCRIBED_FILE_SNF_LONG = TEXT_TRANSCRIBED_FILE.replace(".txt", f"{SNF_LONG_REPLACEMENT}.txt")

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

phon_did_cls = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Deutschland"}

phon_did_cls_inv = {v: k for k, v in phon_did_cls.items()}


class XTTSDataPoint:
    def __init__(
            self,
            sample_name: str,
            orig_text_id: int,
            dialect: str,
            gen_de_text: str,
            gen_phoneme: str = "",
            gen_dialect: str = "",
    ):
        self.sample_name = sample_name.replace(".wav", "")
        self.orig_text_id = orig_text_id
        self.dialect = dialect
        self.gen_de_text = gen_de_text
        self.gen_phoneme = gen_phoneme
        self.gen_dialect = gen_dialect

    @staticmethod
    def number_of_properties():
        return 6

    @staticmethod
    def load_single_datapoint(split_properties: list):
        return XTTSDataPoint(
            sample_name=split_properties[0],
            orig_text_id=int(split_properties[1]),
            dialect=split_properties[2],
            gen_de_text=split_properties[3],
            gen_phoneme=split_properties[4] if len(split_properties) > 4 else "",
            gen_dialect=split_properties[5] if len(split_properties) > 5 else ""
        )

    def to_string(self) -> str:
        to_string = f"{self.sample_name}\t{self.orig_text_id}\t{self.dialect}\t{self.gen_de_text}"
        if self.gen_phoneme:
            to_string += f"\t{self.gen_phoneme}"
        if self.gen_dialect:
            to_string += f"\t{self.gen_dialect}"

        to_string += "\n"
        return to_string


class ReferenceDatapoint:

    def __init__(self,
                 sample_id: int,
                 text: str,
                 snf_sample_id: str):
        self.sample_id = sample_id
        self.text = text
        self.snf_sample_id = snf_sample_id

    @staticmethod
    def load_single_ref_datapoint(split_properties: list):
        return ReferenceDatapoint(sample_id=int(split_properties[0]),
                                  text=split_properties[1],
                                  snf_sample_id=split_properties[2]
                                  )


def setup_gpu_device() -> tuple:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype


def load_transcribed_metadata(path: str) -> list[XTTSDataPoint]:
    metadata = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            metadata.append(XTTSDataPoint.load_single_datapoint(split_line))
    return metadata


def load_reference_sentences(path: str) -> list[ReferenceDatapoint]:
    references = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            split_line = line.replace('\n', '').split('\t')
            references.append(ReferenceDatapoint.load_single_ref_datapoint(split_line))
    return references
