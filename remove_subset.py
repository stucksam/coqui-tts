import os
import sys

DATASETS_PATH = "/scratch/subsets"

CLUSTER_PROJECTS_PATH = "/cluster/projects/"
CLUSTER_PROJECTS_TTS = os.path.join(CLUSTER_PROJECTS_PATH, "TTS-Swiss-German")
TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_subsets")
# TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "test_audio_subsets")  # small subset test

LANG_MAP = {
    'ch_be': 'Bern',
    'ch_bs': 'Basel',
    'ch_gr': 'Graubünden',
    'ch_in': 'Innerschweiz',
    'ch_os': 'Ostschweiz',
    'ch_vs': 'Wallis',
    'ch_zh': 'Zürich',
    "de": "Deutschland"
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}


def remove_previous_subset(subset_to_remove: int) -> None:
    print(f"Deleting subset {subset_to_remove} to scratch")
    os.remove(os.path.join(DATASETS_PATH, f"subset_{subset_to_remove}.hdf5"))
    os.remove(os.path.join(DATASETS_PATH, f"subset_{subset_to_remove}.txt"))
    for dialect in LANG_MAP_INV.keys():
        dialect_path = os.path.join(DATASETS_PATH, f"{dialect}_{subset_to_remove}.txt")
        if os.path.exists(dialect_path):
            os.remove(dialect_path)
            print(f"Removed subset dialect {dialect} of subset {subset_to_remove} from scratch")


if __name__ == "__main__":
    subset = int(sys.argv[1])
    remove_previous_subset(subset)
