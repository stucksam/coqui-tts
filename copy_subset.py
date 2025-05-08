import os
import shutil
import sys

DATASETS_PATH = "/scratch/subsets"

CLUSTER_PROJECTS_PATH = "/cluster/projects/"
CLUSTER_PROJECTS_TTS = os.path.join(CLUSTER_PROJECTS_PATH, "TTS-Swiss-German")
TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_subsets")
# TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "test_audio_subsets")  # small subset test


def copy_subset_to_scratch(subset_to_copy: int) -> None:
    print(f"Copying subset {subset_to_copy} to scratch")
    shutil.copy2(os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{subset_to_copy}.hdf5"), DATASETS_PATH)
    shutil.copy2(os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{subset_to_copy}.txt"), DATASETS_PATH)
    print(f"Successfully copied subset {subset_to_copy} to scratch")


if __name__ == "__main__":
    subset = int(sys.argv[1])
    copy_subset_to_scratch(subset)
