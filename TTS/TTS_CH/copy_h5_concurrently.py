import os
import shutil
from multiprocessing import Process

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
DATASETS_PATH = "/cluster/home/stucksam/datasets/dialects"
DIALECT_TRAIN_PATH = "/scratch/dialects"

def copy_dialect(dialect: str):
    print(f"Copying {dialect} to /scratch partition.")
    shutil.copy2(os.path.join(DATASETS_PATH, f"{dialect}.hdf5"), DIALECT_TRAIN_PATH)
    shutil.copy2(os.path.join(DATASETS_PATH, f"{dialect}.txt"), DIALECT_TRAIN_PATH)
    print(f"Finished copying {dialect}.")


def copy_dialects_to_cluster_concurrently():
    if not os.path.exists(DIALECT_TRAIN_PATH):
        os.makedirs(DIALECT_TRAIN_PATH)

    processes = [
        Process(target=copy_dialect, args=(dialect,))
        for dialect, _ in LANG_MAP_INV.items()
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    copy_dialects_to_cluster_concurrently()