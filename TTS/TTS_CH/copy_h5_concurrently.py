import os
import shutil
import tarfile
from multiprocessing import Process

LANG_MAP = {
    "ch_be": "Bern",
    "ch_bs": "Basel",
    "ch_gr": "Graubünden",
    "ch_in": "Innerschweiz",
    "ch_os": "Ostschweiz",
    "ch_vs": "Wallis",
    "ch_zh": "Zürich",
    "de": "Deutschland"
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}
DATASETS_PATH = "/cluster/home/stucksam/datasets/dialects"
DIALECT_TRAIN_PATH = "/scratch/dialects"


def copy_dialect(dialect: str, tar_files: bool = False) -> None:
    print(f"Copying {dialect} to /scratch partition.")
    shutil.copy2(os.path.join(DATASETS_PATH, f"{dialect}.hdf5"), DIALECT_TRAIN_PATH)
    shutil.copy2(os.path.join(DATASETS_PATH, f"{dialect}.txt"), DIALECT_TRAIN_PATH)

    if tar_files:
        dialect_files = [
            os.path.join(DIALECT_TRAIN_PATH, f"{dialect}.hdf5"),
            os.path.join(DIALECT_TRAIN_PATH, f"{dialect}.txt")
        ]
        with tarfile.open(os.path.join(DIALECT_TRAIN_PATH, f"{dialect}.tar.gzip"), "w:gz") as tar:
            for file in dialect_files:
                tar.add(file, arcname=file)
        shutil.copy2(os.path.join(DIALECT_TRAIN_PATH, f"{dialect}.tar.gzip"), DATASETS_PATH)

    print(f"Finished copying {dialect}.")


def copy_dialects_to_cluster_concurrently(create_tar: bool = False) -> None:
    if not os.path.exists(DIALECT_TRAIN_PATH):
        os.makedirs(DIALECT_TRAIN_PATH)

    processes = [
        Process(target=copy_dialect, args=(dialect, create_tar))
        for dialect in LANG_MAP_INV.keys()
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    copy_dialects_to_cluster_concurrently()
