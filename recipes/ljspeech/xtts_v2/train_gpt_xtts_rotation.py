import logging
import os
import random
import shutil
from datetime import datetime
from multiprocessing import Process

from torch.nn import Embedding, Linear
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.alt_loggers import WandbLogger
from TTS.utils.manage import ModelManager

from xtts_data_point import DialectDataPoint

random.seed(18670209)

logger = logging.getLogger(__name__)

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

# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0"
PROJECT_NAME = "STT4SG_XTTS_trainer"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
# OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")
CLUSTER_HOME_PATH = "/cluster/home/stku"
# OUT_PATH = "/raid/admin/models"
OUT_PATH = f"{CLUSTER_HOME_PATH}/coqui-tts/TTS/TTS_CH/trained"
os.makedirs(OUT_PATH, exist_ok=True)

# DATASETS_PATH = "/raid/admin"  # only if locally on trinity
DATASETS_PATH = "/scratch/subsets"
os.makedirs(DATASETS_PATH, exist_ok=True)

CLUSTER_PROJECTS_PATH = "/cluster/projects/"
CLUSTER_PROJECTS_TTS = os.path.join(CLUSTER_PROJECTS_PATH, "TTS-Swiss-German")
TTS_TRAINING_SUBSETS_PATH = os.path.join(CLUSTER_PROJECTS_TTS, "audio_subsets")

NUMBER_OF_H5_SUBSETS = 10

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 36  # set here the batch size
GRAD_ACUMM_STEPS = 14  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.


# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# Training sentences generations
SPEAKER_REFERENCE = f"{CLUSTER_HOME_PATH}/_speakers/ch_gr/references/6516567b-0d9b-4853-880c-d5f0327dd384/bce2b8c3b3d3bd6ee287e41d0a4b9b41245e2529392472a6c19caf94634d3724.wav"

XTTS_RELOAD = False


def get_most_recent_checkpoint_folder() -> str | None:
    # List all items in the directory with full paths
    folders = [os.path.join(CHECKPOINTS_OUT_PATH, f) for f in os.listdir(CHECKPOINTS_OUT_PATH) if
               os.path.isdir(os.path.join(CHECKPOINTS_OUT_PATH, f))]
    # Get the folder with the most recent modification time
    if folders:
        return max(folders, key=os.path.getmtime)
    else:
        return None


def get_most_recent_model_checkpoint(folder: str) -> str | None:
    # List all items in the directory with full paths
    folders = [os.path.join(folder, f) for f in os.listdir(CHECKPOINTS_OUT_PATH) if
               "checkpoint" in f and os.path.isfile(os.path.join(CHECKPOINTS_OUT_PATH, f))]
    # Get the folder with the most recent modification time
    if folders:
        return max(folders, key=os.path.getmtime)
    else:
        return None


def load_model_files() -> tuple[str, str]:
    if not XTTS_RELOAD:
        tokenizer_file = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
        xtts_checkpoint = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

    else:
        folder = get_most_recent_checkpoint_folder()
        if folder is None:
            raise RuntimeError(f"No most recent folder found in {CHECKPOINTS_OUT_PATH}, please verify")
        model = get_most_recent_model_checkpoint(folder)
        if model is None:
            raise RuntimeError(f"No suitable checkpoint found in {folder}, please verify")

        tokenizer_file = f"{OUT_PATH}/{folder}/vocab.json"  # vocab.json file
        xtts_checkpoint = f"{OUT_PATH}/{folder}/{model}"  # model.pth file

    return tokenizer_file, xtts_checkpoint


def load_subset_metadata(subset_to_load: int) -> list:
    # Normal / All Samples
    meta_data_path = os.path.join(DATASETS_PATH, f"subset_{subset_to_load}.txt")
    assert os.path.exists(meta_data_path), (f"Subset {subset_to_load} was not found under {DATASETS_PATH}, "
                                            f"please check.")

    sample_list = {key: [] for key in LANG_MAP_INV}
    with open(meta_data_path, "rt", encoding='utf-8') as meta_file:
        for line in meta_file:
            split_line = line.replace('\n', '').split('\t')
            sample_point = DialectDataPoint.load_single_datapoint(split_line)
            if sample_point.dialect == "English":
                continue
            sample_list[sample_point.dialect].append(sample_point)

    # drop empty lists should a dialect not be present
    sample_list = {k: v for k, v in sample_list.items() if v}

    # write out the dialect files
    for dialect, samples in sample_list.items():
        dialect_meta_path = os.path.join(DATASETS_PATH, f"{dialect}_{subset_to_load}.txt")
        with open(dialect_meta_path, "wt", encoding="utf-8") as f:
            for line in samples:
                f.write(line.to_string())
        assert os.path.exists(dialect_meta_path), (f"Dialect {dialect} was not found under {DATASETS_PATH}, "
                                                   f"please check.")

    config_list = []
    for dialect, samples in sample_list.items():
        config_list.append(
            BaseDatasetConfig(
                formatter="ljspeech_custom_subset_h5_speaker",  # create custom formatter with speaker name
                dataset_name=f"subset_{subset_to_load}",
                path=DATASETS_PATH,
                meta_file_train=f"{dialect}_{subset_to_load}.txt",
                language=LANG_MAP_INV[dialect],  # create dial_id
            )
        )

    return config_list


def copy_subset_to_scratch(subset_to_copy: int) -> None:
    logger.info(f"Copying subset {subset_to_copy} to scratch")
    shutil.copy2(os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{subset_to_copy}.hdf5"), DATASETS_PATH)
    shutil.copy2(os.path.join(TTS_TRAINING_SUBSETS_PATH, f"subset_{subset_to_copy}.txt"), DATASETS_PATH)
    logger.info(f"Successfully copied subset {subset_to_copy} to scratch")


def remove_previous_subset(subset_to_remove: int) -> None:
    logger.info(f"Deleting subset {subset_to_remove} to scratch")
    os.remove(os.path.join(DATASETS_PATH, f"subset_{subset_to_remove}.hdf5"))
    os.remove(os.path.join(DATASETS_PATH, f"subset_{subset_to_remove}.txt"))
    for dialect in LANG_MAP_INV.keys():
        dialect_path = os.path.join(DATASETS_PATH, f"{dialect}_{subset_to_remove}.txt")
        if os.path.exists(dialect_path):
            os.remove(dialect_path)


def main():
    print("Started")

    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 seconds with sr of 22050
        min_conditioning_length=66150,  # 3 secs with sr of 22050
        debug_loading_failures=False,
        max_wav_length=330750,  # ~15 seconds = 240000/22050 -> 16k is sample rate of wavs -> we now upsample!
        max_text_length=390,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    print("GPTArgs generated...")

    # define audio config
    # audio_config = XttsAudioConfig(sample_rate=16000, dvae_sample_rate=16000, output_sample_rate=24000)
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    print(f"Verifying Sample Rate: {audio_config.sample_rate}")
    print(f"Verifying DVAE Sample Rate: {audio_config.dvae_sample_rate}")
    print(f"Verifying Output Sample Rate: {audio_config.output_sample_rate}")
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        model_param_stats=False,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=2,
        epochs=1,  # IMPORTANT for subset rotation training as we want to train one after the other for 1 epoch
        # eval_split_max_size=256,
        eval_split_size=0.1,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=2000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        wandb_entity="stucksam",
        # target_loss="loss",
        print_eval=False,
        run_eval_steps=2000,
        datasets=DATASETS_CONFIG_LIST,
        shuffle=True,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=6e-05,  # learning rate, maybe change to 0.00018
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        use_h5=True,
        test_sentences=[
            {
                "text": "Diese Privatperson hat sie anscheinend sogar vermietet.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_be',
            },
            {
                "text": "Diese Privatperson hat sie anscheinend sogar vermietet. Und mehr Autos wollen die Schwaben auf keinen Fall bauen.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_zh',
            },
            {
                "text": "Das ist ein Hinweis für die zukünftige Planung. Diese Privatperson hat sie anscheinend sogar vermietet.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_vs',
            },
            {
                "text": "Das ist ein Hinweis für die zukünftige Planung.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_bs',
            },
            {
                "text": "Und mehr Autos wollen die Schwaben auf keinen Fall bauen. Den Weihnachtsbaum haben die Arbeiter sorgfältig über das Wochenende aufgebaut.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_os',
            },
            {
                "text": "Und mehr Autos wollen die Schwaben auf keinen Fall bauen.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_in',
            },
            {
                "text": "Den Weihnachtsbaum haben die Arbeiter sorgfältig über das Wochenende aufgebaut.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_gr',
            },
            {
                "text": "Den Weihnachtsbaum haben die Arbeiter sorgfältig über das Wochenende aufgebaut.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'de',
            },
        ]
    )

    print("GPT Trainer Config generated...")

    config.languages += list(LANG_MAP.keys())

    if not XTTS_RELOAD:
        model = GPTTrainer.init_from_config(config)

        print("Loading new Model...")

        new_toks = ['[ch_be]', '[ch_bs]', '[ch_gr]', '[ch_in]', '[ch_os]', '[ch_vs]', '[ch_zh]']
        model.xtts.tokenizer.tokenizer.add_special_tokens(
            new_toks
        )
        new_ids = [model.xtts.tokenizer.tokenizer.encode(t).ids[0] for t in new_toks]

        old_te = model.xtts.gpt.text_embedding
        old_th = model.xtts.gpt.text_head
        old_number_text_token = model.xtts.gpt.number_text_tokens

        model_dim = old_te.weight.shape[-1]
        number_text_tokens = model.xtts.tokenizer.get_number_tokens()
        model.xtts.args.gpt_number_text_tokens = number_text_tokens

        new_text_embedding = Embedding(number_text_tokens, model_dim)
        new_text_head = Linear(model_dim, number_text_tokens)

        model.xtts.gpt.text_embedding = new_text_embedding
        model.xtts.gpt.text_head = new_text_head

        for i in range(old_number_text_token):
            new_text_embedding.weight.data[i] = old_te.weight.data[i]
            new_text_head.weight.data[i] = old_th.weight.data[i]
            new_text_head.bias.data[i] = old_th.bias.data[i]

    else:
        print("Loading existing model...")
        model = GPTTrainer.init_from_config(config)

    print("Successfully loaded Model. Loading Training Samples now...")

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print("Loaded tts samples.")

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None if not XTTS_RELOAD else XTTS_CHECKPOINT,
            # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("Initialized Trainer...")

    trainer.dashboard_logger = WandbLogger(  # pylint: disable=abstract-class-instantiated
        project=config.project_name,
        name=config.run_name,
        config=config,
        entity=config.wandb_entity,
    )
    model.xtts.tokenizer.tokenizer.save(
        path=os.path.join(trainer.output_path, 'vocab.json')
    )

    print("Start fitting")
    trainer.fit()


if __name__ == "__main__":

    logging.basicConfig(filename=f"{datetime.now().strftime('%Y_%m_%d_%H_%M.log')}", level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    subset = 0
    copy_subset_to_scratch(subset)

    while True:
        # We just train on 0 or 1 and replace respective h5 on iteration
        logger.info(f"Training on subset: {subset}")

        if subset + 1 == NUMBER_OF_H5_SUBSETS:
            next_subset = 0
        else:
            next_subset = subset + 1

        # Copy next h5 in background
        process = Process(target=copy_subset_to_scratch, args=(next_subset,))
        process.start()

        # XTTS transfer learning parameters: You need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
        TOKENIZER_FILE, XTTS_CHECKPOINT = load_model_files()

        # download XTTS v2.0 files if needed
        if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
            logger.info(" > Downloading XTTS v2.0 files!")
            ModelManager._download_model_files(
                [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
            )

        DATASETS_CONFIG_LIST = load_subset_metadata(subset)

        main()

        process.join()
        remove_previous_subset(subset)

        subset = next_subset

        # switch to reloading checkpoints after initial training
        if XTTS_RELOAD is False:
            XTTS_RELOAD = True
