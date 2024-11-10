import os
import random

from torch.nn import Embedding, Linear
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.alt_loggers import WandbLogger
from TTS.utils.manage import ModelManager

random.seed(240753)

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
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT"
PROJECT_NAME = "STT4SG_XTTS_trainer"
DASHBOARD_LOGGER = "wandb"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
# OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")
OUT_PATH = "/cluster/home/stucksam/coqui-tts/TTS/TTS_CH/trained"

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will star with evaluation
BATCH_SIZE = 12  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.

BASE_DATASET_PATH = "/cluster/home/stucksam/datasets/dialects"

DATASETS_CONFIG_LIST = []

txt_files = [f for f in os.listdir(BASE_DATASET_PATH) if f.endswith('.txt')]

for metadata in txt_files:
    dialect_name = metadata.replace(".txt", "")
    with open(os.path.join(BASE_DATASET_PATH, metadata), "rt", encoding='utf-8') as metadata_file:
        nsamples = metadata_file.readlines()
        if len(nsamples) < 100:  # skip small dialects
            continue
        DATASETS_CONFIG_LIST.append(
            BaseDatasetConfig(
                formatter="ljspeech_custom_dialect_speaker",  # create custom formatter with speaker name
                dataset_name=dialect_name,
                path=BASE_DATASET_PATH,
                meta_file_train=metadata,
                language=LANG_MAP_INV[dialect_name],  # create dial_id
            )
        )

# BASE_DATASET_PATH_SD = "/cluster/data/deri/swissdial"
#
# for dial in os.listdir(BASE_DATASET_PATH_SD):
#     with open(os.path.join(BASE_DATASET_PATH_SD, dial, 'metadata.txt'), "rt", encoding='utf-8') as metadata_file:
#         nsamples = metadata_file.readlines()
#         if len(nsamples) < 100:
#             continue
#         language = f'ch_{dial}' if not dial == 'ag' else 'zh'
#         DATASETS_CONFIG_LIST.append(
#             BaseDatasetConfig(
#                 formatter="ljspeech_custom_speaker",  # create custom formatter with speaker name
#                 dataset_name=f"stt4sg_{dial}",
#                 path=os.path.join(BASE_DATASET_PATH_SD, dial),
#                 meta_file_train="metadata.txt",
#                 language=f'ch_{dial}',  # create dial_id
#             )
#         )

# Define here the dataset that you want to use for the fine-tuning on.
# config_dataset = BaseDatasetConfig(
#     formatter="ljspeech_custom_speaker",  # create custom formatter with speaker name
#     dataset_name="dante",
#     path="/cluster/data/deri/dante_dataset",
#     meta_file_train="metadata.txt",
#     language="it",  # create dial_id
# )

# Add here the configs of the datasets
# DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
# os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

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

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
TOKENIZER_FILE = "/cluster/home/stucksam/coqui-tts/TTS_CH/trained/GPT_XTTS_v2.0_LJSpeech_FT-October-07-2024_11+47AM-fc3e366f/vocab.json"  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
XTTS_CHECKPOINT = "/cluster/home/stucksam/coqui-tts/TTS_CH/trained/GPT_XTTS_v2.0_LJSpeech_FT-October-07-2024_11+47AM-fc3e366f/best_model_55500.pth"  # model.pth file

XTTS_RELOAD = True

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# Training sentences generations
SPEAKER_REFERENCE = [
    "/cluster/home/stucksam/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1199.wav",
    "/cluster/home/stucksam/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1191.wav",
    "/cluster/home/stucksam/speakers/b073e82c-ae02-4a2a-a1b4-f5384b8eb9a7_1095.wav",

    # speaker reference to be used in training test sentences
]


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
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
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=16000, dvae_sample_rate=16000, output_sample_rate=16000)
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
        eval_split_max_size=256,
        eval_split_size=0.02,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        wandb_entity="stucksam",
        # target_loss="loss",
        print_eval=False,
        run_eval_steps=2500,
        datasets=DATASETS_CONFIG_LIST,
        shuffle=True,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
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
                "text": "Diese Privatperson hat sie anscheinend sogar vermietet.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_zh',
            },
            {
                "text": "Das ist ein Hinweis für die zukünftige Planung.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_vs',
            },
            {
                "text": "Das ist ein Hinweis für die zukünftige Planung.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_bs',
            },
            {
                "text": "Und mehr Autos wollen die Schwaben auf keinen Fall bauen.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_os',
            },
            {
                "text": "Und mehr Autos wollen die Schwaben auf keinen Fall bauen.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_in',
            },
            {
                "text": "Und mehr Autos wollen die Schwaben auf keinen Fall bauen.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": 'ch_gr',
            },
        ],
    )

    config.languages += list(LANG_MAP.keys())

    if not XTTS_RELOAD:
        model = GPTTrainer.init_from_config(config)

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
        model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

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
    trainer.dashboard_logger = WandbLogger(  # pylint: disable=abstract-class-instantiated
        project=config.project_name,
        name=config.run_name,
        config=config,
        entity=config.wandb_entity,
    )
    model.xtts.tokenizer.tokenizer.save(
        path=os.path.join(trainer.output_path, 'vocab.json')
    )
    trainer.fit()


if __name__ == "__main__":
    main()
