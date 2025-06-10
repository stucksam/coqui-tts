import os
import re

import pandas as pd

CLUSTER_HOME_PATH = "/cluster/home/stku"
CLUSTER_PROJECTS_PATH = "/cluster/projects/TTS-Swiss-German"
OUT_PATH = "/scratch/eval"

MODEL_EVAL_PATH = os.path.join(CLUSTER_PROJECTS_PATH, "generated_speech")

LANG_MAP = {
    'ch_be': 'Bern',
    'ch_bs': 'Basel',
    'ch_gr': 'Graubünden',
    'ch_in': 'Innerschweiz',
    'ch_os': 'Ostschweiz',
    'ch_vs': 'Wallis',
    'ch_zh': 'Zürich',
    'de': 'Deutschland'
}
LANG_MAP_INV = {v: k for k, v in LANG_MAP.items()}


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def extract_epoch_and_subset(folder: str) -> tuple[int, int]:
    match = re.search(r'epoch_(\d+)_subset_(\d+)', folder)
    if match:
        epoch = int(match.group(1))
        subset = int(match.group(2))
        print(f"Epoch={epoch}, Subset={subset}")
    else:
        print("Pattern not found.")

    return epoch, subset


if __name__ == "__main__":
    # Get all SwissGPC checkpoints used in evaluation
    swissgpc_filter = "SwissGPC"

    list_of_directories = [os.path.join(MODEL_EVAL_PATH, model_dir) for model_dir in
                           os.listdir(MODEL_EVAL_PATH) if swissgpc_filter in model_dir]
    # list_of_directories = ["GPT_XTTS_v2.0_SwissGPC_epoch_2_subset_3", "GPT_XTTS_v2.0_SwissGPC_epoch_6_subset_6",
    #                        "GPT_XTTS_v2.0_SwissGPC_epoch_5_subset_0"]

    df = []
    for directory in list_of_directories:
        folder_name = os.path.basename(os.path.normpath(directory))
        epoch, subset = extract_epoch_and_subset(folder_name)

        if not os.path.exists(os.path.join(directory, "speaker_similarity.csv")):
            print(f"{folder_name} has not been evaluated yet, continuing...")
            continue

        df_text = load_csv(os.path.join(directory, "de_text_calc.csv"))
        df_did_regions = load_csv(os.path.join(directory, "did_f1_regions.csv"))

        df_did_overall = load_csv(os.path.join(directory, "did_f1_overall.csv"))
        f1_overall = df_did_overall.iloc[-1]

        df_sim = load_csv(os.path.join(directory, "speaker_similarity.csv"))
        sim_overall = df_sim.iloc[-1]

        entry = {"checkpoint": folder_name,
                 "epoch": epoch,
                 "subset": subset,
                 "bert_avg": df_text["bert_score"].mean(),
                 "bert_med": df_text["bert_score"].median(),
                 "wer_avg": df_text["wer"].mean(),
                 "wer_med": df_text["wer"].median(),
                 "cer_avg": df_text["cer"].mean(),
                 "cer_med": df_text["cer"].median(),
                 "bleu_avg": df_text["bleu_score"].mean(),
                 "bleu_med": df_text["bleu_score"].median(),
                 "wer_low_avg": df_text["wer_lower"].mean(),
                 "wer_low_med": df_text["wer_lower"].median(),
                 "cer_low_avg": df_text["cer_lower"].mean(),
                 "cer_low_med": df_text["cer_lower"].median(),
                 "bleu_low_avg": df_text["bleu_score_lower"].mean(),
                 "bleu_low_med": df_text["bleu_score_lower"].median(),
                 "macro_f1": f1_overall["macro_f1"],
                 "micro_f1": f1_overall["micro_f1"],
                 "weighted_f1": f1_overall["weighted_f1"],
                 "speaker_sim_avg": sim_overall["avg_similarity"],
                 "speaker_sim_avg_rel": sim_overall["avg_rel_similarity"]
                 }
        df.append(entry)
    df = pd.DataFrame(df)
    save_path = os.path.join(MODEL_EVAL_PATH, "SwissGPC_eval_results.csv")
    df.to_csv(save_path, index=False, encoding="utf-8", sep=";")
    print(f"Evaluation result of SwissGPC trained XTTS model has been saved at: {save_path}")
