import argparse
import random
import time
import uuid
from datetime import datetime
from threading import Event
from typing import Tuple, Optional, Callable

import gradio as gr
import pandas as pd
from datasets import Audio, load_from_disk, Dataset, concatenate_datasets

from processing.util import load_reference_sentences

SMOS_INFO = """
<div class='cmos_info info'>
    <p class='desc'>
        <strong>SMOS (Similarity Mean Opinion Score)</strong> is used to evaluate how similar the synthesized speech is to the reference speaker's voice, <strong>irrespective of the content</strong>.
        It measures whether the TTS model captures the speaker's unique characteristics, such as tone, pitch, and speaking style. The SMOS scale ranges from 1 to 5, with increments of 0.5 points:
    </p>
    <ul class='values'>
        <li><strong>1:</strong> Completely Dissimilar.</li>
        <li><strong>2:</strong> Mostly Dissimilar, with minor similarities.</li>
        <li><strong>3:</strong> Somewhat Similar, noticeable differences in tone or style.</li>
        <li><strong>4:</strong> Mostly Similar, minor differences only.</li>
        <li><strong>5:</strong> Same Voice.</li>
    </ul>
</div>
"""

SMOS_VALUES = [x * 0.5 for x in range(2, 11)]

CMOS_INFO = """
<div class='cmos_info info'>
    <p class='desc'>
        <strong>CMOS (Comparative Mean Opinion Score)</strong> is used to evaluate the comparative naturalness of synthesized speech against a given reference speech. 
        "Better" refers to synthesized speech that is smoother, more natural, and closer to human-like qualities than the reference, 
        while "worse" describes speech that sounds robotic, unnatural, or distorted compared to the reference. <strong>The spoken text is of no relevance</spoken>
        The CMOS scale ranges from <strong>-3</strong> to <strong>3</strong>, with intervals of 1:
    </p>
    <ul class='values'>
        <li><strong>-3:</strong> Much worse than the reference.</li>
        <li><strong>-2:</strong> Worse than the reference.</li>
        <li><strong>-1:</strong> Slightly worse than the reference.</li>
        <li><strong>0:</strong> Same as the reference.</li>
        <li><strong>+1:</strong> Slightly better than the reference.</li>
        <li><strong>+2:</strong> Better than the reference.</li>
        <li><strong>+3:</strong> Much better than the reference.</li>
    </ul>
</div>
"""
CMOS_VALUES = list(range(-3, 4))

INTELLIGIBILITY_INFO = """
<div class='cmos_info info'>
    <p class='desc'>
        <strong>Intelligibility</strong> evaluates how easy it is to understand the synthesized speech and ensures that the synthesized speech is clear and comprehensible with the provided sample text.
    </p>
    <ul class='values'>
        <li><strong>1 (Bad): </strong> Very difficult or impossible to understand.</li>
        <li><strong>3 (Fair):</strong> Moderately clear, with some unclear or mispronounced words.</li>
        <li><strong>5 (Excellent):</strong> Perfectly clear and easy to understand.</li>
    </ul>
</div>
"""
INTELLIGIBILITY_VALUES = list(range(1, 6))

text_column = "text"
gen_audio_column = "audio"
ref_audio_column = "audio"
ref_column = "reference_audio_id"
phon_did_cls = {0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Graubünden", 4: "Ostschweiz", 5: "Basel", 6: "Bern",
                7: "Unsicher"}
css = """
.form-fill .form {
    border: 1px solid #ffb37f !important;
}
.info .desc, .info .values li, .info strong {
    color: #444 !important;
}
.info li {
    padding-left: 20px
}
"""


def run_synthesis_evaluation_app(
        gen_datasets: list[Dataset],
        dataset_labels: list,
        ref_dataset: Dataset,
        group: int = 1,
        share: bool = False,
        debug: bool = False,
        seed: int = 123,
        user: str = "test",
        password: str = "123",
        on_submit_cb: Optional[Callable[[pd.DataFrame], None]] = None
):
    """
    Launches a Gradio-based evaluation interface to compare TTS and voice cloning systems.

    :param gen_datasets: List of synthesized audio datasets to evaluate.
    :param dataset_labels: List of labels for each dataset, used for identification.
    :param ref_dataset: Reference dataset for comparing synthesized samples.
    :param group: Group number.
    :param share: Flag to enable public sharing of the Gradio app.
    :param debug: Set true to enable debug mode.
    :param seed: The seed is used at random.
    :param user: The username used for the login to the gradio app.
    :param password: The password used for the login to the gradio app.
    :param on_submit_cb:
    :return: Dictionary with the selected best dataset for each audio sample.
    """

    random.seed(seed)

    assert len(gen_datasets) > 0, 'At least one dataset is required for evaluation, but none were provided.'
    assert ref_dataset, 'Reference dataset is required for evaluation, but none was provided.'
    assert len(dataset_labels) == len(
        gen_datasets), f'Labels {len(dataset_labels)} and datasets {len(gen_datasets)} must have the same length.'

    completion_event = Event()
    session_id = str(uuid.uuid4())

    # Add system tag to datasets to include only selected sample_ids
    gen_datasets = [
        ds.map(lambda x: {**x, "system": dataset_labels[ds_i]})
        for ds_i, ds in enumerate(gen_datasets)
    ]

    random_gen_dataset = concatenate_datasets(gen_datasets).shuffle()
    num_steps = len(random_gen_dataset)

    # Initialize results as a DataFrame
    results_df = pd.DataFrame.from_dict({
        "session_id": session_id,
        "sample_id": random_gen_dataset["sample_id"],
        "orig_dialect": random_gen_dataset["dialect"],
        "speaker_id": random_gen_dataset["speaker_id"],
        "system": random_gen_dataset["system"],
        "smos": [None] * len(random_gen_dataset["sample_id"]),
        "cmos": [None] * len(random_gen_dataset["sample_id"]),
        "intelligibility": [None] * len(random_gen_dataset["sample_id"]),
        "text": random_gen_dataset["text"],
    })

    print(f"Start Evaluation Session {session_id}")

    def count_values_plot():
        """
        Generates a DataFrame showing the count of each label in the results.

        :return: DataFrame containing label counts.
        """
        smos_counts = results_df['smos'].value_counts().sort_index()
        cmos_counts = results_df['cmos'].value_counts().sort_index()

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            "Score": smos_counts.index.tolist() + cmos_counts.index.tolist(),
            "Count": smos_counts.tolist() + cmos_counts.tolist(),
            "Metric": ["SMOS"] * len(smos_counts) + ["CMOS"] * len(cmos_counts),
        })

        return summary_df

    def _select_index(index: int) -> Tuple:
        """
        Advances to the next sample in the evaluation, updating the display accordingly.

        :param index: Next sample index.
        :return: Updated Gradio components for the next sample.
        """

        print(f"Select index {index}")

        gen_sample = random_gen_dataset[index]
        orig_sample = None
        for ref in ref_dataset:
            if ref["speaker_id"] == gen_sample["speaker_id"]:
                orig_sample = ref
                break
        if orig_sample is None:
            print(gen_sample)
            print(gen_sample['name'])
            print(ref_dataset['name'])

        g_text = gen_sample[text_column]
        g_audio = (gen_sample[gen_audio_column]["sampling_rate"], gen_sample[gen_audio_column]["array"])
        r_audio = (orig_sample[ref_audio_column]["sampling_rate"], orig_sample[ref_audio_column]["array"])
        return index, g_text, g_audio, r_audio

    def _next_index(prev_index: int) -> Tuple:
        """

        :param prev_index:
        :return:
        """
        next_index = prev_index + 1
        completed = next_index >= num_steps

        if completed:
            next_result = None, None, None, None
            completion_event.set()  # Signal completion
            print("Completed. End app and submit results...")
        else:
            next_result = _select_index(next_index)

        return gr.update(visible=not completed), gr.update(visible=completed), None, None, None, results_df, *next_result

    def start() -> Tuple:
        """
        Initializes the Gradio app to start the evaluation.

        :return: Gradio components set to the initial sample.
        """
        print(f"Start at Index 0")
        next_result = _select_index(0)
        return gr.update(visible=True), gr.update(visible=False), *next_result

    def submit(sample_index: int, smos: int, cmos: int, intelligibility: int) -> Tuple:
        """

        :param sample_index:
        :param smos:
        :param cmos:
        :return:
        """

        if smos is None:
            gr.Warning("Please select a SMOS value.", duration=5)
            return _next_index(sample_index - 1)

        if cmos is None:
            gr.Warning("Please select a CMOS value.", duration=5)
            return _next_index(sample_index - 1)

        if intelligibility is None:
            gr.Warning("Please select a intelligibility value.", duration=5)
            return _next_index(sample_index - 1)

        # Save results
        results_df.loc[sample_index, "smos"] = smos
        results_df.loc[sample_index, "cmos"] = cmos
        results_df.loc[sample_index, "intelligibility"] = intelligibility


        if on_submit_cb:
            on_submit_cb(results_df)

        print(f"Submit results for index {sample_index}, SMOS: {smos}, CMOS: {cmos}, Intelligibility {intelligibility}")
        return _next_index(sample_index)

    def get_subtitle(sample_index) -> str:
        """
        Gets the current subtitle

        :param sample_index: Index of the sample.
        :return: Title string
        """
        if sample_index is not None:
            return f"<h3 align='center'>{sample_index + 1} of {num_steps}</h3>"
        return ""

    # Gradio Interface
    with gr.Blocks(css=css) as demo:

        sample_idx = gr.State(0)
        gr.Markdown(f"<h1 align='center'>Evaluate TTS Systems</h1>")

        # End Column
        with gr.Column(visible=False) as end_col:
            gr.Markdown(
                "<p align='center' style='width: 70%; margin: 0 auto;'>"
                "You have reached the end of the evaluation session. "
                "Below are the results of your evaluation session, which have been successfully submitted. "
                "Feel free to review them. Once you're done, you can safely close this tab."
                "</p>"
            )

            result_table = gr.Dataframe(value=results_df)

        # Content
        with gr.Column(visible=False) as content_col:
            gr.Markdown(
                "<p align='center' style='width: 70%; margin: 0 auto;'>"
                "Listen to the audio samples below and evaluate them using SMOS, CMOS, and Intelligibility scores."
                "</p>"
            )

            gr.Markdown(get_subtitle, inputs=sample_idx)

            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="Reference Audio", interactive=False)

                with gr.Column():
                    with gr.Row():
                        gen_audio = gr.Audio(label=f"Generated Audio", interactive=False)

            with gr.Row():
                gen_text = gr.Textbox(interactive=False, label="Synthesized Text")
            with gr.Row(elem_classes=["form-fill"]):
                smos_radio = gr.Radio(choices=SMOS_VALUES, label="Similarity Mean Opinion Score (1 to 5)",
                                      interactive=True,
                                      info=SMOS_INFO)
            with gr.Row(elem_classes=["form-fill"]):
                cmos_radio = gr.Radio(choices=CMOS_VALUES, label="Comparative Mean Opinion Score (-3 to 3)",
                                      interactive=True,
                                      info=CMOS_INFO)
            with gr.Row(elem_classes=["form-fill"]):
                intelligibility_radio = gr.Radio(choices=INTELLIGIBILITY_VALUES, label="Intelligibility Score (1 to 5",
                                      interactive=True,
                                      info=INTELLIGIBILITY_INFO)

            with gr.Row():
                submit_button = gr.Button(f"Submit", variant="primary")
                submit_button.click(
                    fn=submit,
                    inputs=[sample_idx, smos_radio, cmos_radio, intelligibility_radio],
                    outputs=[content_col, end_col, smos_radio, cmos_radio, intelligibility_radio, result_table, sample_idx, gen_text,
                             gen_audio, ref_audio]
                )

        # Start
        with gr.Column(visible=True) as start_col:
            gr.Markdown(
                "<div style='font-size: 15px'>"
                "<p align='center' style='width: 70%; margin: 0 auto;'>"
                "In this evaluation, you will compare </strong>longer</strong> audio samples generated by different TTS systems to a reference speech sample of the reference speaker. "
                "The contents / the text of the generated audio reference audio <strong>must not be compared as they are not the same</strong>. "
                "For each speaker, assess each sample for clarity, accuracy, and similarity to the reference by providing scores for SMOS (speaker similarity) and what the general quality of the synthetic speech is (CMOS) and how understandable in general the generated speech is (Intelligibility). "
                "</p>"
                "<hr>"
                "<div style='width: 70%; margin: 20px auto !important;'>"
                f"{SMOS_INFO}"
                "</div>"
                "<div style='width: 70%; margin: 20px auto !important;'>"
                f"{CMOS_INFO}"
                "</div>"
                "<div style='width: 70%; margin: 20px auto !important;'>"
                f"{INTELLIGIBILITY_INFO}"
                "</div>"
                "</div>"
            )

            start_button = gr.Button(f"Start", variant="primary", visible=True)
            start_button.click(
                fn=start,
                outputs=[content_col, start_col, sample_idx, gen_text, gen_audio, ref_audio]
            )

        with gr.Row():
            gr.Markdown(f"<p align='center' style='color: #d5d5d5; font-weight: bold'>Session ID: {session_id}</p>")

    # Launch App
    if share:
        print("Launching with share enabled (auth=('test', '123'), debug=False, share=True)")
        demo.launch(prevent_thread_lock=True, debug=False, auth=(user, password), share=True)
    else:
        print("Launching with debug=True")
        demo.launch(prevent_thread_lock=True, debug=False)

    # Wait for evaluation to be completed
    completion_event.wait()

    gr.Info("Your results are submitted.", duration=5)
    time.sleep(1)

    print(f"End Evaluation Session {session_id}")
    print(results_df)
    date_time = datetime.now()
    results_df.to_csv(
        f"processing/save_long_group_{group}_{date_time.year}_{date_time.month}_{date_time.day}_{date_time.hour}_{date_time.minute}_{date_time.second}.csv",
        index=False)
    # Close Gradio
    print(f"Close App...")
    demo.close()
    time.sleep(1)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch demo with optional share argument.")
    parser.add_argument('--share', action='store_true', default=True, help="Enable share option.")
    parser.add_argument('--num_samples', default=7, help="Number of samples to compare. Defaults to 7.")
    parser.add_argument('--group', default=1, help="Group of evaluators, 1-3")
    args = parser.parse_args()

    dataset_paths_snf_long = [
        "generated_speech/GPT_XTTS_v2.0_Full_7_5/snf_long_dataset",
        "generated_speech/GPT_XTTS_v2.0_Full_7_6_SNF/snf_long_dataset",
        "generated_speech/GPT_XTTS_v2.0_Full_9_2_SNF/snf_long_dataset"
    ]
    dataset_paths_gpt_long = [
        "generated_speech/GPT_XTTS_v2.0_Full_7_5/gpt_long_dataset",
        "generated_speech/GPT_XTTS_v2.0_Full_7_6_SNF/gpt_long_dataset",
        "generated_speech/GPT_XTTS_v2.0_Full_9_2_SNF/gpt_long_dataset"
    ]
    dataset_paths_long = dataset_paths_snf_long + dataset_paths_gpt_long

    datasets_snf = [load_from_disk(ds) for ds in dataset_paths_snf_long]
    datasets_gpt = [load_from_disk(ds) for ds in dataset_paths_gpt_long]

    df_he_eval_snf = pd.read_csv("processing/data/human_eval_snf_long.csv", sep=",")
    df_he_eval_gpt = pd.read_csv("processing/data/human_eval_gpt_long.csv", sep=",")
    df_he_gpt = pd.read_csv("processing/data/7_6_enriched_gpt_long.csv", sep=",")
    end_idx = args.group * args.num_samples
    start_idx = end_idx - args.num_samples
    print(f"{start_idx} -> {end_idx}")

    subset_snf = df_he_eval_snf[start_idx:end_idx]
    subset_gpt = df_he_eval_gpt[start_idx:end_idx]
    # Create a set of valid (dialect, sentence_id) pairs
    valid_pairs_snf = set(zip(subset_snf["dialect_region"], subset_snf["ref_text"]))
    valid_pairs_gpt = set(zip(subset_gpt["speaker_id"], subset_gpt["ref_text"]))

    datasets_snf = [ds.filter(lambda x: (x["dialect"], x["text"]) in valid_pairs_snf) for ds in datasets_snf]
    datasets_gpt = [ds.filter(lambda x: (x["speaker_id"], x["text"]) in valid_pairs_gpt) for ds in datasets_gpt]

    datasets = datasets_snf + datasets_gpt
    for ds_idx, _ in enumerate(dataset_paths_long):
        audio = datasets[ds_idx][0]["audio"]
        if isinstance(audio, dict) and isinstance(audio["array"], list):
            datasets[ds_idx] = datasets[ds_idx].cast_column("audio", Audio(sampling_rate=audio["sampling_rate"]))

    reference = load_from_disk("speakers/cond_dataset")
    df_he_eval_ref = pd.read_csv("processing/data/human_eval_long_ref.csv", sep=",")
    snf_refs = df_he_eval_ref[df_he_eval_ref['speaker_id'].isin(subset_snf['speaker_id'])]
    gpt_refs = df_he_eval_ref[df_he_eval_ref['speaker_id'].isin(subset_gpt['speaker_id'])]

    reference_df = pd.concat([snf_refs, gpt_refs], ignore_index=True)
    reference_df = reference_df.drop_duplicates()
    valid_pairs_ref = set(zip(reference_df["speaker_id"], reference_df["text"]))
    dataset_ref = reference.filter(lambda x: (x["speaker_id"], x["text"]) in valid_pairs_ref)


    result_long = run_synthesis_evaluation_app(
        gen_datasets=datasets,
        dataset_labels=["SNF_long_coqui_srf", "SNF_long_coqui_srf_finetuned", "SNF_long_coqui_base", "GPT_long_coqui_srf", "GPT_long_coqui_srf_finetuned", "GPT_long_coqui_base"],
        # dataset_labels=["SNF_long_coqui_srf_finetuned", "SNF_long_coqui_base", "GPT_long_coqui_srf_finetuned", "GPT_long_coqui_base"],
        ref_dataset=dataset_ref,
        group=args.group,
        share=args.share,
        debug=True,
        seed=123
    )

    print(result_long)
    print("Gradio Evaluation App Finished")
