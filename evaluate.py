import json
import os
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    jaccard_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tabulate import tabulate
from torchmetrics import ConfusionMatrix
from enum import Enum

pd.set_option("display.max_colwidth", None)


class Task(Enum):
    BREAST = "breast"
    CRC100K = "crc100k"
    PCAM = "pcam"
    MSSI = "mssi"
    MHIST = "mhist"

### for CRC100K
answer_to_label_crc100K = {
    "Normal": "NORM", # Set to "No Cancer" for binary classification
    "Cancer": "TUM",
    "Adipose": "ADI",
    "Background": "BACK", # TODO: unused
    "Debris": "DEB",
    "Lymphocytes": "LYM",
    "Mucus": "MUC",
    "Muscle": "MUS",
    "Stroma": "STR",
}

encode_label_crc100k = {
    "NORM": 0,
    "TUM": 1,
    "DEB": 2,
    "ADI": 3,
    "LYM": 4,
    "MUC": 5,
    "MUS": 6,
    "STR": 7,
    "BACK": 8, # TODO: unused
}

### for MHIST
answer_to_label_mhist = {
    "SSA": "SSA",
    "HP": "HP",
}

encode_label_mhist = {
    "HP": 0,
    "SSA": 1,
}


### for PCAM
answer_to_label_pcam = {
    "No Cancer": "NORM",
    "Cancer": "TUM",
}

encode_label_pcam = {
    "NORM": 0,
    "TUM": 1,
}

### for MSSI
answer_to_label_mssi = {
    "MSI": "MSI",
    "MSS": "MSS",
}

encode_label_mssi = {
    "MSS": 0,
    "MSI": 1,
}


### for Breast Cancer
answer_to_label_breast = {
    "No Cancer": "lesion_none",
    "Cancer": "lesion_malignant",
    "Benign Lesion": "lesion_benign",
}

encode_label_breast = {
    "lesion_none": 0,
    "lesion_malignant": 1,
    "lesion_benign": 2,
}


def get_result_files(directory: str = "./Results", subdir: str = None):
    """Return results files"""
    paths = (Path(directory).joinpath(subdir)).glob("*.json")
    paths = [path for path in paths if "rerun" not in path.stem]
    return paths


def get_rerun_files(directory: str = "./Results", subdir: str = None):
    """Return results files that are reruns. This might happen due to timeouts during running"""
    paths = Path(directory).joinpath(subdir).glob("*.json")
    paths = [path for path in paths if "rerun" in path.stem]
    return paths


def match_files(result_files, rerun_files):
    """Return list of tuples of matching result and rerun files"""
    matched_files = {result_file: None for result_file in result_files}
    res_runs = {str(res_file.stem)[-1]: res_file for res_file in result_files}
    rerun_runs = {str(rerun_file.stem)[-1]: rerun_file for rerun_file in rerun_files}
    matched_runs = res_runs.keys() & rerun_runs.keys()
    for matched_run in matched_runs:
        matched_files.update({res_runs[matched_run]: rerun_runs[matched_run]})

    return matched_files


def merge_responses(original_df, rerun_df):
    """Replace non-existing responses in original df with responses from rerun df"""
    original_df.set_index("fname", inplace=True)
    rerun_df.set_index("fname", inplace=True)
    original_df.update(rerun_df)
    original_df.reset_index(inplace=True)
    return original_df


def make_evaluation_df(file_path):
    with fsspec.open(file_path, mode="rb") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def accuracy(df, true_label="label", model_pred="answer", normalize=True):
    return accuracy_score(df[true_label], df[model_pred], normalize=normalize)


def make_stats_table(
    df, save_path, *, true_label="label", model_pred="answer", normalize=True, encode_label=None, multiclass=False, 
):
    results = {}
    true, pred = df[true_label], df[model_pred]
    results["accuracy"] = accuracy_score(true, pred, normalize=normalize)

    trues = torch.Tensor(df[true_label].map(encode_label).tolist())
    preds = torch.Tensor(df[model_pred].map(encode_label).tolist())
    results["precision_score"] = precision_score(
        trues, preds, average="binary" if not multiclass else None
    )  # preds = estimated target
    results["recall_score"] = recall_score(trues, preds, average="binary" if not multiclass else None)  # preds = estimated target
    results["jaccard_score"] = jaccard_score(trues, preds, average="binary" if not multiclass else None)  # preds = estimated labels

    if not multiclass:
        preds_scores = torch.Tensor(df["score"])
        fpr, tpr, _ = roc_curve(trues, preds_scores)
        results["fpr"] = fpr
        results["tpr"] = tpr
        results["roc_auc"] = auc(fpr, tpr)

    results = pd.DataFrame.from_dict(results, orient="index", columns=["value"])
    results.index.name = "metric"
    results.to_csv(f"{save_path}_stats_table.csv")

    return results


def confusion_matrix(
    df,
    save_path,
    *,
    task="binary",
    true_label="label",
    model_pred="answer",
    normalize="true",
    encode_label=None,
    subdir=None,
):
    metric = ConfusionMatrix(
        task=task, num_classes=df[true_label].nunique(), normalize=normalize
    )

    sorted_labels = sorted(df[true_label].unique(), reverse=True)
    encoded = {label: encode_label[label] for label in sorted_labels}
    preds = torch.Tensor(df[model_pred].map(encoded))
    targets = torch.Tensor(df[true_label].map(encoded))
    metric.update(preds, targets)
    conf_matrix = metric.compute()
    counts = pd.DataFrame(index=sorted_labels, columns=sorted_labels, data=0)
    
    for true, pred in zip(df[true_label], df[model_pred]):
        counts.iloc[encoded[true], encoded[pred]] += 1

    annot_matrix = np.array(
        [
            [
                "{:.2f}\n(n={})".format(score, count)
                for score, count in zip(score_row, count_row)
            ]
            for score_row, count_row in zip(conf_matrix, counts.values)
        ]
    )


    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        conf_matrix,
        annot=annot_matrix,
        ax=ax,
        fmt="",
        cmap="Purples",
        square=True,
        vmin=0,
        vmax=1,
        cbar=False,
    )

    decoded_labels = {v: k for k, v in encoded.items()}
    ax.set_xlabel("Predicted Labels", fontsize=8)
    ax.set_ylabel("True Labels", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.set_xticklabels([decoded_labels.get(int(tick.get_text()), "") for tick in ax.get_xticklabels()])
    ax.set_yticklabels([decoded_labels.get(int(tick.get_text()), "") for tick in ax.get_yticklabels()])
    ax.set_title(subdir.replace("_", " ").title() + " Confusion Matrix", fontsize=8)

    font_size = 4 if conf_matrix.shape[0] > 4 else 8

    for text in ax.texts:
        text_string = text.get_text()
        if "\n" in text_string:
            score, count = text_string.split("\n")
            text.set_text(score + "\n" + count)
            text.set_fontsize(font_size)

    plt.tight_layout()
    plt.savefig(f"{save_path}_confusion_matrix.png", dpi=200)

    return fig, ax


def auroc(
    df,
    save_path,
    *,
    true_label="label",
    model_score="score",
    encode_label=None,
    subdir=None,
):
    preds = torch.Tensor(df[model_score])
    # for now: target = 1 should be "Cancer" as the model gives its certainty for this class
    
    # print(df[true_label])
    targets = torch.Tensor(
        [encode_label[x] for x in df[true_label]]
    )  # TODO: as the model returns the score as measure of beeing certain of the predicted class -> reimplement this part here
    fpr, tpr, thresholds = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)

    sns.set_style("white")

    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.lineplot(
        x="False Positive Rate",
        y="True Positive Rate",
        data=roc_data,
        label=f"(AUC = {roc_auc:.2f})",
        lw=2,
    )

    ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Receiver Operating Characteristic for " + subdir.replace("_", " ").title(), fontsize=8)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}_roc_curve.png", dpi=300)

    return fig, ax


def process_df(df, answer_to_label):
    df["model_response"] = df["model_response"].apply(
        lambda x: json.loads(json.loads(x))
    )  # convert from jsonstring to dict
    df_exploded = df["model_response"].apply(pd.Series)  # explode dict into columns
    df = pd.concat([df, df_exploded], axis=1)  # concat back together
    df["label"] = df["sample"].apply(lambda x: x["label"])
    df["fname"] = df["sample"].apply(lambda x: x["fname"])
    df["answer"] = df["answer"].map(answer_to_label)
    df["correct"] = df["label"] == df["answer"]

    return df

def prepare_for_tabulate(df):
    prepared_df = df.copy()
    for column in prepared_df.columns:
        prepared_df[column] = prepared_df[column].apply(
            lambda x: str(x) if isinstance(x, np.ndarray) else x
        )
    return prepared_df


def get_label_dicts(dataset):
    if dataset == Task.CRC100K:
        return answer_to_label_crc100K, encode_label_crc100k
    elif dataset == Task.BREAST:
        return answer_to_label_breast, encode_label_breast
    elif dataset == Task.PCAM:
        return answer_to_label_pcam, encode_label_pcam
    elif dataset == Task.MSSI:
        return answer_to_label_mssi, encode_label_mssi
    elif dataset == Task.MHIST:
        return answer_to_label_mhist, encode_label_mhist
    else:
        raise ValueError("Invalid Dataset")


def main(subdir, task, multiclass):
    terminal_width = os.get_terminal_size().columns
    pd.set_option("display.width", terminal_width)

    results_files = get_result_files(subdir=subdir)
    rerun_files = get_rerun_files(subdir=subdir)

    matched_files = match_files(results_files, rerun_files)

    answer_to_label, encode_label = get_label_dicts(task)

    for res_file, rerun_file in matched_files.items():
        print("*" * terminal_width)
        print(res_file, rerun_file)
        res_df = make_evaluation_df(res_file)
        if rerun_file is not None:
            rerun_df = make_evaluation_df(rerun_file)
            res_df = merge_responses(res_df, rerun_df)

        df = process_df(res_df, answer_to_label=answer_to_label)

        save_dir = f"./Stats/{subdir}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if multiclass:
            confusion_matrix(df, f"{save_dir}/{res_file.stem}", encode_label=encode_label, task="multiclass", subdir=subdir)
        else:
            confusion_matrix(df, f"{save_dir}/{res_file.stem}", encode_label=encode_label, subdir=subdir)
            auroc(df, f"{save_dir}/{res_file.stem}", encode_label=encode_label, subdir=subdir)
        stats_tbl = make_stats_table(df, f"{save_dir}/{res_file.stem}", encode_label=encode_label, multiclass=multiclass)
        stats_tbl.reset_index(inplace=True)
        print(
            tabulate(
                prepare_for_tabulate(stats_tbl),
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )
        df.to_csv(f"./Stats/{subdir}/{res_file.stem}.csv", index=False)
        print("SAVING FILE TO: ")
        print(f"./Stats/{subdir}/{res_file.stem}.csv")
        print("DONE")


if __name__ == "__main__":

    #subdir = "CRC100K/KNN/zero_shot"
    subdir = "CRC100K/KNN/three_shot"
    main(subdir, task=Task.CRC100K, multiclass=True)

    # ### SET BASE DIR HERE
    # base_dir = "Results/PCam/KNN"
    # multiclass = False
    # task = Task.PCAM
    # base_dir = Path(base_dir).resolve()
    # subdirs = [subdir for subdir in base_dir.iterdir() if subdir.is_dir()]
    # for subdir in subdirs:
    #     subdir = Path(*subdir.parts[-3:])
    #     main(subdir, task=task, multiclass=multiclass)
