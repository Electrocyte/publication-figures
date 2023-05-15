#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:25:33 2021

@author: mangi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
from featurewiz import featurewiz
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
import os
import json
import statistics
from string import whitespace
from functools import reduce

pd.options.mode.chained_assignment = None  # default='warn'

# https://stats.stackexchange.com/questions/99162/what-is-one-class-svm-and-how-does-it-work
# https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html#sphx-glr-auto-examples-applications-plot-outlier-detection-wine-py

# The OneClassSVM is known to be sensitive to outliers and thus does not perform very well for outlier detection.
# This estimator is best suited for novelty detection when the training set is not contaminated by outliers.

# combine predictions using indices and return df
def clean_out(
    index: np.ndarray, y_pred: np.ndarray, dirty_df: pd.DataFrame
) -> pd.DataFrame:
    df_OCS = pd.DataFrame(np.vstack((index, y_pred)).T)
    df_OCS.set_index(0, inplace=True)
    df_OCS = df_OCS.rename(columns={1: "OCS"})
    merged_df = pd.merge(dirty_df, df_OCS, left_index=True, right_index=True)
    return merged_df


def clean_strings(string) -> str:
    # from string import whitespace
    # string = "Cumulative frequency for detection of MVM reads per minute for batch19"
    len_str = len(string)
    mid_point = int(len_str / 2)
    whitespaces = [i for i, char in enumerate(string) if char in whitespace]
    cut_string = min(whitespaces, key=lambda x: abs(x - mid_point))
    new_string = f"{string[0:cut_string]}\n{string[cut_string:]}"
    return new_string


def featurewiz_for_fewer_features(
    directory: str,
    true_positives_clean: pd.DataFrame,
    false_positives_clean: pd.DataFrame,
    vees: str,
) -> list:
    # out = f"{directory}/OneClassSVM/"
    # os.makedirs(out, exist_ok=True)

    fw_tp_clean = true_positives_clean
    fw_tp_clean["special_mask"] = "True"
    fw_fp_clean = false_positives_clean
    fw_fp_clean["special_mask"] = "False"
    fw_df = pd.concat([fw_tp_clean, fw_fp_clean])

    encoded_ML = fw_df
    encoded_ML.reset_index(inplace=True, drop=True)

    # creating instance of labelencoder
    labelencoder = (
        LabelEncoder()
    )  # Assigning numerical values and storing in another column
    encoded_ML["encoded_mask"] = labelencoder.fit_transform(encoded_ML["special_mask"])

    # creating instance of one-hot-encoder
    enc = OneHotEncoder(
        handle_unknown="ignore"
    )  # passing bridge-types-cat column (label encoded values of bridge_types)
    enc_df = pd.DataFrame(enc.fit_transform(encoded_ML[["special_mask"]]).toarray())
    encoded_ML = encoded_ML.join(enc_df)
    encoded_ML = encoded_ML.rename(columns={0: "mask-TP", 1: "mask-FP"})
    all_masks = ["special_mask", "encoded_mask", "mask-TP", "mask-FP"]
    target_columns = ["mask-TP", "mask-FP"]
    mask_to_drop = list(set(all_masks) - set(target_columns))
    # other_mask_df = encoded_ML[mask_to_drop]
    encoded_ML = encoded_ML.drop(mask_to_drop, axis=1)

    # find optimal features
    features = featurewiz(encoded_ML, target=target_columns, corr_limit=0.70, verbose=2)
    tuple1, tuple2 = features

    featurecorr = tuple2.corr()
    featurecorr.index = featurecorr.index.str.replace(r"_", " ")
    featurecorr.columns = featurecorr.columns.str.replace(r"_", " ")

    feat_labels = tuple1
    print(f"\n\nNo. features + labels: {len(feat_labels), feat_labels}\n\n")

    with open(
        f"{directory}/feature-wiz-features-OCS-{vees}-BLAST.json", "w", encoding="utf-8"
    ) as f:
        json.dump(feat_labels, f)

    mask = np.zeros_like(featurecorr)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(featurecorr, mask=mask)
    # ax = sns.heatmap(featurecorr)
    title = "Heat map of featurewiz OneClassSVM important features against the special mask - BLAST"
    newline_title = clean_strings(title)
    plt.title(newline_title, size=40)
    plt.tight_layout()
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=30)

    plt.savefig(f"{directory}/{title}-AA-check.png", dpi=300, bbox_inches="tight")
    plt.show()

    return feat_labels


def combine_reads(id_cols: list, df: pd.DataFrame, BLAST_cols: list) -> pd.DataFrame:
    mean_reads = {}
    blast_dict = {}
    # n = 0
    for name, group in df.groupby(id_cols):
        if len(group) > 1:
            # number_duplicates = len(group)
            reads = []
            blast_data = []
            # split df into index and rows
            for row_index, row in group.iterrows():
                # find all read values for entire group
                reads.append(row["length_count"])
                # extract blast rows
                values = row.loc[BLAST_cols]
                blast_data.append(values)
            blast_dict[name] = blast_data

            # remove 0s from the list
            reads = [int(math.ceil(i)) for i in reads if i != 0]
            # calculate mean of non-zero values
            mean_group = statistics.mean(reads)
            mean_group = int(math.ceil(mean_group))
            mean_reads[name] = [mean_group]
            intermed_df = pd.DataFrame(blast_data)
            intermed_mean = pd.DataFrame(intermed_df.mean()).T
            blast_dict[name] = intermed_mean
        else:
            mean_reads[name] = [int(group["length_count"])]
            blast_dict[name] = group[BLAST_cols]

    # generate all dfs separately
    mean_reads_df = pd.DataFrame.from_dict(
        mean_reads, orient="index", columns=["reads"]
    )
    mean_reads_df.index = pd.MultiIndex.from_tuples(mean_reads_df.index)

    blast_dict_df = pd.concat(blast_dict)
    blast_dict_df.index = blast_dict_df.index.droplevel(8)

    # combine into single df
    data_frames = [mean_reads_df, blast_dict_df]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        data_frames,
    )
    df_merged = df_merged.reset_index()
    df_merged = df_merged.rename(
        columns={
            "level_0": "date",
            "level_1": "NA",
            "level_2": "strain",
            "level_3": "concentration_CFU",
            "level_4": "batch",
            "level_5": "duration_h",
            "level_6": "name",
            "level_7": "sseqid",
        }
    )
    return df_merged


def main(
    species: dict,
    ML_df: pd.DataFrame,
    clean_drop_cols: list,
    std_cols: list,
    directory: str,
    BLASTn_name: str,
    vees: str,
) -> (pd.DataFrame, int, str, list):
    print(f"\n\n\n######### BLAST #########")
    ML_df = ML_df.drop(std_cols, axis=1)

    # vs = []
    if isinstance(species, dict):
        true_masks = []
        ML_df["mask"] = "Not_TP"
        for k, vv in species.items():
            if isinstance(vv, list):
                for v in vv:
                    print(vv, v)
                    true_mask = ML_df.loc[
                        (ML_df.strain.str.contains(v)) & (ML_df.name.str.contains(k))
                    ].index
                    ML_df.loc[true_mask, "mask"] = "True_positive"
                    true_masks.append(true_mask)
            else:
                true_mask = ML_df.loc[
                    (ML_df.strain.str.contains(vv)) & (ML_df.name.str.contains(k))
                ].index
                ML_df.loc[true_mask, "mask"] = "True_positive"
                true_masks.append(true_mask)
        true_positives = ML_df.loc[ML_df["mask"].str.contains("True_positive")]
        # vs.append(vv)
    # vees = "-".join(vs)

    positives = len(true_positives) / len(ML_df)
    print(positives, len(true_positives))

    if isinstance(species, dict):
        false_positives = ML_df.loc[ML_df["mask"] == "Not_TP"]

    negatives = len(false_positives) / len(ML_df)
    print(negatives, len(false_positives), "\n")

    inputs = f"\nStarting true positives (not broken down by species): {len(true_positives)}/{len(ML_df)} \
          \nPositive population: {positives*100:.2f}%; \
          \nNegative population: {negatives*100:.2f}%. \
          \nTotal predictions: {len(ML_df)}\n"

    true_positives_clean = true_positives.drop(clean_drop_cols, axis=1)
    false_positives_clean = false_positives.drop(clean_drop_cols, axis=1)

    feat_labels = featurewiz_for_fewer_features(
        directory, true_positives_clean, false_positives_clean, vees
    )
    # out = f"{directory}/OneClassSVM/"

    true_positives_clean = true_positives_clean[feat_labels]
    false_positives_clean = false_positives_clean[feat_labels]

    indices = false_positives_clean.index.values

    X_train, X_test, indices_train, indices_test = train_test_split(
        false_positives_clean, indices, test_size=0.2, random_state=736
    )

    # # model specification
    classifier = OneClassSVM(kernel="rbf", gamma=0.1, nu=positives)
    classifier.fit(X_train)

    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    y_pred_outliers = classifier.predict(true_positives_clean)

    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # save the model ##############################################
    filename_save = (
        f"OCS_describe_group_mean_qscores_hsblastn_{BLASTn_name}_spp_model.sav"
    )
    OCS_model_save = f"{directory}/{filename_save}"
    print(f"Saving one class SVM model to: {OCS_model_save}")
    pickle.dump(classifier, open(OCS_model_save, "wb"))
    ###############################################################

    train = clean_out(indices_train, y_pred_train, false_positives)
    train["ML_data"] = "train"
    test = clean_out(indices_test, y_pred_test, false_positives)
    test["ML_data"] = "test"

    true_positives["OCS"] = y_pred_outliers
    all_out = train.append(test)
    true_positives["ML_data"] = "true_positive"
    all_out = all_out.append(true_positives)

    further_testing = all_out.loc[all_out.OCS == -1]

    cfts = further_testing.reset_index(drop=True)

    cfts_groups = list(
        cfts.groupby(
            ["concentration_CFU", "date", "NA", "strain", "batch", "duration_h"]
        )
    )
    top_ten_predictions = []
    for cft in cfts_groups:
        sample = cft[1]
        sample = sample.nlargest(10, "length_count")
        top_ten_predictions.append(sample)
    cfts_out = pd.concat(top_ten_predictions)
    cfts_out = cfts_out.reset_index(drop=True)

    if isinstance(species, dict):
        true_positives_check = cfts_out.loc[
            cfts_out["mask"].str.contains("True_positive")
        ]
        false_positives_check = cfts_out.loc[cfts_out["mask"].str.contains("Not_TP")]

    positives_out = len(true_positives_check) / len(cfts_out)

    negatives_out = len(false_positives_check) / len(cfts_out)
    outputs = f"Actual true positives: {len(true_positives_check['sample'].unique())}/{len(true_positives['sample'].unique())} \
          \nPositive population: {positives_out*100:.2f}%; \
          \nNegative population: {negatives_out*100:.2f}%. \
          \nTotal predictions: {len(cfts_out)} \
          \nIncorrect predictions: {n_error_outliers}"

    print(
        f"One Class SVM analysis \
            \nSample input: species: {species} \
            {inputs}\n{outputs}"
    )

    print(
        f"{directory}/hsblastn_describe_group_mean_summary_parameters_{BLASTn_name}_{vees}.txt"
    )
    with open(
        f"{directory}/hsblastn_describe_group_mean_summary_parameters_{BLASTn_name}_{vees}.txt",
        "w",
    ) as text_file:
        text_file.write(
            f"One Class SVM analysis \
                        \nSample input: species: {species} - BLASTn_name: {BLASTn_name} \
                        {inputs}\n{outputs}"
        )
    return cfts_out, n_error_outliers, OCS_model_save, feat_labels
