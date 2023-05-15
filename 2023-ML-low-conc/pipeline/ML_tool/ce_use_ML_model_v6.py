#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:25:33 2021

@author: mangi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# from sklearn.svm import OneClassSVM
from featurewiz import featurewiz
import matplotlib.pyplot as plt
import seaborn as sns
from string import whitespace
from pathlib import Path
import json
import pickle
import os

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


def main(
    species: dict,
    ML_df: pd.DataFrame,
    clean_drop_cols: list,
    std_cols: list,
    directory: str,
    kingdom: str,
    ml_model_file: str,
    feat_labels: list,
    vees: str,
    save_loc: str,
    vees_unknown: str,
) -> (pd.DataFrame, int):
    print(f"\n\n\n######### Centrifuge USE #########")
    ML_df = ML_df.drop(std_cols, axis=1)

    ML_df["mask"] = "Not_TP"

    if isinstance(species, dict):
        false_positives = ML_df.loc[ML_df["mask"] == "Not_TP"]

    negatives = len(false_positives) / len(ML_df)
    print(negatives, len(false_positives), "\n")

    false_positives_clean = false_positives.drop(clean_drop_cols, axis=1)

    if len(feat_labels) == 0:
        with open(f"{directory}/feature-wiz-features-OCS-{vees}-centrifuge.json") as json_fw:
            feat_labels = json.loads(json_fw.read())
    print(f"Featurewiz labels: {feat_labels}")

    if len(vees_unknown) > 0:
        vees = f"{vees}{vees_unknown}"

    false_positives_clean = false_positives_clean[feat_labels]

    indices = false_positives_clean.index.values

    n_error_outliers = 0
    if Path(ml_model_file).is_file():
        loaded_model = pickle.load(open(ml_model_file, "rb"))
        print("\nOCS - prebuilt model")

        y_pred_eval = loaded_model.predict(false_positives_clean)

        evaluate = clean_out(indices, y_pred_eval, false_positives)
        evaluate["ML_data"] = "eval"

        further_testing = evaluate.loc[evaluate.OCS == -1]

        cfts = further_testing.reset_index(drop=True)

        print(f"Number of anomalous species for centrifuge: {len(cfts)}")

        cfts_groups = list(cfts.groupby(["sample", "db"]))
        top_ten_predictions = []
        for cft in cfts_groups:
            sample = cft[1]
            sample = sample.nlargest(10, "numUniqueReads")
            top_ten_predictions.append(sample)
        cfts_out = pd.concat(top_ten_predictions)
        cfts_out = cfts_out.reset_index(drop=True)

        if isinstance(species, dict):
            false_positives_check = cfts_out.loc[
                cfts_out["mask"].str.contains("Not_TP")
            ]

        negatives_out = len(false_positives_check) / len(cfts_out)
        outputs = f"Negative population: {negatives_out*100:.2f}%. \
              \nTotal predictions: {len(cfts_out)} \
              \nIncorrect predictions: {n_error_outliers}"

        print(
            f"One Class SVM analysis \
                \nSample input: species: {species} - kingdom: {kingdom} \
                \n{outputs}"
        )

        print(
            f"{save_loc}/centrifuge_describe_summary_parameters_{kingdom}_{vees}_evaluation.txt"
        )
        with open(
            f"{save_loc}/centrifuge_describe_summary_parameters_{kingdom}_{vees}_evaluation.txt",
            "w",
        ) as text_file:
            text_file.write(
                f"One Class SVM analysis \
                            \nSample input: species: {species} - kingdom: {kingdom} \
                            \n{outputs}"
            )
        return cfts_out, n_error_outliers
