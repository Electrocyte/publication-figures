#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:09:12 2021

@author: mangi
"""

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from string import whitespace
import statistics
import json
import pickle
import matplotlib.pyplot as plt
from featurewiz import featurewiz
import seaborn as sns
from functools import partial
from pathlib import Path


def clean_strings(string: str, cut_off: int) -> str:
    # from string import whitespace
    # string = "Cumulative frequency for detection of MVM reads per minute for batch19"
    len_str = len(string)
    mid_point = int(len_str / 2)
    whitespaces = [i for i, char in enumerate(string) if char in whitespace]
    cut_string = min(whitespaces, key=lambda x: abs(x - mid_point))
    new_string = f"{string[0:cut_string]}\n{string[cut_string:]}"

    subsplits = ""
    if len_str > cut_off:
        split_again = new_string.split("\n")
        for new_str in split_again:
            subsplit = clean_strings(new_str, cut_off)
            subsplits += subsplit

    if len(subsplits) > 0:
        len_str = len(subsplits)
        mid_point = int(len_str / 2)
        whitespaces = [i for i, char in enumerate(subsplits) if char in whitespace]
        cut_string = min(whitespaces, key=lambda x: abs(x - mid_point))
        new_string = f"{subsplits[0:cut_string]}\n{subsplits[cut_string:]}"

    return new_string


def custom_label_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if name in row["sample_true_mask"]:
            return encoded


def one_hot_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if encoded == row["sample_encoded_mask"]:
            if encoded == 0:
                return (1, 0)
            if encoded == 1:
                return (0, 1)
            else:
                return (0, 0)


def h(x):
    return True if x["sample_true_mask"] == x["XGB_sample_prediction"] else False


def check_true_mask_sample(x):
    return True if x["sample_true_mask"] == x["XGB_sample_pred"] else False


def j(x):
    return True if x["Decision_mask"] == x["Decision"] else False


def make_decision(row):
    if (
        row["XGB_AA_prediction"] == "True_positive"
        and row["XGB_sample_prediction"] == "True_positive"
    ):
        return "True_positive"
    if (
        row["XGB_AA_prediction"] == "True_negative"
        and row["XGB_sample_prediction"] == "True_positive"
    ):
        return "False_negative"
    if (
        row["XGB_AA_prediction"] == "Background"
        and row["XGB_sample_prediction"] == "True_positive"
    ):
        return "False_positive"
    if (
        row["XGB_AA_prediction"] == "Background"
        and row["XGB_sample_prediction"] == "True_negative"
    ):
        return "False_positive"
    if (
        row["XGB_AA_prediction"] == "False_positive"
        and row["XGB_sample_prediction"] == "True_negative"
    ):
        return "True_negative"
    if (
        row["XGB_AA_prediction"] == "False_positive"
        and row["XGB_sample_prediction"] == "True_positive"
    ):
        return "False_positive"
    if (
        row["XGB_AA_prediction"] == "True_positive"
        and row["XGB_sample_prediction"] == "True_negative"
    ):
        return "False_negative"


def decision_mask(row):
    if row["mask"] == "True_positive" and row["sample_true_mask"] == "True_positive":
        return "True_positive"
    if row["mask"] == "True_negative" and row["sample_true_mask"] == "True_positive":
        return "False_negative"
    if row["mask"] == "Background" and row["sample_true_mask"] == "True_positive":
        return "False_positive"
    if row["mask"] == "Background" and row["sample_true_mask"] == "True_negative":
        return "False_positive"
    if row["mask"] == "False_positive" and row["sample_true_mask"] == "True_negative":
        return "True_negative"
    if row["mask"] == "False_positive" and row["sample_true_mask"] == "True_positive":
        return "False_positive"
    if row["mask"] == "True_positive" and row["sample_true_mask"] == "True_negative":
        return "False_negative"


def sample_decision(row):
    # TPs, FPs, TNs, FNs
    if (
        not pd.isna(row["False_positive"])
        and not pd.isna(row["False_negative"])
        and not pd.isna(row["True_negative"])
        and not pd.isna(row["True_positive"])
    ):
        return "Sample contains a lot of noise, could be sterile or contaminated"
    # FPs
    elif (
        pd.isna(row["False_negative"])
        and pd.isna(row["True_negative"])
        and pd.isna(row["True_positive"])
    ):
        return "Sterile, background detected"
    # FNs
    elif (
        pd.isna(row["False_positive"])
        and pd.isna(row["True_negative"])
        and pd.isna(row["True_positive"])
    ):
        return "Contaminated"
    # TNs
    elif (
        pd.isna(row["False_positive"])
        and pd.isna(row["False_negative"])
        and pd.isna(row["True_positive"])
    ):
        return "Sterile"
    # TPs
    elif (
        pd.isna(row["False_positive"])
        and pd.isna(row["False_negative"])
        and pd.isna(row["True_negative"])
    ):
        return "Contaminated"
    # TNs, FPs
    elif pd.isna(row["False_negative"]) and pd.isna(row["True_positive"]):
        return "Sterile, background detected"
    # TPs, FPs
    elif pd.isna(row["False_negative"]) and pd.isna(row["True_negative"]):
        return "Potential contamination"
    # FNs, TNs
    elif pd.isna(row["False_positive"]) and pd.isna(row["True_positive"]):
        if row["False_negative"] - row["True_negative"] > -5:
            return "Potential contamination"
        else:
            return "Sterile, background detected"
    # FNs, TPs
    elif pd.isna(row["False_positive"]) and pd.isna(row["True_negative"]):
        return "Contaminated"
    # FNs, FPs, TNs
    elif (
        pd.isna(row["True_positive"])
        and not pd.isna(row["False_positive"])
        and not pd.isna(row["True_negative"])
        and not pd.isna(row["False_negative"])
    ):
        return "Sterile, background detected"
    # TPs, FPs, TNs
    elif (
        pd.isna(row["False_negative"])
        and not pd.isna(row["False_positive"])
        and not pd.isna(row["True_negative"])
        and not pd.isna(row["True_positive"])
    ):
        return "Contaminated"
    # TPs, FPs, FNs
    elif (
        pd.isna(row["True_negative"])
        and not pd.isna(row["False_positive"])
        and not pd.isna(row["False_negative"])
        and not pd.isna(row["True_positive"])
    ):
        return "Potential contamination"
    # TPs, FPs, FNs
    elif pd.isna(row["True_negative"]) and pd.isna(row["True_positive"]):
        return "Inconclusive"
    else:
        return "Unknown"


def check_tts_status(row):
    if row["Dec_Correct"] == True and row["XGBoost-tts-decision"] == "eval":
        return "TRUE-eval"
    if row["Dec_Correct"] == False and row["XGBoost-tts-decision"] == "eval":
        return "FALSE-eval"


def main(
    save: bool,
    directory: str,
    json_file: str,
    species: dict,
    eval_out: str,
    featurewiz_XGB: bool,
    vees,
    XGB_out: str,
    centrifuge_db: str,
) -> None:
    ##############################################################################
    # DATA CLEANING #
    ##############################################################################
    pd.options.mode.chained_assignment = None  # default='warn'
    filename_save = "xgboost_check_decision_v6.sav"
    ##############################################################################
    # DATA CLEANING #
    #############################################################################

    xgboost_csv_filename = (
        f"XGBoost-per-contaminant-predictions-for-{vees}-{centrifuge_db}.csv"
    )
    ML_data_csv = (
        f"data-for-XGBoost-per-contaminant-predictions-for-{vees}-{centrifuge_db}.csv"
    )
    print(f"{eval_out}/{xgboost_csv_filename}")
    location = glob.glob(f"{eval_out}/{xgboost_csv_filename}")[0]
    ml_location = glob.glob(f"{eval_out}/{ML_data_csv}")[0]

    df_initial = pd.read_csv(location)
    ml_df_xgboost = pd.read_csv(ml_location)

    with open(json_file) as json_data:
        true_mask = json.loads(json_data.read())
    #################### IMPORT DATA ####################

    ##############################################################################
    # DATA CLEANING #
    ##############################################################################
    # sample_true_mask refers to an individual sample, NOT A PREDICTED SPECIES
    true_masked_dfs = []
    for k, v in true_mask.items():
        true_mask_df = pd.DataFrame.from_dict(
            v, columns=["sample_true_mask"], orient="index"
        )
        true_mask_df["db"] = k
        true_masked_dfs.append(true_mask_df)
    cat_true_mask_df = pd.concat(true_masked_dfs)
    cat_true_mask_df.reset_index(inplace=True)
    cat_true_mask_df = cat_true_mask_df.rename(columns={"index": "sample"})
    cat_true_mask_df = cat_true_mask_df.replace("CFU", "", regex=True)
    true_mask_df_merge = pd.merge(
        df_initial,
        cat_true_mask_df,
        how="left",
        left_on=["sample", "db"],
        right_on=["sample", "db"],
    )

    true_mask_df_merge = true_mask_df_merge.drop(
        true_mask_df_merge.index[
            true_mask_df_merge.loc[
                true_mask_df_merge["sample_true_mask"] == "False_positive"
            ].index
        ]
    )
    true_mask_df_merge.reset_index(inplace=True, drop=True)

    ml_df_xgboost.reset_index(inplace=True, drop=True)
    ml_df_xgboost = ml_df_xgboost.drop(["sample", "name", "mask", "db"], axis=1)
    data_merge = true_mask_df_merge.merge(
        ml_df_xgboost, how="outer", left_index=True, right_index=True
    )

    ML_drop_cols = [
        "sample",
        "name",
        "mask",
        "db",
        "XGB_AA_prediction",
        "Correct",
        "XGBoost-tts",
        "check_tts_status",
    ]
    ##############################################################################

    ################## ADJUST FOR SKEW ##################
    # Compute class weights with sklearn method
    def generate_class_weights(class_series):
        class_labels = np.unique(class_series)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=class_labels, y=class_series
        )
        return dict(zip(class_labels, class_weights))

    # remove NaN samples
    stm_idx = (
        data_merge["sample_true_mask"]
        .loc[~pd.isna(data_merge["sample_true_mask"])]
        .index
    )
    data_merge = data_merge.iloc[stm_idx]
    data_merge.reset_index(inplace=True, drop=True)

    class_weights_ = generate_class_weights(data_merge["sample_true_mask"])
    with open(
        f"{eval_out}/class-weights-XGB-decision-use-{centrifuge_db}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(class_weights_, f)

    def adjust_for_class_weights(class_weights_) -> (str, int):
        # min_key = ""
        min_val = 0
        final_key = ""
        final_min = 0
        for k, v in class_weights_.items():
            if v < min_val:
                final_key = k
                final_min = v
            else:
                # min_key = k
                min_val = v
        return final_key, final_min

    final_key, final_min = adjust_for_class_weights(class_weights_)
    ################## ADJUST FOR SKEW ##################

    # ###################### USING TRAIN TEST SPLIT ######################
    ML_reduced = data_merge[["name", "sample", "db"]]
    ML_df = data_merge.drop(ML_drop_cols, axis=1)

    ######## Feature WIZ ########

    ######## Test encoding ########

    encoded_ML = ML_df

    encoding_dict = {"True_positive": 0, "True_negative": 1}
    target_columns = ["sample-mask-TP", "sample-mask-TN"]

    label_partial_func = partial(custom_label_encoder, encoding_dict)
    encoded_ML["sample_encoded_mask"] = encoded_ML.apply(label_partial_func, axis=1)

    enc_partial_func = partial(one_hot_encoder, encoding_dict)
    encoded_ML["temp-mask"] = encoded_ML.apply(enc_partial_func, axis=1)
    encoded_ML[target_columns] = pd.DataFrame(
        encoded_ML["temp-mask"].tolist(), index=encoded_ML.index
    )
    encoded_ML = encoded_ML.drop(["temp-mask"], axis=1)

    all_masks = [
        "sample_true_mask",
        "sample_encoded_mask",
        "sample-mask-TN",
        "sample-mask-TP",
    ]

    ######## Test encoding ########

    mask_to_drop = list(set(all_masks) - set(target_columns))
    other_mask_df = encoded_ML[mask_to_drop]
    encoded_ML = encoded_ML.drop(mask_to_drop, axis=1)

    with open(
        f"{XGB_out}/feature-wiz-xgboost-features-{vees}-decision.json"
    ) as json_sample_XGB:
        feat_labels = json.loads(json_sample_XGB.read())

    encoded_ML = encoded_ML[feat_labels + target_columns]

    fw_heatmap_df = encoded_ML
    human_readable = encoded_ML

    featurecorr = fw_heatmap_df.corr()
    featurecorr.index = featurecorr.index.str.replace(r"_", " ")
    featurecorr.columns = featurecorr.columns.str.replace(r"_", " ")

    if save:
        human_readable.to_csv(
            f"{eval_out}/feature-wiz-evaluation-data-XGB-AA-for-{vees}-decision.csv",
            index=False,
        )

    mask = np.zeros_like(featurecorr)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(featurecorr, mask=mask)
    title = f"Heat map of featurewiz important features against the true mask - {centrifuge_db} - XGBoost-decision"
    plt.title(title, size=40)
    plt.tight_layout()
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=30)
    if save:
        plt.savefig(f"{eval_out}/{title}-decision.png", dpi=300, bbox_inches="tight")
    plt.show()

    feature_wiz_df = pd.merge(
        ML_reduced, human_readable, how="left", left_index=True, right_index=True
    )
    feature_wiz_df = pd.merge(
        feature_wiz_df, other_mask_df, how="left", left_index=True, right_index=True
    )
    feature_wiz_df.reset_index(inplace=True, drop=True)

    # ######## USE Feature WIZ ########
    if featurewiz_XGB:
        y = feature_wiz_df["sample_encoded_mask"].values  # ***
        indices = feature_wiz_df.index.values
        feature_wiz_df = feature_wiz_df.drop(
            all_masks + ["name", "sample", "db"], axis=1
        )
        X = feature_wiz_df.iloc[:, :].values
    # ######## USE Feature WIZ ########

    ######## USE ALL features ########
    if not featurewiz_XGB:
        y = ML_df["sample_encoded_mask"].values  # ***
        indices = ML_df.index.values
        ML_df = ML_df.drop([all_masks + ["temp-mask"]], axis=1)
        X = ML_df.iloc[:, :].values
    ######## USE ALL features ########

    #### STANDARD SCALER APPROACH ####
    # Feature Scaling
    sc = pickle.load(open(f"{XGB_out}/sc-sample-XGB-scaler.pkl", "rb"))

    X_eval = sc.fit_transform(X)

    XGB_model_save = f"{XGB_out}/{filename_save}"
    if Path(XGB_model_save).is_file():
        XGB_classifier_model = pickle.load(open(XGB_model_save, "rb"))

        y_pred = XGB_classifier_model.predict(X_eval)

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y, y_pred)

        # Get predicted probabilities for each class
        y_preds_proba = XGB_classifier_model.predict_proba(X_eval)

        from sklearn.metrics import roc_curve

        try:
            with open(f"{XGB_out}/ROC-decision-XGB-check.json") as json_data:
                plotting_data = json.loads(json_data.read())
            auc_score2, auc_score3, fpr2, tpr2, fpr3, tpr3 = plotting_data
            fpr2, tpr2, fpr3, tpr3 = (
                np.array(fpr2),
                np.array(tpr2),
                np.array(fpr3),
                np.array(tpr3),
            )

            # roc curve for models
            fpr1, tpr1, thresh1 = roc_curve(y, y_preds_proba[:, 1], pos_label=1)

            random_probs = [0 for i in range(len(y))]
            p_fpr, p_tpr, _ = roc_curve(y, random_probs, pos_label=1)

            from sklearn.metrics import roc_auc_score

            # auc scores
            auc_score1 = roc_auc_score(y, y_preds_proba[:, 1])

            centrifuge_plot = centrifuge_db
            centrifuge_label = centrifuge_db
            if centrifuge_db == "v_f_b":
                centrifuge_plot = "viral, fungal and bacterial species database"
                centrifuge_label = "viral-fungal-bacterial"

            plt.style.use("seaborn")
            f, ax = plt.subplots(figsize=(15, 15))
            # plot roc curves
            plt.plot(
                fpr1,
                tpr1,
                linestyle="--",
                color="red",
                label=f"XGB-eval-{centrifuge_label}",
            )
            plt.plot(fpr2, tpr2, linestyle="--", color="green", label="XGB-test")
            plt.plot(fpr3, tpr3, linestyle="--", color="orange", label="XGB-train")
            plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)
            # title
            title = f"ROC curve using the XGBoost Classifier for sample contaminant status identification with the {centrifuge_plot}"
            new_title = clean_strings(title, 120)
            plt.title(new_title, size=40)
            # x label
            plt.xlabel("False Positive Rate", size=40)
            # y label
            plt.ylabel("True Positive rate", size=40)

            plt.text(
                0.8,
                0.45,
                f"AUC-eval: {auc_score1:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                size=30,
            )
            plt.text(
                0.8,
                0.4,
                f"AUC-test: {auc_score2:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                size=30,
            )
            plt.text(
                0.8,
                0.35,
                f"AUC-train: {auc_score3:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                size=30,
            )

            plt.legend(loc=4, prop={"size": 30}, markerscale=10)
            plt.savefig(f"{eval_out}/ROC-sample-XGB-use-{centrifuge_db}.png", dpi=300)
            plt.show()

        except:
            print("Missing examples of class X, cannot plot")

        # 5 folds, scored on accuracy
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        cvs2 = cross_val_score(
            XGB_classifier_model, X_eval, y, cv=5, scoring="accuracy"
        )

        from sklearn.metrics import classification_report, confusion_matrix

        print(
            "-------------------------------------------------------------------------------"
        )
        print(
            f"XGBoost - decision - Evaluation or New DATA for {centrifuge_db}; {encoding_dict}"
        )
        print("  ")
        print(confusion_matrix(y, y_pred))  # ***
        print(classification_report(y, y_pred))  # Output
        print(
            "-------------------------------------------------------------------------------"
        )
        print(f"Cross-val #2 mean: {statistics.mean(cvs2):.2f}")
        print(f"Cross-val #2 stdev: {statistics.stdev(cvs2):.2f}")

        AA_sample = data_merge[
            ["sample", "name", "mask", "XGB_AA_prediction", "sample_true_mask", "db"]
        ]
        AA_sample.loc[indices, "XGB_sample_prediction"] = XGB_classifier_model.predict(
            X_eval
        )

        encoding_dict_rev = {value: key for (key, value) in encoding_dict.items()}
        AA_sample = AA_sample.replace({"XGB_sample_prediction": encoding_dict_rev})

        AA_sample["Correct"] = AA_sample.apply(h, axis=1)

        if save:
            AA_sample.to_csv(
                f"{eval_out}/xgboost-per-sample-predictions-for-{vees}.csv", index=False
            )

        AA_sample["Decision"] = AA_sample.apply(make_decision, axis=1)
        AA_sample["Decision_mask"] = AA_sample.apply(decision_mask, axis=1)
        AA_sample["Dec_Correct"] = AA_sample.apply(j, axis=1)
        if save:
            AA_sample.to_csv(f"{eval_out}/Decision-with-mask.csv", index=False)

        decision_df = (
            AA_sample.groupby(["sample", "db", "Decision"]).size().reset_index()
        )
        decision_pivot = decision_df.pivot_table(0, ["sample", "db"], "Decision")
        if "False_positive" not in decision_pivot.columns:
            decision_pivot["False_positive"] = np.nan
        if "False_negative" not in decision_pivot.columns:
            decision_pivot["False_negative"] = np.nan
        if "True_negative" not in decision_pivot.columns:
            decision_pivot["True_negative"] = np.nan()
        if "True_positive" not in decision_pivot.columns:
            decision_pivot["True_positive"] = np.nan

        decision_pivot["Sterility"] = decision_pivot.apply(sample_decision, axis=1)
        decision_pivot.reset_index(inplace=True)

        if save:
            decision_pivot.to_csv(
                f"{eval_out}/Pivot-decision-table-{centrifuge_db}.csv", index=False
            )

        AA_sample.loc[indices, "XGBoost-tts-decision"] = "eval"
        AA_sample["check-tts-status-decision"] = AA_sample.apply(
            check_tts_status, axis=1
        )

        cb_performance = AA_sample["check-tts-status-decision"].value_counts()
        if save:
            AA_sample.to_csv(f"{eval_out}/decision_with_tts_mask.csv", index=False)

        true_test = cb_performance[cb_performance.index.str.startswith("TRUE-eval")]
        if len(true_test) > 0:
            true_test = true_test[0]
        else:
            true_test = 0

        false_test = cb_performance[cb_performance.index.str.startswith("FALSE-eval")]
        if len(false_test) > 0:
            false_test = false_test[0]
        else:
            false_test = 0

        print(
            f"Correct test-case predictions: {true_test/(true_test+false_test)*100:.2f}%"
        )

        positives = len(AA_sample.loc[AA_sample["Correct"] == True])

        print(
            f"Positives (train+test) correctly predicted: {positives}/{len(AA_sample)}; {positives/len(AA_sample)*100:.2f} %"
        )

        ###################### USING TRAIN TEST SPLIT ######################

        # # VISUALISATION ONLY

        # ########## cleaning ##########

        AA_sample = AA_sample.drop(["mask"], axis=1)
        drop_cols = ["sample", "name", "mask", "db", "sample_true_mask"]
        post_data_merge = data_merge.drop(
            [
                "sample",
                "name",
                "db",
                "XGB_AA_prediction",
                "Correct",
                "XGBoost-tts",
                "check_tts_status",
                "sample_true_mask",
            ],
            axis=1,
        )
        if save:
            post_data_merge.to_csv(f"{eval_out}/data-for-decision-LDA.csv", index=False)
        original_lda_df = AA_sample.merge(
            post_data_merge, how="outer", left_index=True, right_index=True
        )

        original_lda_df["XGB_sample_prediction"]
        indices = original_lda_df.index.values

        try:
            lda_df = original_lda_df.drop(
                drop_cols
                + [
                    "Decision",
                    "Decision_mask",
                    "Dec_Correct",
                    "XGBoost-tts-decision",
                    "check-tts-status-decision",
                    "XGB_AA_prediction",
                    "Correct",
                ],
                axis=1,
            )
        except:
            lda_df = original_lda_df.drop(drop_cols, axis=1)

        # # place species name at end of df
        cols = lda_df.columns.tolist()
        cols = cols[1:] + cols[:1]
        lda_df = lda_df[cols]
        lda_df = lda_df.fillna(0)

        # remap labels
        # int_dict = {"True_positive": 1, "True_negative": 2}
        lda_df = lda_df.replace({"XGB_sample_prediction": encoding_dict})

        X = lda_df.iloc[:, :-1].values
        y = lda_df.iloc[:, -1].values

        # get the indices for the training set (note difference X2 is still a df)
        X2 = lda_df.iloc[:, :-1]
        X2_train, X2_test, indices_train, indices_test = train_test_split(
            X2, indices, test_size=0.2, random_state=736
        )

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, indices, test_size=0.2, random_state=736
        )
        y_train, y_test = y[indices_train], y[indices_test]

        L_sc = StandardScaler()
        X_train = L_sc.fit_transform(X_train)
        X_test = L_sc.transform(X_test)

        # Applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

        try:
            lda = LDA(n_components=1)
            X_train = lda.fit_transform(X_train, y_train)
            X_test = lda.transform(X_test)

            # Training the Logistic Regression model on the Training set
            from sklearn.linear_model import LogisticRegression

            classifier = LogisticRegression(random_state=736)
            classifier.fit(X_train, y_train)

            # Making the Confusion Matrix
            from sklearn.metrics import confusion_matrix, accuracy_score

            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(accuracy_score(y_test, y_pred))

            lda_results_train = pd.DataFrame(X_train, columns=["lda1"])
            lda_results_train.index = list(X2_train.index)

            lda_results_test = pd.DataFrame(X_test, columns=["lda1"])
            lda_results_test.index = list(X2_test.index)

            lda_combined = pd.concat([lda_results_train, lda_results_test])

            lda_df_updated = pd.merge(
                original_lda_df, lda_combined, left_index=True, right_index=True
            )
            if save:
                lda_df_updated.to_csv(
                    f"{eval_out}/LDA-per-sample-predictions-for-{vees}.csv", index=False
                )



            # find the most frequent prediction for whether the sample is a TN or TP
            majority_dict = {}
            for i, group in lda_df_updated.groupby(["sample", "db"])[
                "XGB_sample_prediction"
            ]:
                max_val = group.value_counts().max()
                samp_df = pd.DataFrame(group.value_counts() == max_val)
                samp_bool = samp_df.index[
                    samp_df["XGB_sample_prediction"] == True
                ].tolist()
                majority_dict[i] = samp_bool[0]
            XGB_decision = pd.DataFrame.from_dict(
                majority_dict, columns=["XGB_sample_pred"], orient="index"
            )
            mean_lda = lda_df_updated.groupby(["sample", "db"])["lda1"].mean()
            STM = lda_df_updated[["sample", "db", "sample_true_mask"]].drop_duplicates()
            sSTM = STM.set_index(["sample", "db"])
            summ_LDA = pd.concat([XGB_decision, mean_lda, sSTM], axis=1).reset_index()
            summ_LDA["Correct"] = summ_LDA.apply(check_true_mask_sample, axis=1)
            if save:
                summ_LDA.to_csv(
                    f"{eval_out}/LDA-per-sample-only-predictions-for-{vees}.csv",
                    index=False,
                )

            hue_order = ["True_negative", "True_positive"]

            f, ax = plt.subplots(figsize=(15, 15))

            hue = "XGB_sample_prediction"

            # Draw a scatterplot where one variable is categorical.
            sns.stripplot(
                x="sample_true_mask",
                y="lda1",
                hue=lda_df_updated[hue],
                palette=sns.color_palette(
                    "bright", n_colors=len(lda_df_updated[hue].unique())
                ),
                data=lda_df_updated,
                s=10,
                order=hue_order,
                hue_order=hue_order,
            )
            sns.stripplot(
                x="sample_true_mask",
                y="lda1",
                hue=summ_LDA["XGB_sample_pred"],
                palette=sns.color_palette(
                    "dark", n_colors=len(summ_LDA["XGB_sample_pred"].unique())
                ),
                data=summ_LDA,
                s=15,
                order=hue_order,
                hue_order=hue_order,
            )

            ax.set_xlabel("Sample prediction", size=40)
            ax.set_ylabel("LD1", size=40)

            title = "LDA BLAST analysis for original dataset - dimensional reduction ONLY (SAMPLE) - xgboost"
            newline_title = clean_strings(title)
            plt.title(newline_title, size=40)
            plt.tight_layout()
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)

            ax.set_ylim([-15, 15])

            plt.legend(
                loc=1,
                prop={"size": 25},
                markerscale=3,
                labels=["True negative", "True positive"],
            )  # , bbox_to_anchor=(1.05, 1))

            if save:
                print(f"{eval_out}/{title}.png")
                plt.savefig(f"{eval_out}/{title}.png", dpi=300, bbox_inches="tight")
            plt.show()
        except:
            pass
