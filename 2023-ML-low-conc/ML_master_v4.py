#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:35:31 2022

@author: mangi
"""

import pandas as pd
import numpy as np
import time
import json
import os
from collections.abc import Iterable
from pipeline import xgboost_data_cleaning, ML_assessment, PCA

from sklearn.model_selection import train_test_split
from functools import partial
from featurewiz import featurewiz
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from string import whitespace
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import statistics
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from sklearn.metrics import accuracy_score

################## PARAMETERS ##################

# DATA INPUT

# Data augmentation
data_aug = True
gauss_noise = True
transforms = False

# model tweaking - feature selection, gridsearch, reg
regularisation = True
featurewiz_XGB = True
run_gridsearch = False
run_randomsearch = False

# for best results
data_cleaning = True
feature_engineering = True
run_models = True

v_f_b = False
v_f_b2 = True

# # SUBSET SAMPLES
subset = True
independent_var = "_aDNA_" # amplicon

# required for flow cell subsetting
sample_data_file = "aDNA_all.csv"
flow_cell_file = "FlowCellMetaData.csv"
first_flowcell = False
no_sample_prep_test = False

# INDIVIDUAL EXPERIMENT INPUTS - SPIKE SPECIES, LABELS, TRAINING DATABASE, SAMPLES TO EXCLUDE FOR EVALUATION.
directory = "D:/SequencingData/Harmonisation/DNA/analysis"
github = "D:/GitHub/SMART-CAMP/"
map_files = "D:/SequencingData/Centrifuge_libraries/v_f_b2/"

save = True
json_mask = "catboost_decision_true_mask_bact_fungal-human-dictionary-824.json"
    
if v_f_b:
    species = {"Pseudomonas aeruginosa":["PA"], "Cutibacterium acnes":["Cacnes","Pacnes"], \
                "Escherichia coli":["EC"], "Klebsiella pneumoniae":["Klebpneu"], \
                 "Candida ":["Calbicans"], "Staphylococcus aureus":["Saureus"], \
                   "Bacillus subtilis": ["Bsubtilis"]}
    database_dict = {"v_f_b":"v_f_b"}
    
if v_f_b2:
    species = {"Pseudomonas aeruginosa":["PA"], "Cutibacterium acnes":["Cacnes","Pacnes"], \
                "Escherichia coli":["EC"], "Klebsiella pneumoniae":["Klebpneu"], \
                 "Candida albicans":["Calbicans"], "Staphylococcus aureus":["Saureus"], \
                   "Bacillus subtilis": ["Bsubtilis"]}
    database_dict = {"v_f_b2":"v_f_b2"}

# DATABASE PARAMETERS
epoch_time = str(int(time.time()))

samples_to_subset = []

################## SETUP COMPLETE ##################

_ML_out_ = f"{directory}/ML_training-VFB/"
out_OCS = f"{_ML_out_}/OneClassSVM-VFB/"
cat_out = f"{_ML_out_}/OCS-catboost-VFB/"
XGB_out = f"{_ML_out_}/OCS-XGBoost-VFB/"
os.makedirs(_ML_out_, exist_ok=True)
os.makedirs(cat_out, exist_ok=True) 
os.makedirs(XGB_out, exist_ok=True)   
os.makedirs(out_OCS, exist_ok=True)

BLASTn_name = list(database_dict.keys())[0]
kingdom = list(database_dict.values())[0]

################## PARAMETERS ##################
#
#
#
################## DATA CLEANING ##################

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

vs = []
for k,v in species.items():
    vs.append(v)
vs = list(flatten(vs))
vees = "-".join(vs)


hd_data_cols, \
cd_data_cols, \
    cn_df, \
    bn_df = \
        xgboost_data_cleaning.main(
        directory,
        BLASTn_name,
        kingdom,
        species, 
        _ML_out_,
        subset,
        independent_var,
        github,
        map_files)
        
################## DATA CLEANING ##################
#
#
#
################## FEATURE ENGINEERING / SELECTION ##################

# add mask for TP & TN vs other
def apply_mask(df: pd.DataFrame, species: dict) -> pd.DataFrame:
    df["strain"] = df["sample"].str.split("_", expand=True)[2]

    true_masks = []
    for k, vv in species.items():
        if isinstance(vv, list):
            for v in vv:
                true_mask = df.loc[
                    (df.strain.str.contains(v)) & (df.name.str.contains(k))
                ].index
                df.loc[true_mask, "mask"] = "True_positive"
                true_masks.append(true_mask)
        else:
            true_mask = df.loc[
                (df.strain.str.contains(vv)) & (df.name.str.contains(k))
            ].index
            df.loc[true_mask, "mask"] = "True_positive"
            true_masks.append(true_mask)
    true_masks = list(set([item for sublist in true_masks for item in sublist]))

    false_mask = list(set(list(df.index)) - set(true_masks))
    df.loc[false_mask, "mask"] = "False_positive"
    df.loc[true_masks, "mask"] = "True_positive"
    return df


def clean_sample_prep(df: pd.DataFrame) -> pd.DataFrame:
    sample_prep_tests = [x for x in df["sample"].unique() if "ITS" in x or "23S" in x]
    df = df.loc[~df["sample"].isin(sample_prep_tests)]
    df.reset_index(inplace = True, drop = True)
    return df


def get_flow_cell_info(github: str, 
                       sample_data_file: str,
                       flow_cell_file: str,
                       df: pd.DataFrame) -> pd.DataFrame:
    all_sample_original = pd.read_csv(f"{github}/configs/{sample_data_file}")
    flow_cell_data = pd.read_csv(f"{github}/configs/{flow_cell_file}")
    first_use = flow_cell_data["Flow-cell"].drop_duplicates(keep='first').index
    first_use_df = flow_cell_data.iloc[first_use]

    unique_run = set(all_sample_original["Identifier"])
    retain_first = set(first_use_df["FileID"])
    kept_run = list(set.intersection(unique_run, retain_first))
    kept_run = [x.replace("S", "_") for x in kept_run]
    kept_run = [x.replace("B", "") for x in kept_run]
    subset_df = df.loc[df["sample"].str.contains('|'.join(kept_run))]
    subset_df.reset_index(inplace = True, drop = True)
    return subset_df


masked_ce = apply_mask(cn_df, species)
masked_bl = apply_mask(bn_df, species)

# 1. Only use first run on flow cell

if first_flowcell:
    masked_ce = get_flow_cell_info(github, sample_data_file, flow_cell_file, masked_ce) 
    masked_bl = get_flow_cell_info(github, sample_data_file, flow_cell_file, masked_bl) 

# 2. Remove all sample prep "test" samples

if no_sample_prep_test:
    masked_ce = clean_sample_prep(masked_ce)
    masked_bl = clean_sample_prep(masked_bl)

# 3. Only use gauss values??

# 4. Incorporate viral data??

################## FEATURE ENGINEERING / SELECTION ##################
#
#
#
################## DATA SPLIT - TRAIN + TEST / EVALUATION ##################


# SPLIT DATA INTO TRAIN, TEST, EVALUATION
def split_data(df: pd.DataFrame, columns: list) -> (pd.DataFrame, pd.DataFrame):
    X = df[columns].values
    indices = df.index.values
    
    train_test, evaluation, indices_train_test, indices_evaluation = train_test_split(
        X, indices, test_size=0.25, random_state=736
    )
    
    train_test_df = df.iloc[indices_train_test]
    evaluation_df = df.iloc[indices_evaluation]
    return train_test_df, evaluation_df

train_test_df_ce, evaluation_df_ce = split_data(masked_ce, cd_data_cols)
train_test_df_bn, evaluation_df_bn = split_data(masked_bl, hd_data_cols)

print(f"Split counts for training, centrifuge: \n{train_test_df_ce['mask'].value_counts()}")
print(f"Split counts for evaluation, centrifuge: \n{evaluation_df_ce['mask'].value_counts()}")
print(f"Split counts for training, BLAST: \n{train_test_df_bn['mask'].value_counts()}")
print(f"Split counts for evaluation, BLAST: \n{evaluation_df_bn['mask'].value_counts()}")

################## DATA SPLIT - TRAIN + TEST / EVALUATION ##################
#
#
#
################## MACHINE LEARNING ##################
################## TRAINING + TESTING ##################


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


def prepare_data_sample(true_mask: dict, 
                        input_df: pd.DataFrame,
                        data_cols: list
                        ) -> (pd.DataFrame, list, 
                              pd.DataFrame, list, 
                              pd.DataFrame, dict,
                              pd.DataFrame):

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
        
        ###############################################
        input_df,
        ###############################################
        
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
    
    encoding_dict = {"True_negative": 0, "True_positive": 1}
    target_columns = ["sample-mask-TP", "sample-mask-TN"]
    
    encoded_ML = true_mask_df_merge[data_cols + ["sample_true_mask"]]
    
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
    
    mask_to_drop = list(set(all_masks) - set(target_columns))
    other_mask_df = encoded_ML[mask_to_drop]
    encoded_ML = encoded_ML.drop(mask_to_drop, axis=1)
    
    return encoded_ML, target_columns, true_mask_df_merge, all_masks, other_mask_df, encoding_dict, true_mask_df
    

def check_prediction(x):
    return True if x["sample_true_mask"] == x["XGB_sample_prediction"] else False


def draw_heatmap(fw_heatmap_df: pd.DataFrame, 
                 XGB_out: str,
                 dataset_used: str,
                 metagenome_classifier_used: str,
                 title: str) -> None:
    featurecorr = fw_heatmap_df.corr()
    featurecorr.index = featurecorr.index.str.replace(r"_", " ")
    featurecorr.columns = featurecorr.columns.str.replace(r"_", " ")
    
    mask = np.zeros_like(featurecorr)
    mask[np.triu_indices_from(mask)] = True
    
    f, ax = plt.subplots(figsize=(15, 15))
    ax = sns.heatmap(featurecorr, mask=mask)

    new_title = clean_strings(title, 120)
    plt.title(new_title, size=40)
    plt.tight_layout()
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=30)
    plt.savefig(f"{XGB_out}/{title}-decision.png", dpi=300, bbox_inches="tight")
    plt.show()
              

def check_mask_correct_pred(x):
    return True if x["mask"] == x["XGB_gen_con_prediction"] else False


def custom_samp_label_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if name in row["XGB_sample_prediction"]:
            return encoded


def one_hot_samp_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if encoded == row["XGB_sample_prediction"]:
            if encoded == 1:
                return (1, 0)
            if encoded == 0:
                return (0, 1)
            else:
                return (0, 0)
                
 
def custom_contaminant_label_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if name in row["mask"]:
            return encoded


def one_hot_contaminant_encoder(encoding_dict: dict, row: int):
    for name, encoded in encoding_dict.items():
        if encoded == row["encoded_mask"]:
            if encoded == 1:
                return (1, 0)
            if encoded == 0:
                return (0, 1)
            else:
                return (0, 0)
                 

pd.options.mode.chained_assignment = None  # default='warn'

json_file = f"{cat_out}{json_mask}"

with open(json_file) as json_data:
    true_mask = json.loads(json_data.read())
    
ce_id_cols = ["name", "sample", "seqID"]
bl_id_cols = ["name", "sample", "sseqid"]

################## MACHINE LEARNING ##################
################## TRAINING + TESTING ##################

train_test_df_ce = train_test_df_ce.drop_duplicates(subset=ce_id_cols)
train_test_df_ce.reset_index(inplace=True, drop=True)
train_test_df_bn = train_test_df_bn.drop_duplicates(subset=bl_id_cols)
train_test_df_bn.reset_index(inplace=True, drop=True)

train_test_df_bn.to_csv(f"{_ML_out_}/test-hs-dataset.csv", index=False)
train_test_df_ce.to_csv(f"{_ML_out_}/test-ce-dataset.csv", index=False)

try:
    # %matplotlib qt
    PCA.run(train_test_df_ce, "centrifuge", False, data_aug, XGB_out)
    PCA.run(train_test_df_bn, "BLAST", False, data_aug, XGB_out)
except:
    pass


def evaluate_model_performance(y_pred, 
                               XGB_classifier_model, 
                               X_test, 
                               X_train, 
                               y_train, 
                               y_test, 
                               metagenome_classifier_used: str, 
                               ml_model: str,
                               _type_: str, 
                               encoding_dict: dict):    
    # 5 folds, scored on accuracy
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    if _type_ == "test":
        cvs1 = cross_val_score(
            XGB_classifier_model, X_train, y_train, cv=10, scoring="accuracy"
        )
    cvs2 = cross_val_score(
        XGB_classifier_model, X_test, y_test, cv=10, scoring="accuracy"
    )
    
    e_d_list = list({f"{k}-{v}" for k,v in encoding_dict.items()})
    
    print(
        "-------------------------------------------------------------------------------"
    )
    print(f"XGBoost - {ml_model} status - {_type_} - {metagenome_classifier_used}\nEncoding dictionary: {e_d_list[0]}, {e_d_list[1]}")
    print("  ")
    confusion = confusion_matrix(y_test, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    # Classification Accuracy
    print("Classification Accuracy:", f"{(TP + TN) / float(TP + TN + FP + FN):.2f}")
    # print("Classification Accuracy:", f"{accuracy_score(y_test, y_pred):.2f}")
    # Classification Error
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print("Classification Error:", f"{classification_error:.2f}")
    # print("Classification Error:", f"{1 - accuracy_score(y_test, y_pred):.2f}")    
    # Sensitivity
    sensitivity = TP / float(FN + TP)
    print(f"Sensitivity: {sensitivity:.2f}")
    # print(f"Sensitivity: {recall_score(y_test, y_pred):.2f}")
    # Specificity
    specificity = TN / (TN + FP)
    print(f"Specificity: {specificity:.2f}")
    # False Positive Rate
    false_positive_rate = FP / float(TN + FP)
    print(f"FPR: {false_positive_rate:.2f}")
    # print(f"FPR: {1 - specificity:.2f}")
    # Precision
    precision = TP / float(TP + FP)
    print(f"Precision: {precision:.2f}")
    # print(f"Precision: {precision_score(y_test, y_pred):.2f}")   
    
    jaccard_index = TP / (TP + FP + TN)
    print(f"Jaccard index / critical success index: {jaccard_index:.2f}")
    
    print(confusion)
    print(classification_report(y_test, y_pred))  # Output
    print(
        "-------------------------------------------------------------------------------"
    )

    if _type_ == "test":    
        print(f"Cross-val #1 mean: {statistics.mean(cvs1):.2f}")
        print(f"Cross-val #1 stdev: {statistics.stdev(cvs1):.2f}")
    
    print(f"Cross-val #2 mean: {statistics.mean(cvs2):.2f}")
    print(f"Cross-val #2 stdev: {statistics.stdev(cvs2):.2f}")
    




def run_xgboost_sample_status(metagenome_classifier_used: str,
                              dataset_used: str,
                              true_mask: dict,
                              input_df: pd.DataFrame,
                              data_cols: list, 
                              id_cols: list, 
                              vees: str, 
                              XGB_out: str,
                              save: bool,
                              run_gridsearch: bool,
                              gauss_noise: bool,
                              transforms: bool, 
                              run_randomsearch: bool) -> (pd.DataFrame, pd.DataFrame):
    filename_save = f"xgboost_sample_status-{metagenome_classifier_used}.sav"
    
    encoded_ML, target_columns, true_mask_df_merge, all_masks, \
        other_mask_df, encoding_dict, true_mask_df = \
        prepare_data_sample(true_mask, input_df, data_cols)
    
    
    if data_aug:
        encoded_ML = xgboost_data_cleaning.data_aug(encoded_ML, 
                                                    metagenome_classifier_used, 
                                                    XGB_out,
                                                    gauss_noise,
                                                    transforms)
    
    # find optimal features
    features = featurewiz(encoded_ML, target=target_columns, corr_limit=0.70, verbose=2)
    feature_names, feature_df = features
    fw_heatmap_df = feature_df
    human_readable = feature_df
    
    title = f"XGBoost-sample-contaminant-status -- Heat map depicting important features for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using featurewiz"
    draw_heatmap(fw_heatmap_df, 
                     XGB_out,
                     dataset_used,
                     metagenome_classifier_used,
                     title)

    feat_labels = feature_names
    print(feat_labels)
    
    with open(
        f"{XGB_out}/feature-wiz-xgboost-features-{vees}-{metagenome_classifier_used}-sample-status.json",
        "w", encoding="utf-8") as f:
        json.dump(feat_labels, f)
    
    instance_names_df = true_mask_df_merge[id_cols]
    
    feature_wiz_df = pd.merge(
        instance_names_df, human_readable, how="left", left_index=True, right_index=True
    )
    feature_wiz_df = pd.merge(
        feature_wiz_df, other_mask_df, how="left", left_index=True, right_index=True
    )
    feature_wiz_df = feature_wiz_df.drop_duplicates(subset=id_cols)
    feature_wiz_df.reset_index(inplace=True, drop=True)
    
    ######################################
    y = feature_wiz_df["sample_encoded_mask"].values  
    indices = feature_wiz_df.index.values
    feature_wiz_clean = feature_wiz_df.drop(
        all_masks + id_cols, axis=1
    )
    X = feature_wiz_clean.iloc[:, :].values
    
    true_mask_df_merge = true_mask_df_merge.drop_duplicates(subset=id_cols)
    true_mask_df_merge.reset_index(inplace=True, drop=True)
    
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, indices, test_size=0.25, random_state=736
    )
    y_train, y_test = y[indices_train], y[indices_test]
    ######################################
    
    sc = StandardScaler()
    
    training_X = sc.fit_transform(X_train)
    
    test_X = sc.transform(X_test)

    eval_metric = ["logloss", "auc", "error"]
    eval_set = [(training_X, y_train), (test_X, y_test)]
    
    ######################################
    XGB_classifier_model = XGBClassifier(
        verbosity=0, random_state=736, alpha=0, gamma=0, max_depth=5, reg_lambda = 0,
        subsample=0.5, colsample_by_tree = 0.5, objective = "binary:logistic"    )
    
    if regularisation:
        if metagenome_classifier_used == "centrifuge":
            XGB_classifier_model = XGBClassifier(
                verbosity=0, random_state=736, alpha=1.0, gamma=0.01, max_depth=10, reg_lambda = 7,
                eta=0.5, subsample=0.5, colsample_by_tree = 0.5, objective = "binary:logistic"         )

        if metagenome_classifier_used == "BLAST":
            XGB_classifier_model = XGBClassifier(
                verbosity=0, random_state=736, alpha=3, gamma=0.01, max_depth=4, reg_lambda = 3,
                eta=0.5, subsample=0.5, colsample_by_tree = 1.0, objective = "binary:logistic"           )
    
    XGB_classifier_model.fit(training_X, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=False)
    ######################################
    
    y_pred = XGB_classifier_model.predict(test_X)

    print(f"\nAccuracy score for test sample: {accuracy_score(y_test, y_pred):.2f}\n")
    
    # save the scaler
    if save:
        pickle.dump(sc, open(f"{XGB_out}/sc-sample-status-XGB-scaler-{metagenome_classifier_used}.pkl", "wb"))
    
    ############ GRID SEARCH ##############
    
    # Method 1
    if run_gridsearch:
        grid = dict()
        grid["silent"] = [1]
        grid["gamma"] = [0.001, 0.01, 0.1, 2, 5]
        grid["colsample_bytree"] = [0.1, 0.5, 1]
        grid["max_depth"] = [3, 4, 5]
        grid["reg_lambda"] = [5] # [0, 2, 5]
        grid["alpha"] = [1] # [1, 5]
        grid["eta"] = [0.1, 0.5, 1] # learning rate
    
        # Instantiate GridSearchCV
        xgb = XGBClassifier(verbosity=0, random_state=736)
        gscv = GridSearchCV(estimator=xgb, param_grid=grid, scoring="accuracy", cv=5)
    
        # fit the model
        gscv.fit(training_X, y_train)
    
        # returns the estimator with the best performance
        print(f"{metagenome_classifier_used}, sample-status")
        print(gscv.best_estimator_)
    
        # returns the best score
        print(gscv.best_score_)
    
        # returns the best parameters
        print(gscv.best_params_)
        if save:
            pickle.dump(gscv.best_estimator_, open(f"{XGB_out}/gs-xgb-ss-{metagenome_classifier_used}.sav", "wb"))
    ############ GRID SEARCH ##############
    # Method 2
    if run_randomsearch:
        rand = dict()
        rand['silent'] = [1]
        rand['gamma'] = list(np.logspace(-2,2,num=2--2+1,base=10,dtype='float')) + list(np.logspace(-2,2,num=2--2+1,base=10,dtype='float') * 5)
        rand['colsample_bytree'] = list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float')) + list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float') * 5)
        rand['max_depth'] = list(range(4, 11))
        rand['reg_lambda'] = list(range(1, 11, 2))
        rand['alpha'] = list(range(1, 11, 2))
        rand['eta'] = list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float')) + list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float') * 5)
        
        xgb = XGBClassifier(verbosity=0, random_state=736)
        
        n_iter = 50  
    
        # Instantiate RandomSearchCV
        model_random_search = RandomizedSearchCV(estimator = xgb, param_distributions=rand, n_iter=n_iter)

        # Record the current time 
        # start = time()# Fit the selected model
        model_random_search.fit(X_train, y_train)# Print the time spend and number of models ran
        # print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time() - start), len(model_random_search.cv_results_['params'])))

        y_pred_random = model_random_search.predict(X_test)
        accuracy_random = accuracy_score(y_test, y_pred_random) 
        print(f"{metagenome_classifier_used}, sample, Random search accuracy: {accuracy_random}")
        print(f"Best random parameters found: {model_random_search.best_params_}\n")
    ############ RANDOM SEARCH ##############
    
    # Get predicted probabilities for each class
    y_preds_proba = XGB_classifier_model.predict_proba(test_X)
    y_preds_train = XGB_classifier_model.predict_proba(training_X)
    
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, y_preds_proba[:, 1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_train, y_preds_train[:, 1], pos_label=1)
    
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    
    # auc scores
    auc_score1 = roc_auc_score(y_test, y_preds_proba[:, 1])
    auc_score2 = roc_auc_score(y_train, y_preds_train[:, 1])
    
    plotting_data = [
        auc_score1,
        auc_score2,
        fpr1.tolist(),
        tpr1.tolist(),
        fpr2.tolist(),
        tpr2.tolist(),
        ]
    
    if save:
        with open(f"{XGB_out}/ROC-sample-status-XGB-plotting-data-{metagenome_classifier_used}.json", "w", encoding="utf-8") as f:
            json.dump(plotting_data, f)
    
    plt.style.use("seaborn")
    f, ax = plt.subplots(figsize=(15, 15))
    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle="--", color="red", label="XGB-test")
    plt.plot(fpr2, tpr2, linestyle="--", color="orange", label="XGB-train")
    # plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    # title
    title = f"AUC-ROC curve for sample contaminant status for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used}"
    new_title = clean_strings(title, 120)
    plt.title(new_title, size=40)
    # x label
    plt.xlabel("False Positive Rate", size=40)
    # y label
    plt.ylabel("True Positive rate", size=40)
    plt.text(
        0.8,
        0.2,
        f"AUC-test: {auc_score1:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=30,
    )
    plt.text(
        0.8,
        0.25,
        f"AUC-train: {auc_score2:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=30,
    )
    plt.legend(loc=4, prop={"size": 40}, markerscale=10)
    if save:
        plt.savefig(f"{XGB_out}/{title}.png", dpi=300)
    plt.show()
    
    
    XGB_model_save = f"{XGB_out}/{filename_save}"
    print(f"Saving one XGBoost classifier model to: {XGB_model_save}")
    if save:
        pickle.dump(XGB_classifier_model, open(XGB_model_save, "wb"))


    evaluate_model_performance(y_pred, XGB_classifier_model, 
                                   X_test, X_train, y_train, 
                                   y_test, metagenome_classifier_used,
                                   "sample contaminant",
                                   "test", encoding_dict)
        
    # to get convert dictionary to list
    # count number of true labels and samples
    t_positives = 0
    no_samples = len(true_mask_df)
    for db, dict_ in true_mask.items():
        words = list(true_mask[db].values())
        keyss = Counter(words).keys()  # equals to list(set(words)))
        valuess = Counter(words).values()  # counts the elements' frequency)
        index_key = 0
        if "True_positive" in keyss:
            index_key = list(keyss).index("True_positive")
            t_positive = list(valuess)[index_key]
            if t_positive > 0:
                t_positives = t_positive
    
    print(
        f"Positive labels: {t_positives}/{no_samples}; {t_positives/no_samples*100:.2f} %"
        )
    
    sample_status = true_mask_df_merge[
        ["sample", "name", "mask", "sample_true_mask"]
    ]
    sample_status.loc[indices_test, "XGB_sample_prediction"] = XGB_classifier_model.predict(
        test_X
    )
    sample_status.loc[
        indices_train, "XGB_sample_prediction"
    ] = XGB_classifier_model.predict(training_X)
    
    sample_status.loc[indices_test, "sample-status"] = "test"
    sample_status.loc[indices_train, "sample-status"] = "train"
    
    encoding_dict_rev = {value: key for (key, value) in encoding_dict.items()}
    sample_status = sample_status.replace({"XGB_sample_prediction": encoding_dict_rev})
    
    sample_status["sample-pred-correct"] = sample_status.apply(check_prediction, axis=1)
    
    return sample_status, true_mask_df
   

ce_tt_sample_status, true_mask_df = run_xgboost_sample_status("centrifuge", "train-test", \
                true_mask, train_test_df_ce, cd_data_cols, ce_id_cols, vees, XGB_out, \
                save, run_gridsearch, gauss_noise, transforms, run_randomsearch)
hs_tt_sample_status, _ = run_xgboost_sample_status("BLAST", "train-test", \
                true_mask, train_test_df_bn, hd_data_cols, bl_id_cols, vees, XGB_out, \
                save, run_gridsearch,  gauss_noise, transforms, run_randomsearch)

                
def run_xgboost_genuine_contaminant_status(metagenome_classifier_used: str,
                              dataset_used: str,
                              true_mask: dict,
                              input_df: pd.DataFrame,
                              input_df_xgboost: pd.DataFrame,
                              data_cols: list, 
                              id_cols: list, 
                              vees: str,
                              true_mask_df: pd.DataFrame, 
                              XGB_out: str,
                              save: bool,
                              run_gridsearch: bool,
                              gauss_noise: bool,
                              transforms: bool,
                              run_randomsearch: bool) -> pd.DataFrame:

    filename_save = f"xgboost_genuine_contaminant_status-{metagenome_classifier_used}.sav"
    
    genuine_contaminant_df = pd.merge(input_df, input_df_xgboost[['sample_true_mask', 'XGB_sample_prediction', 'sample-status',
           'sample-pred-correct']], left_index=True, right_index=True)
    
    # convert prediction T/F into encoded 0,1
    encoding_dict_samp = {"True_positive": 1, "True_negative": 0}
    target_columns_samp = ["sample-pred-TP", "sample-pred-TN"]
    label_partial_func = partial(custom_samp_label_encoder, encoding_dict_samp)
    genuine_contaminant_df["XGB_sample_prediction"] = genuine_contaminant_df.apply(label_partial_func, axis=1)
    
    enc_samp_partial_func = partial(one_hot_samp_encoder, encoding_dict_samp)
    genuine_contaminant_df["temp-mask"] = genuine_contaminant_df.apply(enc_samp_partial_func, axis=1)    
    
    genuine_contaminant_df[target_columns_samp] = pd.DataFrame(
        genuine_contaminant_df["temp-mask"].tolist(), index=genuine_contaminant_df.index
    )
    
    genuine_contaminant_df = genuine_contaminant_df.drop(["temp-mask"], axis=1)
    
    encoded_ML = genuine_contaminant_df[data_cols + ["mask"] + target_columns_samp]

    encoding_dict = {"False_positive": 0, "True_positive": 1}
    target_columns = ["mask-FP", "mask-TP"]
    
    label_partial_func = partial(custom_contaminant_label_encoder, encoding_dict)
    encoded_ML["encoded_mask"] = encoded_ML.apply(label_partial_func, axis=1)
    
    enc_partial_func = partial(one_hot_contaminant_encoder, encoding_dict)
    encoded_ML["temp-mask"] = encoded_ML.apply(enc_partial_func, axis=1)
    encoded_ML[target_columns] = pd.DataFrame(
        encoded_ML["temp-mask"].tolist(), index=encoded_ML.index
    )
    encoded_ML = encoded_ML.drop(["temp-mask"], axis=1)
    
    all_masks = ["mask", "encoded_mask", "mask-FP", "mask-TP"]
    
    mask_to_drop = list(set(all_masks) - set(target_columns))
    other_mask_df = encoded_ML[mask_to_drop]
    encoded_ML = encoded_ML.drop(mask_to_drop, axis=1)

    if data_aug:
        encoded_ML = xgboost_data_cleaning.data_aug(encoded_ML, 
                                                    metagenome_classifier_used, 
                                                    XGB_out,
                                                    gauss_noise,
                                                    transforms)
    
    # find optimal features
    features = featurewiz(encoded_ML, target=target_columns, corr_limit=0.70, verbose=2)
    feature_names, feature_df = features
    fw_heatmap_df = feature_df
    human_readable = feature_df
    
    title = f"XGBoost-genuine-contaminant-status -- Heat map depicting important features for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using featurewiz"
    draw_heatmap(fw_heatmap_df, 
                     XGB_out,
                     dataset_used,
                     metagenome_classifier_used,
                     title)
    
    feat_labels = feature_names
    if save:
        with open(
            f"{XGB_out}/feature-wiz-xgboost-features-{vees}-genuine-contaminant-{metagenome_classifier_used}.json",
            "w", encoding="utf-8") as f:
            json.dump(feat_labels, f)

    instance_names_df = genuine_contaminant_df[id_cols]
    
    feature_wiz_df = pd.merge(
        instance_names_df, human_readable, how="left", left_index=True, right_index=True
    )
    feature_wiz_df = pd.merge(
        feature_wiz_df, other_mask_df, how="left", left_index=True, right_index=True
    )
    feature_wiz_df = feature_wiz_df.drop_duplicates(subset=id_cols)
    feature_wiz_df.reset_index(inplace=True, drop=True)
    
    y = feature_wiz_df["encoded_mask"].values  
    indices = feature_wiz_df.index.values
    feature_wiz_clean = feature_wiz_df.drop(
        all_masks + id_cols, axis=1
    )
    X = feature_wiz_clean.iloc[:, :].values
    
    instance_names_df = instance_names_df.drop_duplicates(subset=id_cols)
    instance_names_df.reset_index(inplace=True, drop=True)
    
    X_train, X_test, indices_train, indices_test = train_test_split(
        X, indices, test_size=0.25, random_state=736
    )
    y_train, y_test = y[indices_train], y[indices_test]
    
    sc = StandardScaler()
    
    training_X = sc.fit_transform(X_train)
    
    test_X = sc.transform(X_test)

    XGB_classifier_model = XGBClassifier(
        verbosity=0, random_state=736, alpha=0, gamma=0, reg_lambda = 0,
        max_depth=5, subsample=0.5, colsample_by_tree = 0.5    )
        
    if regularisation:
        if metagenome_classifier_used == "centrifuge":
            XGB_classifier_model = XGBClassifier(
                verbosity=0, random_state=736, alpha=5, gamma=0.05, max_depth=6, reg_lambda = 7,
                eta=0.5, subsample=0.5, colsample_by_tree = 0.5, objective = "binary:logistic")       
        if metagenome_classifier_used == "BLAST":
            XGB_classifier_model = XGBClassifier(
                verbosity=0, random_state=736, alpha=3, gamma=0.1, max_depth=6, reg_lambda = 9,
                eta=1.0, subsample=1.0, colsample_by_tree = 1, objective = "binary:logistic")   
            
    XGB_classifier_model.fit(training_X, y_train)
    
    # print("\n\nConformal analysis")
    # conformal_analysis(dataset_used, metagenome_classifier_used, test_X, y_test, XGB_classifier_model, training_X, y_train, "genuine status", XGB_out)
    
    y_pred = XGB_classifier_model.predict(test_X)

    print(f"\nAccuracy score for test sample: {accuracy_score(y_test, y_pred):.2f}\n")

    # save the scaler
    if save:
        pickle.dump(sc, open(f"{XGB_out}/sc-genuine-contaminant-XGB-scaler-{metagenome_classifier_used}.pkl", "wb"))
    
    ############ GRID SEARCH ##############
    
    # Method 1
    if run_gridsearch:
        grid = dict()
        grid["silent"] = [1]
        grid["gamma"] = [0.001, 0.01, 0.1, 2, 5]
        grid["colsample_bytree"] = [0.1, 0.5, 1]
        grid["max_depth"] = [3, 4, 5]
        grid["reg_lambda"] = [5] # [0, 2, 5]
        grid["alpha"] = [1] # [1, 5]
        grid["eta"] = [0.1, 0.5, 1] # learning rate
    
        # Instantiate GridSearchCV
        xgb = XGBClassifier(verbosity=0, random_state=736)
        gscv = GridSearchCV(estimator=xgb, param_grid=grid, scoring="accuracy", cv=5)
    
        # fit the model
        gscv.fit(training_X, y_train)
    
        # returns the estimator with the best performance
        print(f"{metagenome_classifier_used}, contaminant-status")
        print(gscv.best_estimator_)
    
        # returns the best score
        print(gscv.best_score_)
    
        # returns the best parameters
        print(gscv.best_params_)
        if save:
            pickle.dump(gscv.best_estimator_, open(f"{XGB_out}/gs-xgb-cs-{metagenome_classifier_used}.sav", "wb"))
    ############ GRID SEARCH ##############
    # Method 2
    if run_randomsearch:
        rand = dict()
        rand['silent'] = [1]
        rand['gamma'] = list(np.logspace(-2,2,num=2--2+1,base=10,dtype='float')) + list(np.logspace(-2,2,num=2--2+1,base=10,dtype='float') * 5)
        rand['colsample_bytree'] = list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float')) + list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float') * 5)
        rand['max_depth'] = list(range(4, 11))
        rand['reg_lambda'] = list(range(1, 11, 2))
        rand['alpha'] = list(range(1, 11, 2))
        rand['eta'] = list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float')) + list(np.logspace(-1,1,num=1--1+1,base=10,dtype='float') * 5)
        
        xgb = XGBClassifier(verbosity=0, random_state=736)
        
        n_iter = 50  
    
        # Instantiate RandomSearchCV
        model_random_search = RandomizedSearchCV(estimator = xgb, param_distributions=rand, n_iter=n_iter)

        # Record the current time 
        # start = time()# Fit the selected model
        model_random_search.fit(X_train, y_train)# Print the time spend and number of models ran
        # print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." % ((time() - start), len(model_random_search.cv_results_['params'])))

        y_pred_random = model_random_search.predict(X_test)
        accuracy_random = accuracy_score(y_test, y_pred_random) 
        print(f"{metagenome_classifier_used}, contaminant, Random search accuracy: {accuracy_random}")
        print(f"Best random parameters found: {model_random_search.best_params_}\n")
    ############ RANDOM SEARCH ##############
    
    y_preds_proba = XGB_classifier_model.predict_proba(test_X)
    y_preds_train = XGB_classifier_model.predict_proba(training_X)
    
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, y_preds_proba[:, 1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_train, y_preds_train[:, 1], pos_label=1)
    
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    
    # auc scores
    auc_score1 = roc_auc_score(y_test, y_preds_proba[:, 1])
    auc_score2 = roc_auc_score(y_train, y_preds_train[:, 1])
    
    plotting_data = [
        auc_score1,
        auc_score2,
        fpr1.tolist(),
        tpr1.tolist(),
        fpr2.tolist(),
        tpr2.tolist(),
        ]
    
    if save:
        with open(f"{XGB_out}/ROC-adventitious-agent-XGB-plotting-data-{metagenome_classifier_used}.json", "w", encoding="utf-8") as f:
            json.dump(plotting_data, f)
    
    plt.style.use("seaborn")
    f, ax = plt.subplots(figsize=(15, 15))
    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle="--", color="red", label="XGB-test")
    plt.plot(fpr2, tpr2, linestyle="--", color="orange", label="XGB-train")
    # plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    # title
    title = f"AUC-ROC curve for genuine contaminant status for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used}"
    new_title = clean_strings(title, 120)
    plt.title(new_title, size=40)
    # x label
    plt.xlabel("False Positive Rate", size=40)
    # y label
    plt.ylabel("True Positive rate", size=40)
    plt.text(
        0.8,
        0.2,
        f"AUC-test: {auc_score1:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=30,
    )
    plt.text(
        0.8,
        0.25,
        f"AUC-train: {auc_score2:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=30,
    )
    plt.legend(loc=4, prop={"size": 40}, markerscale=10)
    if save:
        plt.savefig(f"{XGB_out}/{title}.png", dpi=300)
    plt.show()
    
    
    XGB_model_save = f"{XGB_out}/{filename_save}"
    print(f"Saving one XGBoost classifier model to: {XGB_model_save}")
    if save:
        pickle.dump(XGB_classifier_model, open(XGB_model_save, "wb"))

    evaluate_model_performance(y_pred, XGB_classifier_model, 
                                       X_test, X_train, y_train, 
                                       y_test, metagenome_classifier_used,
                                       "genuine contaminant",
                                       "test",
                                       encoding_dict)

    # to get convert dictionary to list
    # count number of true labels and samples
    t_positives = 0
    no_samples = len(true_mask_df)
    for db, dict_ in true_mask.items():
        words = list(true_mask[db].values())
        keyss = Counter(words).keys()  # equals to list(set(words)))
        valuess = Counter(words).values()  # counts the elements' frequency)
        index_key = 0
        if "True_positive" in keyss:
            index_key = list(keyss).index("True_positive")
            t_positive = list(valuess)[index_key]
            if t_positive > 0:
                t_positives = t_positive
    
    print(
        f"Positive labels: {t_positives}/{no_samples}; {t_positives/no_samples*100:.2f} %"
        )
    
    gen_con_status = genuine_contaminant_df[
        ["sample", "name", "mask", "sample_true_mask", "XGB_sample_prediction"]
    ]
    gen_con_status.loc[indices_test, "XGB_gen_con_prediction"] = XGB_classifier_model.predict(
        test_X
    )
    gen_con_status.loc[
        indices_train, "XGB_gen_con_prediction"
    ] = XGB_classifier_model.predict(training_X)
    
    gen_con_status.loc[indices_test, "sample-status"] = "test"
    gen_con_status.loc[indices_train, "sample-status"] = "train"
    
    encoding_dict_rev = {value: key for (key, value) in encoding_dict.items()}
    gen_con_status = gen_con_status.replace({"XGB_gen_con_prediction": encoding_dict_rev})
    
    gen_con_status["gen-con-pred-correct"] = gen_con_status.apply(check_mask_correct_pred, axis=1)
    
    return gen_con_status


ce_tt_gen_con_status = run_xgboost_genuine_contaminant_status("centrifuge", \
                      "train-test", true_mask, train_test_df_ce, \
                          ce_tt_sample_status, cd_data_cols, ce_id_cols, vees, \
                              true_mask_df, XGB_out, save, \
                                  run_gridsearch, gauss_noise, transforms,
                                  run_randomsearch)
hs_tt_gen_con_status = run_xgboost_genuine_contaminant_status("BLAST", \
                      "train-test", true_mask, train_test_df_bn, \
                          hs_tt_sample_status, hd_data_cols, bl_id_cols, vees, \
                              true_mask_df, XGB_out, save, \
                                  run_gridsearch, gauss_noise, transforms,
                                  run_randomsearch)

################## TRAINING + TESTING ##################
################## MACHINE LEARNING ##################
#
#
#
################## MACHINE LEARNING ##################
################## EVALUATION ##################

evaluation_df_ce = evaluation_df_ce.drop_duplicates(subset=ce_id_cols)
evaluation_df_ce.reset_index(inplace=True, drop=True)
evaluation_df_bn = evaluation_df_bn.drop_duplicates(subset=bl_id_cols)
evaluation_df_bn.reset_index(inplace=True, drop=True)


def run_xgboost_sample_status_evaluate(metagenome_classifier_used: str,
                              dataset_used: str,
                              true_mask: dict,
                              input_df: pd.DataFrame,
                              data_cols: list, 
                              id_cols: list, 
                              vees: str, 
                              XGB_out: str,
                              save: bool,
                              run_gridsearch: bool, gauss_noise: bool, transforms: bool) -> pd.DataFrame:

    encoded_ML, target_columns, true_mask_df_merge, all_masks, \
        other_mask_df, encoding_dict, true_mask_df = \
        prepare_data_sample(true_mask, input_df, data_cols)
    
    if data_aug:
        encoded_ML = xgboost_data_cleaning.data_aug(encoded_ML, 
                                                    metagenome_classifier_used, 
                                                    XGB_out,
                                                    gauss_noise,
                                                    transforms)
    
    with open(
        f"{XGB_out}/feature-wiz-xgboost-features-{vees}-{metagenome_classifier_used}-sample-status.json"
    ) as json_sample_XGB:
        feat_labels = json.loads(json_sample_XGB.read())
        
    sc = pickle.load(open(f"{XGB_out}/sc-sample-status-XGB-scaler-{metagenome_classifier_used}.pkl", "rb"))
    
    filename_save = f"xgboost_sample_status-{metagenome_classifier_used}.sav"
    XGB_model_save = f"{XGB_out}/{filename_save}"
    if Path(XGB_model_save).is_file():
        XGB_classifier_model = pickle.load(open(XGB_model_save, "rb"))
        
        if run_gridsearch:
            XGB_classifier_model = pickle.load(open(f"{XGB_out}/gs-xgb-ss-{metagenome_classifier_used}.sav", "rb"))
    
        fw_heatmap_eval_df = encoded_ML[feat_labels + target_columns]
        title = f"XGBoost-sample-contaminant-status -- Heat map depicting important features for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using featurewiz"
        draw_heatmap(fw_heatmap_eval_df, 
                         XGB_out,
                         dataset_used,
                         metagenome_classifier_used,
                         title)
        
        instance_names_df = true_mask_df_merge[id_cols]
        
        feature_wiz_df = pd.merge(
            instance_names_df, fw_heatmap_eval_df, how="left", left_index=True, right_index=True
        )
        feature_wiz_df = pd.merge(
            feature_wiz_df, other_mask_df, how="left", left_index=True, right_index=True
        )
        feature_wiz_df = feature_wiz_df.drop_duplicates(subset=id_cols)
        feature_wiz_df.reset_index(inplace=True, drop=True)
        
        y = feature_wiz_df["sample_encoded_mask"].values  
        indices = feature_wiz_df.index.values
        feature_wiz_clean = feature_wiz_df.drop(
            all_masks + id_cols, axis=1
        )
        X = feature_wiz_clean.iloc[:, :].values
        
        X_eval = sc.fit_transform(X)
        
        y_pred = XGB_classifier_model.predict(X_eval)
    
        print(f"\nAccuracy score for evaluation sample: {accuracy_score(y, y_pred):.2f}\n")
    
        # Get predicted probabilities for each class
        y_preds_proba = XGB_classifier_model.predict_proba(X_eval)
    
        try:
            with open(f"{XGB_out}/ROC-sample-status-XGB-plotting-data-{metagenome_classifier_used}.json") as json_data:
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
    
            # auc scores
            auc_score1 = roc_auc_score(y, y_preds_proba[:, 1])
    
            centrifuge_plot = "viral, fungal and bacterial database"
            
            plt.style.use("seaborn")
            f, ax = plt.subplots(figsize=(15, 15))
            # plot roc curves
            plt.plot(
                fpr1,
                tpr1,
                linestyle="--",
                color="red",
                label="XGB-eval",
            )
            plt.plot(fpr2, tpr2, linestyle="--", color="green", label="XGB-test")
            plt.plot(fpr3, tpr3, linestyle="--", color="orange", label="XGB-train")
            plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)
            # title
            title = f"AUC-ROC curve for sample contaminant status for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using the {centrifuge_plot}"
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
            plt.savefig(f"{XGB_out}/{title}.png", dpi=300)
            plt.show()
    
        except:
            print("Missing examples of class X, cannot plot")
    

        evaluate_model_performance(y_pred, XGB_classifier_model, 
                                           X_eval, _, _, 
                                           y, metagenome_classifier_used,
                                           "sample contaminant",
                                           "evaluate",
                                           encoding_dict)
    
        sample_status = true_mask_df_merge[
            ["sample", "name", "mask", "sample_true_mask"]
        ]
        sample_status.loc[indices, "XGB_sample_prediction"] = XGB_classifier_model.predict(
            X_eval
        )
        
        sample_status.loc[indices, "sample-status"] = "eval"
    
        encoding_dict_rev = {value: key for (key, value) in encoding_dict.items()}
        sample_status = sample_status.replace({"XGB_sample_prediction": encoding_dict_rev})
        
        sample_status["sample-pred-correct"] = sample_status.apply(check_prediction, axis=1)
        
        return sample_status
    

ce_eval_sample_status = run_xgboost_sample_status_evaluate("centrifuge", "evaluation", \
                        true_mask, evaluation_df_ce, cd_data_cols, ce_id_cols, vees, XGB_out, save, run_gridsearch, gauss_noise, transforms)
hs_eval_sample_status = run_xgboost_sample_status_evaluate("BLAST", "evaluation", \
                        true_mask, evaluation_df_bn, hd_data_cols, bl_id_cols, vees, XGB_out, save, run_gridsearch, gauss_noise, transforms)


def run_xgboost_genuine_contaminant_status_evaluate(metagenome_classifier_used: str,
                              dataset_used: str,
                              true_mask: dict,
                              input_df: pd.DataFrame,
                              input_df_xgboost: pd.DataFrame,
                              data_cols: list, 
                              id_cols: list, 
                              vees: str,
                              true_mask_df: pd.DataFrame, 
                              XGB_out: str,
                              save: bool,
                              run_rch: bool, gauss_noise: bool, transforms: bool) -> pd.DataFrame:

    filename_save = f"xgboost_genuine_contaminant_status-{metagenome_classifier_used}.sav"
    
    genuine_contaminant_df = pd.merge(input_df, input_df_xgboost[['sample_true_mask', 'XGB_sample_prediction', 'sample-status',
           'sample-pred-correct']], left_index=True, right_index=True)    
    
    # convert prediction T/F into encoded 0,1
    encoding_dict_samp = {"True_positive": 1, "True_negative": 0}
    target_columns_samp = ["sample-pred-TP", "sample-pred-TN"]
    label_partial_func = partial(custom_samp_label_encoder, encoding_dict_samp)
    genuine_contaminant_df["XGB_sample_prediction"] = genuine_contaminant_df.apply(label_partial_func, axis=1)
    
    enc_samp_partial_func = partial(one_hot_samp_encoder, encoding_dict_samp)
    genuine_contaminant_df["temp-mask"] = genuine_contaminant_df.apply(enc_samp_partial_func, axis=1)    
    
    genuine_contaminant_df[target_columns_samp] = pd.DataFrame(
        genuine_contaminant_df["temp-mask"].tolist(), index=genuine_contaminant_df.index
    )
    
    genuine_contaminant_df = genuine_contaminant_df.drop(["temp-mask"], axis=1)
    
    encoded_ML = genuine_contaminant_df[data_cols + ["mask"] + target_columns_samp]
    
    encoding_dict = {"False_positive": 0, "True_positive": 1}
    target_columns = ["mask-FP", "mask-TP"]
    
    label_partial_func = partial(custom_contaminant_label_encoder, encoding_dict)
    encoded_ML["encoded_mask"] = encoded_ML.apply(label_partial_func, axis=1)
    
    enc_partial_func = partial(one_hot_contaminant_encoder, encoding_dict)
    encoded_ML["temp-mask"] = encoded_ML.apply(enc_partial_func, axis=1)
    encoded_ML[target_columns] = pd.DataFrame(
        encoded_ML["temp-mask"].tolist(), index=encoded_ML.index
    )
    encoded_ML = encoded_ML.drop(["temp-mask"], axis=1)
    
    all_masks = ["mask", "encoded_mask", "mask-FP", "mask-TP"]
    
    mask_to_drop = list(set(all_masks) - set(target_columns))
    other_mask_df = encoded_ML[mask_to_drop]
    encoded_ML = encoded_ML.drop(mask_to_drop, axis=1)
    
    if data_aug:
        encoded_ML = xgboost_data_cleaning.data_aug(encoded_ML, 
                                                    metagenome_classifier_used, 
                                                    XGB_out,
                                                    gauss_noise,
                                                    transforms)
    
    with open(
        f"{XGB_out}/feature-wiz-xgboost-features-{vees}-genuine-contaminant-{metagenome_classifier_used}.json"
    ) as json_sample_XGB:
        feat_labels = json.loads(json_sample_XGB.read())
        
    sc = pickle.load(open(f"{XGB_out}/sc-genuine-contaminant-XGB-scaler-{metagenome_classifier_used}.pkl", "rb"))
    
    filename_save = f"xgboost_genuine_contaminant_status-{metagenome_classifier_used}.sav"
    XGB_model_save = f"{XGB_out}/{filename_save}"
    if Path(XGB_model_save).is_file():
        XGB_classifier_model = pickle.load(open(XGB_model_save, "rb"))
        
        if run_gridsearch:
            XGB_classifier_model = pickle.load(open(f"{XGB_out}/gs-xgb-cs-{metagenome_classifier_used}.sav", "rb"))
    
        fw_heatmap_eval_df = encoded_ML[feat_labels + target_columns]
    
        title = f"XGBoost-genuine-contaminant-status -- Heat map depicting important features for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using featurewiz"
        draw_heatmap(fw_heatmap_eval_df, 
                         XGB_out,
                         dataset_used,
                         metagenome_classifier_used,
                         title)
    
        instance_names_df = genuine_contaminant_df[id_cols]
        
        feature_wiz_df = pd.merge(
            instance_names_df, fw_heatmap_eval_df, how="left", left_index=True, right_index=True
        )
        feature_wiz_df = pd.merge(
            feature_wiz_df, other_mask_df, how="left", left_index=True, right_index=True
        )
        feature_wiz_df = feature_wiz_df.drop_duplicates(subset=id_cols)
        feature_wiz_df.reset_index(inplace=True, drop=True)
        
        y = feature_wiz_df["encoded_mask"].values  
        indices = feature_wiz_df.index.values
        feature_wiz_clean = feature_wiz_df.drop(
            all_masks + id_cols, axis=1
        )
        X = feature_wiz_clean.iloc[:, :].values
        
        X_eval = sc.fit_transform(X)
        
        y_pred = XGB_classifier_model.predict(X_eval)

        # print("\n\nConformal analysis")
        # conformal_analysis(dataset_used, metagenome_classifier_used, X_eval, y, XGB_classifier_model, [], [], "genuine status", XGB_out)
    
        print(f"\nAccuracy score for evaluation sample: {accuracy_score(y, y_pred):.2f}\n")
    
        # Get predicted probabilities for each class
        y_preds_proba = XGB_classifier_model.predict_proba(X_eval)
        
        try:
            with open(f"{XGB_out}/ROC-adventitious-agent-XGB-plotting-data-{metagenome_classifier_used}.json") as json_data:
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
    
            # auc scores
            auc_score1 = roc_auc_score(y, y_preds_proba[:, 1])
    
            centrifuge_plot = "viral, fungal and bacterial database"
            # centrifuge_label = "viral-fungal-bacterial"
            
            plt.style.use("seaborn")
            f, ax = plt.subplots(figsize=(15, 15))
            # plot roc curves
            plt.plot(
                fpr1,
                tpr1,
                linestyle="--",
                color="red",
                label="XGB-eval",
            )
            plt.plot(fpr2, tpr2, linestyle="--", color="green", label="XGB-test")
            plt.plot(fpr3, tpr3, linestyle="--", color="orange", label="XGB-train")
            plt.plot(p_fpr, p_tpr, linestyle="--", color="blue")
            ax.tick_params(axis="x", labelsize=25)
            ax.tick_params(axis="y", labelsize=25)
            # title
            title = f"AUC-ROC curve for genuine contaminant status for dataset - {dataset_used} and metagenome classifier - {metagenome_classifier_used} using the {centrifuge_plot}"
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
            plt.savefig(f"{XGB_out}/{title}.png", dpi=300)
            plt.show()
    
        except:
            print("Missing examples of class X, cannot plot")
        
        evaluate_model_performance(y_pred, XGB_classifier_model, 
                                           X_eval, _, _, 
                                           y, metagenome_classifier_used,
                                           "genuine contaminant",
                                           "evaluate",
                                           encoding_dict)
    
        gen_con_status = genuine_contaminant_df[
            ["sample", "name", "mask", "sample_true_mask", "XGB_sample_prediction"]
        ]
        gen_con_status.loc[indices, "XGB_gen_con_prediction"] = XGB_classifier_model.predict(
            X_eval
        )
    
        gen_con_status.loc[indices, "sample-status"] = "eval"
    
        encoding_dict_rev = {value: key for (key, value) in encoding_dict.items()}
        gen_con_status = gen_con_status.replace({"XGB_gen_con_prediction": encoding_dict_rev})
        
        gen_con_status["gen-con-pred-correct"] = gen_con_status.apply(check_mask_correct_pred, axis=1)
        
        return gen_con_status

        
ce_eval_gen_con_status = run_xgboost_genuine_contaminant_status_evaluate("centrifuge", \
                                 "evaluation", true_mask, evaluation_df_ce, \
                                     ce_eval_sample_status, cd_data_cols, \
                                         ce_id_cols, vees, true_mask_df, \
                                             XGB_out, save, run_gridsearch, gauss_noise, transforms)
hs_eval_gen_con_status = run_xgboost_genuine_contaminant_status_evaluate("BLAST", \
                                 "evaluation", true_mask, evaluation_df_bn, \
                                     hs_eval_sample_status, hd_data_cols, \
                                         bl_id_cols, vees, true_mask_df, \
                                             XGB_out, save, run_gridsearch, gauss_noise, transforms)

################## EVALUATION ##################
################## MACHINE LEARNING ##################
#
#
#
################## ASSESSMENT OF EFFICACY ##################

cs_acc = ML_assessment.calculate_accuracy(ce_eval_sample_status, "sample-pred-correct", "evaluation-sample", "centrifuge")
hs_acc = ML_assessment.calculate_accuracy(hs_eval_sample_status, "sample-pred-correct", "evaluation-sample", "BLAST")
cg_acc = ML_assessment.calculate_accuracy(ce_eval_gen_con_status, "gen-con-pred-correct", "evaluation-genuine", "centrifuge")
hg_acc = ML_assessment.calculate_accuracy(hs_eval_gen_con_status, "gen-con-pred-correct", "evaluation-genuine", "BLAST")

ML_assessment.calculate_classification_accuracy_interval(cs_acc, ce_eval_sample_status, "evaluation-sample", "centrifuge")
ML_assessment.calculate_classification_accuracy_interval(hs_acc, hs_eval_sample_status, "evaluation-sample", "BLAST")
ML_assessment.calculate_classification_accuracy_interval(cg_acc, ce_eval_gen_con_status, "evaluation-genuine", "centrifuge")
ML_assessment.calculate_classification_accuracy_interval(hg_acc, hs_eval_gen_con_status, "evaluation-genuine", "BLAST")

decision_pivot_ce, eval_gen_con_status_ce = ML_assessment.calculate_twin_model_accuracy(ce_eval_gen_con_status, "centrifuge", "evaluation")
decision_pivot_hs, eval_gen_con_status_hs = ML_assessment.calculate_twin_model_accuracy(hs_eval_gen_con_status, "BLAST", "evaluation")  
