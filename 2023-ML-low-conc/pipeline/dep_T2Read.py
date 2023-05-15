#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:15:14 2022

@author: mangi
"""

import pandas as pd
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from datetime import datetime
from collections import Counter
import time
import json
import glob
import os
import re


def human_ref_handler(row):
    compiled = re.compile(r"\|(\w+.\w+)")
    return compiled.findall(row)[0]


def RVDB_ref_handler(row):
    compiled = re.compile(r"(\w+)\|(\w+)\|([\w\.\d]+)\|(.*[\.\-\w]+)")
    return compiled.findall(row)[0][2]


def clean_qseqid(row):
    compile_clean = re.compile(r"(\w+\-\w+\-\w+\-\w+\-\w+$)")
    if len(row) < 40:
        return row
    if len(row) > 40:
        cleaned = compile_clean.findall(row)
        if len(cleaned) > 0:
            len_row = len(cleaned[0])
            remove = len_row - 36
            return cleaned[0][remove:]


def hs_blastn_import(directory: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{directory}",
        sep="\t",
        header=None,
        names=[
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "mismatches",
            "gap_opens",
            "q_start",
            "q_end",
            "s_start",
            "s_end",
            "evalue",
            "bitscore",
        ],
    )
    if not df.empty:
        df = df.dropna()
        df.reset_index(drop=True)
        df.qseqid = df.qseqid.apply(clean_qseqid)
        row_indices = df.dropna()
        df = df.loc[row_indices.index].reset_index()
        df = df.drop(["index"], axis=1)
        if df["sseqid"].str.contains("ref")[0]:
            df["sseqid"] = df["sseqid"].apply(human_ref_handler)
        if df["sseqid"].str.contains("acc")[0]:
            df["sseqid"] = df["sseqid"].apply(RVDB_ref_handler)
        if not df.empty:
            return df


def split_name(df: pd.DataFrame) -> pd.DataFrame:
    df_split = pd.DataFrame(
        df.name.str.split(" ", 2).tolist(), columns=["genus", "species", "strain"]
    )
    df_split["name"] = df_split[["genus", "species", "strain"]].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )
    return df_split


# collect species genus, species and strain
def import_deanonymised_nomenclature(directory):
    seqid_deanonymiser = pd.read_csv(
        directory, sep=",", header=None, names=["sseqid", "name"]
    )
    seqid_deanonymiser = seqid_deanonymiser.iloc[1:]
    seqid_deanonymised_names = split_name(seqid_deanonymiser)

    seqid_deanonymised_clean = seqid_deanonymiser.merge(
        seqid_deanonymised_names, on="name", how="inner"
    )
    seqid_deanonymised_clean = seqid_deanonymised_clean.drop_duplicates(
        subset=["sseqid"]
    )
    return seqid_deanonymised_clean


def getseqid(root: str, bN_name: str) -> str:
    # get all human, bacteria sequence ids
    seqid_dir = f"{root}/all_seqids.csv"
    if bN_name == "cviral":
        seqid_dir = f"{root}/C_RVDB_seqids.csv"
    print(seqid_dir)
    return seqid_dir


def hs_clean_up_sseqids_preload(
    sample: str,
    github: str,
    db: str,
    analysis_directory: str,
    spp: str,
    hs_sseqid: str,
) -> pd.DataFrame:

    print("\n\nBLAST")
    unique_sseqids_list = []
    blastn_file = f"{analysis_directory}/sample_data/{sample}/blast*/trimmed_*{db}.hs_bn.tsv"
    print(blastn_file)
    blastn_files = glob.glob(blastn_file)
    if len(blastn_files) > 0:
        print(f"Sample with reads: {sample}")
        blastn_file = blastn_files[0]
        blastn_import_df_one = hs_blastn_import(blastn_file)
        if blastn_import_df_one is not None:
            unique_sseqid = blastn_import_df_one["sseqid"].unique()
            unique_sseqids_list.append(unique_sseqid)

    unique_sseqids_list_out = list(set(x for lst in unique_sseqids_list for x in lst))
    seqid_dir = getseqid(github, db)
    sseqids = import_deanonymised_nomenclature(seqid_dir)
    sseqids_list = sseqids.loc[sseqids["sseqid"].isin(unique_sseqids_list_out)]

    # save the relevant seqids to individual files for later use during multiprocessing
    blastn_files = glob.glob(blastn_file)
    if len(blastn_files) > 0:
        blastn_file = blastn_files[0]
        blastn_import_df_two = hs_blastn_import(blastn_file)
        if blastn_import_df_two is not None:
            print(f"\n{hs_sseqid}")
            sseqids_from_dict = sseqids.loc[
                sseqids["sseqid"].isin([hs_sseqid])
            ]

            seqid_deanonymised = blastn_import_df_two.merge(
                sseqids_from_dict, on="sseqid", how="inner"
            )

            print(f"Read count: {len(seqid_deanonymised)}")
            out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{hs_sseqid}/"
            sample_subset = f"{out_dir}/hs_sample_target_species.tsv"
            print(f"Saving: {sample_subset}")
            seqid_deanonymised.to_csv(sample_subset, index=False)

            sseqids_list.to_csv(f"{out_dir}/hs-sseqids-list.csv", index=False)

    print("Completed loading in sseqids and deanonymising.\n")
    return sseqids_list


def ce_clean_up_sseqids_preload(
    sample: str,
    github: str,
    db: str,
    analysis_directory: str,
    spp: str,
    ce_sseqid: str,
) -> pd.DataFrame:

    print("\n\nCentrifuge")
    unique_sseqids_list = []
    ce_file = f"{analysis_directory}/sample_data/{sample}/centrifuge/{sample}*{db}*troubleshooting_report.tsv"
    print(ce_file)
    ce_files = glob.glob(ce_file)
    if len(ce_files) > 0:
        print(f"Sample with reads: {sample}")
        troubleshooting = ce_files[0]

        if os.stat(troubleshooting).st_size != 0:
            df = pd.read_csv(troubleshooting, delimiter="\t")
            df = df.rename(columns={"seqID": "sseqid"})

            if df is not None:

                unique_sseqid = df["sseqid"].unique()
                unique_sseqids_list.append(unique_sseqid)

    unique_sseqids_list_out = list(set(x for lst in unique_sseqids_list for x in lst))
    seqid_dir = getseqid(github, db)
    sseqids = import_deanonymised_nomenclature(seqid_dir)

    sseqids_list = sseqids.loc[sseqids["sseqid"].isin(unique_sseqids_list_out)]

    # save the relevant seqids to individual files for later use during multiprocessing
    ce_files = glob.glob(ce_file)

    if len(ce_files) > 0:
        troubleshooting = ce_files[0]

        if os.stat(troubleshooting).st_size != 0:
            df2 = pd.read_csv(troubleshooting, delimiter="\t")
            df2 = df2.rename(columns={"seqID": "sseqid"})

            if df2 is not None:
                print(f"\n{ce_sseqid}")
                sseqids_from_dict = sseqids.loc[
                    sseqids["sseqid"].isin([ce_sseqid])
                ]

                seqid_deanonymised = df2.merge(
                    sseqids_from_dict, on="sseqid", how="inner"
                )

                print(f"Read count: {len(seqid_deanonymised)}")
                out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{ce_sseqid}/"
                sample_subset = f"{out_dir}/ce_sample_target_species.tsv"
                print(f"Saving: {sample_subset}")
                seqid_deanonymised.to_csv(sample_subset, index=False)

                sseqids_list.to_csv(f"{out_dir}/ce-sseqids-list.csv", index=False)

    print("Completed loading in sseqids and deanonymising.\n")
    return sseqids_list


#############################################

datetime_cleaner = re.compile(r"(\.\d+)")
def import_time(i: str) -> float:
    # before_format = "2021-04-21T07:46:10Z" # before
    # datetime.strptime(before_format, '%Y-%m-%dT%H:%M:%SZ').timestamp() # 1618962370.0

    # # They added milliseconds in their great wisdom
    # after_format = '2021-12-11T04:07:28.832003-05:00' # new
    # efter_format = '2022-01-11T13:42:02.067140-05:00' # new
    # # this fails as is ...
    # # datetime.strptime(after_format, '%Y-%m-%dT%H:%M:%S%z').timestamp() # 1639213648.0
    if "." in i:
        # Nanopore changed the timekeeping to include milliseconds
        remove_milliseconds = datetime_cleaner.findall(i)[0]
        removed_ms = i.replace(remove_milliseconds, "")
        read_time = datetime.strptime(removed_ms, "%Y-%m-%dT%H:%M:%S%z").timestamp()
        return read_time
    else:
        try:
            read_time = datetime.strptime(i, "%Y-%m-%dT%H:%M:%SZ").timestamp()
            return read_time

        except:
            return 0


def fix_broken_time(time_to_fix: list) -> list:
    times_fixed = []
    for i in time_to_fix:
        read_time = import_time(i)
        times_fixed.append(read_time)
    return times_fixed


def loop(data_input: list, end: int, start: int) -> list:
    lst = []
    for i in data_input:
        if isinstance(i, float):
            time_since_start = ((end - start) - (end - i)) / 60
            lst.append(time_since_start)
        if isinstance(i, str):
            read_time = import_time(i)
            time_since_start = ((end - start) - (end - read_time)) / 60
            lst.append(time_since_start)

    clean_list = [round(x) for x in lst]
    return clean_list


def process(lines=None):
    ks = ["name", "sequence", "optional", "quality"]
    return {k: v for k, v in zip(ks, lines)}


from itertools import zip_longest

# saw this in the python docs. looks like exactly what you need
def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


# compiled_time = re.compile(r"start_time=(\d+\-\d+\-.+Z)")
compiled_time = re.compile(r"start_time=(\d+\-\d+\-.+)\sflow")
compiled_read = re.compile(r"^@(\w+\-\w+\-\w+\-\w+\-\w+)")
def gen_fq(fastq_file: str, read_ids: list) -> (pd.DataFrame, pd.DataFrame):
    found_readIDs = {}
    with open(fastq_file, "r") as a:
        for readID, sequence, plus, quality in grouper(a, 4, None):
            timestamp = compiled_time.findall(readID)
            if len(timestamp) > 0:
                timestamp = timestamp[0]
                found_readID = compiled_read.findall(readID)[0]
                if '_' in found_readID:
                    found_readID = found_readID.split("_")[0]
                found_readIDs[found_readID] = timestamp

    v_timestamps = ([found_readIDs[x] for x in read_ids if x in found_readIDs])

    if len(v_timestamps) > 0:
        all_df = pd.DataFrame(found_readIDs.values(), columns = ['timestamp'])
        AA_df = pd.DataFrame(v_timestamps, columns = ['timestamp'])

        return AA_df, all_df

    else:
        return pd.DataFrame(), pd.DataFrame()


def run_datagen(
    sample_subset: str,
    adventitious_agent: str,
    fastq_in: str,
    sample: str,
    analysis_directory: str,
) -> (list, list):

    print(f"Loading {sample_subset} summary file into df.")

    # import seqid_deanonymised file
    seqid_deanonymised = pd.read_csv(sample_subset)
    seqid_deanonymised = seqid_deanonymised.rename(columns={"qseqid": "readID"})

    if len(seqid_deanonymised["name"].unique()) > 0:

        readids = seqid_deanonymised['readID']
        readids = readids.str.replace('_\d', '', regex=True)

        if not readids.empty:

            if len(readids) > 0:
                # spp_times, all_times = generate_fastq_df(
                #     fastq_in, readids, sample, analysis_directory
                # )

                spp_times, all_times = gen_fq(fastq_in, readids)

                if not spp_times.empty:
                    pred_time = list(all_times.timestamp)
                    spp_times = list(spp_times.timestamp)

                    times = fix_broken_time(pred_time)
                    spp_times_fix = fix_broken_time(spp_times)

                    start = min(times)
                    end = max(times)

                    t2minute_targ = loop(spp_times_fix, end, start)
                    t2minute_all_ = loop(pred_time, end, start)
                    # print(len(t2minute_targ), len(t2minute_all_))

                    return t2minute_targ, t2minute_all_
                else:
                    return [], []
    else:
        return [], []


def freq_pdf_cdf(raw_data: list) -> pd.DataFrame:
    s = pd.Series(raw_data, name="time")
    df = pd.DataFrame(s)

    # Frequency
    stats_df = (
        df.groupby("time")["time"]
        .agg("count")
        .pipe(pd.DataFrame)
        .rename(columns={"time": "frequency"})
    )

    # PDF
    stats_df["pdf"] = stats_df["frequency"] / sum(stats_df["frequency"])
    # CDF
    stats_df["cdf"] = stats_df["pdf"].cumsum()
    stats_df = stats_df.reset_index()
    # calculate time between read detection (minute)
    stats_df["dt"] = stats_df["time"].diff()
    # account for frequency if more than 1 predicted read in a given minute
    stats_df["dtdf"] = stats_df["dt"] / stats_df["frequency"]

    return stats_df


def multiprocess_sample(
    analysis_directory: str,
    all_samples: list,
) -> None:

    sample, db, spp, sseqid = all_samples

    a = f"_{sample.split('_')[3]}CFU"
    b = f"_{sample.split('_')[3]}"
    sample = sample.replace(b, a)

    sample_folder = f"{analysis_directory}/sample_data/{sample}/"
    out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{sseqid}/"

    if not Path(f"{out_dir}/Cumulative_frequency.csv").is_file():

        if os.path.isdir(sample_folder):

            if os.path.isdir(f"{sample_folder}/trimmed"):
                fastq_file = (f"{sample_folder}/trimmed/trimmed_*.fastq")

                if os.path.isfile(f"{sample_folder}/trimmed/no_host_{sample}.fastq"):
                    fastq_file = f"{sample_folder}/trimmed/no_host_{sample}.fastq"

                fastq_globs = glob.glob(fastq_file)

                if len(fastq_globs) > 0:
                    fastq_file = fastq_globs[0]

                    sample_subset = f"{out_dir}/*_sample_target_species.tsv"
                    sts_globs = glob.glob(sample_subset)

                    if len(sts_globs) > 0:
                        file = sts_globs[0]
                        t2minute, t2minute_all = run_datagen(
                            file,
                            sseqid,
                            fastq_file,
                            sample,
                            analysis_directory,
                        )

                        if len(t2minute) > 0:
                            counts = Counter(t2minute)
                            df_m = pd.DataFrame.from_dict(
                                counts, orient="index"
                            ).reset_index()
                            df_m = df_m.rename(columns={"index": "time", 0: "spp_count"})

                            counts_all = Counter(t2minute_all)
                            df_a = pd.DataFrame.from_dict(
                                counts_all, orient="index"
                            ).reset_index()
                            df_a = df_a.rename(columns={"index": "time", 0: "all_count"})

                            merged = pd.merge(df_a, df_m, on="time", how="outer")
                            # e.g. 2000 total reads / 200 MVM 10:1 ratio; 2000 total reads / 1000 MVM reads 2:1 ratio; 2000 total reads / 20 MVM reads 100:1 ratio
                            # this means that a higher ratio is worse than a lower one.
                            # i.e. 2:1 is good for desired species e.g. MVM
                            # while 100:1 implies there are many contaminants from host or incorrect predictions
                            merged["ratio"] = merged["all_count"] / merged["spp_count"]

                            cum_df = freq_pdf_cdf(t2minute)

                            print(f"Saving to: {out_dir}/Cumulative_frequency.csv")

                            if not merged.empty:
                                merged.to_csv(
                                    f"{out_dir}/Summarised_counts.csv",
                                    index=False)

                            if not cum_df.empty:
                                cum_df.to_csv(
                                    f"{out_dir}/Cumulative_frequency.csv",
                                    index=False)

#############################################

def hs_multiprocess_extract_reads(
    github: str,
    analysis_directory: str,
    save_directory: str,
    hs_sample_training_lol: list) -> None:

    sample, db, spp, hs_sseqid = hs_sample_training_lol

    a = f"_{sample.split('_')[3]}CFU"
    b = f"_{sample.split('_')[3]}"
    sample = sample.replace(b, a)
    samp_dir = f"{save_directory}/{sample}/"
    os.makedirs(samp_dir, exist_ok=True)

    db_dir = f"{samp_dir}/{db}/"
    os.makedirs(db_dir, exist_ok=True)

    spp_dir = f"{db_dir}/{hs_sseqid}/"
    os.makedirs(spp_dir, exist_ok=True)

    out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{hs_sseqid}/"
    hs_sample_seqids = f"{out_dir}/hs_sample_target_species.tsv"

    if not Path(hs_sample_seqids).is_file():
        hs_clean_up_sseqids_preload(sample, github, db, analysis_directory, spp, hs_sseqid)

#############################################

#############################################

def ce_multiprocess_extract_reads(
    github: str,
    analysis_directory: str,
    save_directory: str,
    ce_sample_training_lol: list) -> None:

    sample, db, spp, ce_sseqid = ce_sample_training_lol

    a = f"_{sample.split('_')[3]}CFU"
    b = f"_{sample.split('_')[3]}"
    sample = sample.replace(b, a)
    samp_dir = f"{save_directory}/{sample}/"
    os.makedirs(samp_dir, exist_ok=True)

    db_dir = f"{samp_dir}/{db}/"
    os.makedirs(db_dir, exist_ok=True)

    spp_dir = f"{db_dir}/{ce_sseqid}/"
    os.makedirs(spp_dir, exist_ok=True)

    out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{ce_sseqid}/"
    ce_sample_seqids = f"{out_dir}/ce_sample_target_species.tsv"

    if not Path(ce_sample_seqids).is_file():
        ce_clean_up_sseqids_preload(sample, github, db, analysis_directory, spp, ce_sseqid)

#############################################

def run_multiprocessing(save_directory: str, analysis_directory: str, github: str, input_type: str) -> list:
    ######### BLAST MULTIPROCESSING EXTRACT READS #########
    #######################################################

    hs_sample_names_training = f"{save_directory}/samples-{input_type}-BLAST"
    with open(f"{hs_sample_names_training}.json") as json_data:
        hs_sample_training_lol = json.loads(json_data.read())

    # run for BLAST samples.

    func1 = partial(
        hs_multiprocess_extract_reads, github, analysis_directory, save_directory
    )
    with Pool(processes=30) as p:
        p.map(func1, hs_sample_training_lol)  # process data_inputs iterable with pool


    ######### CENTRIFUGE MULTIPROCESSING EXTRACT READS #########
    ############################################################

    ce_sample_names_training = f"{save_directory}/samples-{input_type}-ce"
    with open(f"{ce_sample_names_training}.json") as json_ce:
        ce_sample_training_lol = json.loads(json_ce.read())

    # # run for centrifuge samples.

    func2 = partial(
        ce_multiprocess_extract_reads, github, analysis_directory, save_directory
    )
    with Pool(processes=30) as p:
        p.map(func2, ce_sample_training_lol)  # process data_inputs iterable with pool


    ######### TIME TO NEXT READ MULTIPROCESSING #########
    #####################################################

    print("\n######### TIME TO NEXT READ MULTIPROCESSING #########\n")

    all_training_samples = hs_sample_training_lol + ce_sample_training_lol

    func3 = partial(
        multiprocess_sample, analysis_directory
    )
    with Pool(processes=10) as p:
        p.map(func3, all_training_samples)  # process data_inputs iterable with pool

    return all_training_samples


def main(analysis_directory: str, input_type: str, github: str):
    save_directory = f"{analysis_directory}/T2Read/"
    os.makedirs(save_directory, exist_ok=True)

    all_training_samples = run_multiprocessing(save_directory, analysis_directory, github, input_type)

    de_cum_dfs = []
    cum_dfs = []

    for individual_sample in all_training_samples:

        sample, db, spp, sseqid = individual_sample

        a = f"_{sample.split('_')[3]}CFU"
        b = f"_{sample.split('_')[3]}"
        sample = sample.replace(b, a)

        out_dir = f"{analysis_directory}/T2Read/{sample}/{db}/{sseqid}/"

        cum_freq = f"{out_dir}/Cumulative_frequency.csv"

        if Path(cum_freq).is_file():

            if os.stat(cum_freq).st_size != 0:

                cum_df = pd.read_csv(cum_freq)
                cum_df["sample"] = sample
                cum_df["db"] = db
                cum_df["spp"] = spp
                cum_df["sseqid"] = sseqid

                # fix broken time values
                if cum_df.loc[cum_df["time"] > 2880]["time"].any():
                    a_df = cum_df.loc[cum_df["time"] > 2880]
                    b_df = cum_df.loc[cum_df["time"] <= 2880]
                    min_a = a_df.groupby(["sample"])["time"].min().to_dict()

                    dfs = [b_df]
                    for k,v in min_a.items():
                        aa = a_df.loc[a_df["sample"] == k]
                        aa['time'] = aa['time'] - v
                        dfs.append(aa)
                    cum_df = pd.concat(dfs)
                    # /end fix broken time values

                cum_dfs.append(cum_df)

                describe_cum_df = cum_df.groupby(["sample", "db", "spp", "sseqid"])[
                                        [
                                            "time",
                                            "frequency",
                                            "pdf",
                                            "cdf",
                                            "dt",
                                            "dtdf",
                                        ]
                                    ].describe()
                describe_cum_df.columns = ["_".join(a) for a in describe_cum_df.columns.to_flat_index()]
                describe_cum_df.reset_index(inplace=True)
                de_cum_dfs.append(describe_cum_df)

    cat_cum_dfs = pd.concat(cum_dfs)
    cat_de_cum_dfs = pd.concat(de_cum_dfs)
    cat_de_cum_dfs = cat_de_cum_dfs.drop_duplicates()

    print(f"Saving concatenated cdfs to {analysis_directory}/T2Read/{input_type}_concatenated_CDFs.csv")
    cat_cum_dfs.to_csv(f"{analysis_directory}/T2Read/{input_type}_complete_concatenated_CDFs.csv", index = False)
    cat_de_cum_dfs.to_csv(f"{analysis_directory}/T2Read/{input_type}_concatenated_CDFs.csv", index = False)


if __name__ == "__main__":
    analysis_directory = "/mnt/d/"

    input_type = "tt"
    github = "~/GitHub/"
    main(analysis_directory, input_type, github)

# time ./T2Read.py
