#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:44:10 2023

@author: mangi
"""

from typing import List, Dict
from scipy.stats import norm
from scipy import stats
from statistics import mean, stdev
import seaborn as sns
import matplotlib.pyplot as plt
import math
import glob
import os
import pandas as pd
import matplotlib.axes


def annotate_subset(pmin: int, hue: str, ax: matplotlib.axes.Axes, subset: pd.DataFrame, annotated_hues: set, x: str, y: str, prev_confidence: float = None) -> None:
    """
    Annotate the first point in the subset where confidence is >=99 with
    a magenta arrow on the specified axes for each hue.
    Do not plot an arrow if the first value is already 99% and an arrow was plotted in the previous subset.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to annotate the subset.
    subset (pd.DataFrame): The subset of the data to be annotated.
    annotated_hues (set): The set of hues that have already been annotated.
    x (str): Column name for x-axis.
    y (str): Column name for y-axis.
    prev_confidence (float, optional): The confidence value of the last point in the previous subset.

    Returns:
    None
    """

    for index, row in subset.iterrows():
        # If confidence >= 99 and hue not annotated yet
        if row[y] >= pmin and row[hue] not in annotated_hues:
            # If first value is already >= 99, and previous confidence was also >= 99, skip
            if index == subset.index[0] and (prev_confidence is None or prev_confidence >= pmin):
                continue
            ax.annotate('', xy=(row[x], row[y]),
                        xytext=(row[x], row[y] - 10),  # keep x the same
                        arrowprops=dict(facecolor='magenta', shrink=0.05))
            # Annotate with the actual value just below the arrow
            if x == "Reads":
                ax.annotate(f"{row[x]:.0f}", xy=(row[x], row[y] - 12),
                        color='black', ha='center', fontsize=12, rotation=45) # Center alignment and fontsize of 12 for the text

            annotated_hues.add(row[hue])  # add the hue to the set of annotated hues



def sequencing_rate(t: float, r_initial: float, r_plateau: float, k: float) -> float:
    """
    Compute the sequencing rate at time t using an exponential decay model.

    :param t: The time at which the sequencing rate is computed.
    :param r_initial: The initial sequencing rate.
    :param r_plateau: The plateau sequencing rate.
    :param k: The decay constant.
    :return: The sequencing rate at time t.

    :Example:

    >>> sequencing_rate(2, 10, 2, 0.5)
    3.572406063565237
    """
    return r_plateau + (r_initial - r_plateau) * math.exp(-k * t)


def hypergeometric_pmf_approx(k: int, N: int, K: int, n: int) -> float:
    """
    Approximate hypergeometric probability mass function.

    :param k: observed successes (number of virus reads)
    :param N: total number of items
    :param K: total number of successes (total virus reads)
    :param n: number of draws
    :return: the calculated probability
    """
    p = K / N
    # Use normal approximation if N is large
    # if (K > 5 and N - K > 5) or (N > 20 and 0.1 < p < 0.9):
    if N > 500 or (K > 5 and N - K > 5): # or (N > 100 and 0.1 < p < 0.9):
        # Use normal approximation
        mean = n * p
        variance = n * p * (1 - p)
        std_dev = math.sqrt(variance)
        return norm.pdf(k, mean, std_dev)
    else:
        return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)


def detection_confidence(N: int, K: int, n: int) -> float:
    """
    Calculate detection confidence.

    :param N: total number of items
    :param K: total number of successes (total virus reads)
    :param n: number of draws
    :return: the detection confidence
    """
    return 1 - hypergeometric_pmf_approx(0, N, K, n)


def reads_needed_for_sensitivity(N: int, K: int, target_confidence: float = 0.99) -> int:
    """
    Calculate reads needed for sensitivity.

    :param N: total number of items
    :param K: total number of successes (total virus reads)
    :param target_confidence: desired confidence level
    :return: the number of reads needed for sensitivity
    """
    if K > N:
        raise ValueError("Number of virus reads (K) can't exceed the total number of reads (N).")
    n = 0
    while True:
        n += 1
        confidence = detection_confidence(N, K, n)
        if confidence >= target_confidence:
            return n


def calculate_sequencing_time(spp: str, group: pd.DataFrame, r_initial: float, r_plateau: float, k: float, L: int, desired_sensitivity: float = 0.99) -> List[Dict[str, float]]:
    """
    Calculate sequencing time.

    :param group: dataframe group by virus name
    :param r_initial: initial sequencing rate in bp/s
    :param r_plateau: plateau sequencing rate in bp/s
    :param k: decay constant
    :param L: average read length in bp
    :param desired_sensitivity: desired sensitivity
    :return: a list of dictionaries containing CFU and time needed for each row in the group
    """
    results = []
    for index, row in group.iterrows():
        N = row['totalUniqReads']  # Total reads
        K = row['numUniqueReads']  # Viral reads
        if K > 0:
            print(row["CFU"])
            required_reads = reads_needed_for_sensitivity(N, K, desired_sensitivity)
            t = 0
            cumulative_reads = 0
            while cumulative_reads < required_reads:
                R = sequencing_rate(t, r_initial, r_plateau, k) / L
                cumulative_reads += R
                t += 1  # Increase time by one unit (second)
            time_needed = t
            results.append({
                'name': spp,
                'CFU': row['CFU'],
                'time_needed': time_needed,
                'required_reads': required_reads
            })
    return results


def reads_at_time(t: float, r_initial: float, r_plateau: float, k: float, L: int) -> int:
    """
    Estimate the number of reads at time t.

    :param t: the time in seconds
    :param r_initial: initial sequencing rate in bp/s
    :param r_plateau: plateau sequencing rate in bp/s
    :param k: decay constant
    :param L: average read length in bp
    :return: estimated number of reads at time t
    """
    R = sequencing_rate(t, r_initial, r_plateau, k) / L
    return int(R * t)


def confidence_at_time(t: float, N: int, K: int, r_initial: float, r_plateau: float, k: float, L: int) -> float:
    """
    Estimate the confidence of sterility at time t.

    :param t: the time in seconds
    :param N: total number of items
    :param K: total number of successes (total virus reads)
    :param r_initial: initial sequencing rate in bp/s
    :param r_plateau: plateau sequencing rate in bp/s
    :param k: decay constant
    :param L: average read length in bp
    :return: estimated confidence at time t
    """
    n = reads_at_time(t, r_initial, r_plateau, k, L)
    return detection_confidence(N, K, n)


def confidence_vs_reads_range(N: int, K: int, reads_range: list) -> pd.DataFrame:
    """
    Calculate confidence for a range of reads.

    :param N: total number of items
    :param K: total number of successes (total virus reads)
    :param reads_range: a list containing a range of read numbers
    :return: DataFrame containing the number of reads and corresponding confidence
    """
    confidences = [detection_confidence(N, K, n) for n in reads_range]
    return pd.DataFrame({
        'Reads': reads_range,
        'Confidence': confidences
    })


def plot_species_confidence(df: pd.DataFrame, hue: str, x: str, y: str, x1, x2, x3) -> None:
    """
    Plot scatter plots of confidence over time for each species in the dataframe.

    The function divides the plot into three subplots:
    1. From 0 to 30 minutes
    2. From 30 minutes to 5 hours
    3. From 5 hours to 72 hours.

    Args:
    - df (pd.DataFrame): The input dataframe containing 'Species' and other specified columns for plotting.
    - hue (str): Column name for scatter plot hue (coloring).
    - x (str): Column name for x-axis.
    - y (str): Column name for y-axis.

    Returns:
    None: The function outputs the plots but does not return any value.

    Usage:
    >>> df = pd.read_csv('your_dataset.csv')
    >>> plot_species_confidence(df, hue='CFU', x='Time', y='Confidence')
    """

    unique_species = df['Species'].unique()

    for species in unique_species:
        # Filter dataframe for the current species
        df_species = df[df['Species'] == species]

        plt.rcParams.update({'font.size': 25})  # setting font size to 16

        # Create three subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.025}, figsize=(20, 15))

        annotated_hues = set()
        last_conf1 = df_species[df_species[x] <= x1][y].iloc[-1] if not df_species[df_species[x] <= x1].empty else None
        # Plot the first part in the first subplot (0 to 30 minutes)
        sns.scatterplot(data=df_species[(df_species[x] <= x1)], x=x, y=y, hue=hue, ax=ax1, s=200)
        ax1.set_xlim(0, x1)
        ax1.legend_.remove()  # remove the legend on ax1

        last_conf2 = df_species[(df_species[x] > x1) & (df_species[x] <= x2)][y].iloc[-1] if not df_species[(df_species[x] > x1) & (df_species[x] <= x2)].empty else None
        # Plot the second part in the second subplot (30 minutes to x2 hours)
        sns.scatterplot(data=df_species[(df_species[x] > x1) & (df_species[x] <= x2)], x=x, y=y, hue=hue, ax=ax2, s=200)
        ax2.set_xlim(x1, x2)
        ax2.legend_.remove()  # remove the legend on ax2

        last_conf3 = df_species[df_species[x] > x2][y].iloc[-1] if not df_species[df_species[x] > x2].empty else None
        # Plot the third part in the third subplot (x2 hours to x3 hours)
        sns.scatterplot(data=df_species[df_species[x] > x2], x=x, y=y, hue=hue, ax=ax3)
        ax3.set_xlim(x2, x3)

        annotate_subset(pmin, hue, ax1, df_species[(df_species[x] <= x1)], annotated_hues, x, y, prev_confidence=last_conf1)
        annotate_subset(pmin, hue, ax2, df_species[(df_species[x] > x1) & (df_species[x] <= x2)], annotated_hues, x, y, prev_confidence=last_conf2)
        annotate_subset(pmin, hue, ax3, df_species[(df_species[x] > x2)], annotated_hues, x, y, prev_confidence=last_conf3)

        # Hide y-ticks on ax2 and ax3
        ax2.set_yticks([])
        ax3.set_yticks([])

        # Assume ax3 is the AxesSubplot object received from sns.scatterplot
        legend = ax3.legend()
        plt.legend(markerscale=4, title="Spike \nConcentration", title_fontsize='30')

        # Set font size for the items in the legend
        for t in legend.get_texts():
            t.set_fontsize('25')  # Set the font size of the legend items

        # Add a zigzag or break mark at the boundary between subplots
        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(color='k', clip_on=False)  # arguments for the diagonal lines
        ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)  # top-left diagonal on ax1
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)  # top-right diagonal on ax1
        ax2.plot((1 - d, 1 + d), (-d, +d), transform=ax2.transAxes, **kwargs)  # top-left diagonal on ax2
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)  # top-right diagonal on ax2

        # Set title and labels
        plt.suptitle(f'Scatter Plot of Confidence over {x} for {species}', fontsize=35, y=0.93)
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('')
        ax1.set_ylabel('Confidence (%)', fontsize=30)

        # Set a single x-axis label for all subplots
        if hue == "Time":
            fig.text(0.5, 0.07, 'Time (hours)', ha='center', va='center', fontsize=30)
            # Manually set x-tick labels
            ax2.set_xticks([1, 2, 3, 4])
            ax2.set_xticklabels(['1', '2', '3', '4'])
            ax3.set_xticks([6, 20, 40, 60, 72])
            ax3.set_xticklabels(['6', '20', '40', '60', '72'])
        elif hue == "Reads":
            fig.text(0.5, 0.07, 'Reads', ha='center', va='center', fontsize=30)
            # Manually set x-tick labels
            ax2.set_xticks([100, 1e3, 4e3, 9e3])
            ax2.set_xticklabels(['100', '1000', '4000', '9000'])
            ax3.set_xticks([1e4, 1e5, 5e6, 1e6])
            ax3.set_xticklabels(['10000', '100000', '500000', '1000000'])

        ax1.set_yticks([0, 20, 40, 60, 80, 100])
        ax1.set_yticklabels(['0', '20', '40', '60', '80', '100'])

        # Show the plot
        plt.tight_layout()
        plt.show()


directory = "SequencingData/Viral_human/analysis/Confidence/"
rec_glob = glob.glob(f"{directory}recentrifuge-all.csv")
rec_df = pd.read_csv(rec_glob[0])

t = ["Feline leukemia virus", "Minute virus of mice", "Porcine circovirus 1"]
t_Df = rec_df.loc[rec_df["name"].isin(t)]

# usage
r_initial = 450  # initial sequencing rate in bp/s
r_plateau = 200  # plateau sequencing rate in bp/s
k = 0.01  # decay constant
L = 1000  # Assume average read length is 1000bp
pval = 0.99
pmin = 99

### TARGETED

felv_df = pd.read_csv(f"{directory}FELV-raw.csv")
mvm_df = pd.read_csv(f"{directory}MVM-raw.csv")
pcv1_df = pd.read_csv(f"{directory}PCV1-raw.csv")

all_data_df = pd.concat([felv_df, mvm_df, pcv1_df], ignore_index=True)
all_data_df.rename(columns={'spp': 'name', 'count':'numUniqueReads', 'reads-analysed':'totalUniqReads', 'concentration':'CFU'}, inplace=True)

eall_results = []
ecos = []
econf_dfs = []

save_name = f"{directory}.combined-raw.csv"
if not os.path.isfile(save_name):
    for spp, group in all_data_df.groupby("name"):
        g = group[["CFU", "numUniqueReads", "totalUniqReads"]]
        results = calculate_sequencing_time(spp, g, r_initial, r_plateau, k, L, pval)

        # Display results
        print(f"\n=== Results for {spp}: ===")
        for result in results:
            if result['time_needed'] < 120:
                print(f"At CFU {result['CFU']}, you need to sequence for {result['time_needed']:.2f} seconds (required reads: {result['required_reads']:.0f}) to achieve {pmin}% sensitivity.")
            elif result['time_needed'] < 3600:
                print(f"At CFU {result['CFU']}, you need to sequence for {result['time_needed']/60:.2f} minutes (required reads: {result['required_reads']:.0f}) to achieve {pmin}% sensitivity.")
            else:
                print(f"At CFU {result['CFU']}, you need to sequence for {result['time_needed']/3600:.2f} hours (required reads: {result['required_reads']:.0f}) to achieve {pmin}% sensitivity.")
        eall_results.append(results)

        # Generate timepoints every 10 minutes in seconds
        timepoints = list(range(600, 86400*3, 600))  # from 10 minutes to 3 days

        # Generate additional timepoints every minute between 0 to 2 hours
        additional_timepoints = list(range(30, 1800, 30))  # from 1 minute to 2 hours

        # Combine and sort all the timepoints
        timepoints = sorted(set(timepoints + additional_timepoints))

        group_means = g.groupby('CFU').mean()

        print("\n")
        for index, row in group_means.iterrows():

            N = row['totalUniqReads']  # Total reads
            K = row['numUniqueReads']  # Viral reads

            reads_range = list(range(1, int(1e2), 1)) + list(range(int(1e2), int(1e4), 100)) + list(range(int(1e4), int(1e6), 1000)) # Adjust this range as needed
            conf_df = confidence_vs_reads_range(N, K, reads_range)
            conf_df["Confidence"] = conf_df["Confidence"].apply(lambda x: max(0, x))  # Ensure confidence is not below 0
            conf_df["CFU"] = index
            conf_df["Species"] = spp
            econf_dfs.append(conf_df)

            for t in timepoints:
                conf = confidence_at_time(t, N, K, r_initial, r_plateau, k, L)
                if conf < 0:
                    conf = 0
                ecos.append([spp, index, t/3600, conf*100])
    efinal_conf_df = pd.concat(econf_dfs, ignore_index=True)
    efinal_conf_df["Confidence"] = efinal_conf_df["Confidence"]*100

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(ecos, columns=['Species', 'CFU', 'Time', 'Confidence'])

    plot_species_confidence(efinal_conf_df, "CFU", "Reads", "Confidence", 6e2, 2e4, 1e6)
    plot_species_confidence(df, "CFU", "Time", "Confidence", 0.5, 5, 72)


    concs = []
    for conc, ggroup in all_data_df.groupby("CFU"):
        gg = ggroup[["name", "CFU", "numUniqueReads", "totalUniqReads"]]
        cresults = calculate_sequencing_time("All", gg, r_initial, r_plateau, k, L, pval)
        concs .append(cresults)
    # Flatten the list of lists
    flattened_list = [item for sublist in concs for item in sublist]
    flattened_list2 = [item for sublist in eall_results for item in sublist]

    # Convert the flattened list to a DataFrame
    conc_df = pd.DataFrame(flattened_list)
    mean_conc = conc_df.groupby(["CFU"]).mean().reset_index()
    mean_conc["name"] = "All"

    indi_df = pd.DataFrame(flattened_list2)


    cat_df = pd.concat([conc_df, indi_df])
    cat_df.to_csv()

else:
    cat_df = pd.read_csv(save_name)

order = ['virus-control2', 'baseline 1 - 1', 'baseline 0.1 - 1', '10 - 1', '1 - 1',
         '0.1 - 1', '0.01 - 1',  '0.001 - 1']

palette = {
    "All": "blue",
    "FeLV (8.4 KB)": "#d9d9d9",  # Light grey
    "MVM (5000bp)": "#a6a6a6",  # Medium grey
    "PCV1-double (3.4 kb)": "#595959"  # Dark grey
}

# Create the bar plot
f, ax = plt.subplots(figsize=(20, 20))

plt.rcParams['legend.fontsize'] = 30

ax = sns.boxplot(x="CFU", y="required_reads", hue="name", data=cat_df, order=order, palette=palette)
ax.set_yscale("log")
plt.ylim(1, 5e8)
plt.xticks(rotation=90, ha='right')

plt.xlabel("Spiking concentration", fontsize = 35)
plt.ylabel("Reads for 99% confidence", fontsize = 35)
plt.title("Required Sequencing Reads vs. Spiking Concentration", fontsize = 40)

plt.xticks(fontsize=30)  # Change the number 12 to your desired font size
plt.yticks(fontsize=30)  # Change the number 12 to your desired font size


plt.tight_layout()
plt.legend(title="Organism(s)", loc="upper left")
plt.show()
