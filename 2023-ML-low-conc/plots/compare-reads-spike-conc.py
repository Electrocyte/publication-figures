# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:11:14 2023

@author: mangi
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator
import seaborn as sns
import numpy as np

dir_ = "D:/DropBox/AA SMART/Papers/My Papers/Submission-ASM-Spectrum/Supplementary/Table-S2-Centrifuge-metagenomic-summary.csv"
ddir_ = "D:/DropBox/AA SMART/Papers/My Papers/Submission-ASM-Spectrum/Supplementary/dPCR.csv"

df = pd.read_csv(dir_)

orgs = {
            "Pseudomonas aeruginosa": ["PAO1", "9027"],
            "Escherichia coli": ["EC"],
            "Cutibacterium acnes": ["Cacnes", "Pacnes"],
            "Klebsiella pneumoniae": ["Kleb"],
            "Staphylococcus aureus": ["Saureus"],
            "Bacillus subtilis": ["Bsubtilis"],
            "Candida albicans": ["Calbicans"],
            "Clostridium": ["clost"],
        }

TPs = []
for name, strains in orgs.items():
    for strain in strains:
        TP = df.loc[(df["name"].str.contains(name)) & (df["sample"].str.contains(strain))]
        print(name, len(TP))
        if "Clostridium" in name:
            TP = TP.loc[TP["name"] .str.contains ("Clostridium botulinum")]
        TPs.append(TP)
cat_TPs = pd.concat(TPs)
cat_TPs["sname"] = cat_TPs["sample"].str.split("_", expand = True)[2]
cat_TPs["conc"] = cat_TPs["sample"].str.split("_", expand = True)[3]
cat_TPs = cat_TPs[["sname", "name", "sample", "numUniqueReads", "score_count","conc"]]
cat_10_TPs = cat_TPs.loc[cat_TPs["conc"] .isin(["10", "100"])]
groups = cat_10_TPs.groupby(["name", "sample"]).max()
groups.reset_index(inplace = True)
groups["gs"] = groups["name"].str.split(" ", expand = True)[0] +" "+ groups["name"].str.split(" ", expand = True)[1]
# groups["sname-spp"] = groups["sample"].str.split("_", expand = True)[2] + " " + groups["gs"]
groups = groups.loc[groups["sample"] .str.contains("TC")]

# g10 = groups.loc[groups["conc"].str.contains("10")]
# g10.reset_index(inplace = True)
# g100 = groups.loc[groups["conc"].str.contains("100")]
# g100.reset_index(inplace = True)


f, ax = plt.subplots(figsize=(20, 15))

# ax = sns.scatterplot(data = groups, x="sname", y="score_count", color="blue", hue="gs", s=300, style="conc")
ax = sns.boxplot(data = groups, x="gs", y="score_count", hue="conc", palette = "Blues")

ax.set_yscale("log")
ax.set_xlabel("Spike organism", size=40)
ax.set_ylabel("Read Count", size=40)
ax.set_ylim([1e0, 1e6])

# plt.title("Comparison of read counts for different \nspecies at 10 & 100 CFU / mL samples.", size=40)

plt.rcParams['legend.title_fontsize'] = 30
plt.xticks(rotation=90)
plt.tight_layout()
ax.tick_params(axis="x", labelsize=30)
ax.tick_params(axis="y", labelsize=30)

ax.legend(title="Spike concentration", prop={"size": 25}, markerscale = 5)#, bbox_to_anchor=[1,0.7])
plt.show()


ddf = pd.read_csv(ddir_)

f, ax2 = plt.subplots(figsize=(20, 15))

ax2 = sns.barplot(data = ddf, x="Spike Concentration", y="Mean concentration (copies/μL)", color="blue")
# ax2 = sns.scatterplot(data = ddf, x="Spike Concentration", y="Mean concentration (copies/μL)", color="blue", s=300)

# ax2.set_xscale("log")
# ax2.set_xlim([1e0, 1e5])

ax2.set_yscale("log")
ax2.set_xlabel("Spike concentration (CFU/mL)", size=40)
ax2.set_ylabel("Mean concentration (copies/μL)", size=40)
ax2.set_ylim([1e8, 1e12])
ax2.tick_params(axis="x", labelsize=30)
ax2.tick_params(axis="y", labelsize=30)

# Define major ticks from 10^8 to 10^11
major_ticks = np.logspace(0, 12, num=13) # 10^8, 10^9, 10^10, 10^11

# Set major ticks on the y-axis
ax2.yaxis.set_major_locator(FixedLocator(major_ticks))

# Set minor ticks on the y-axis
ax2.yaxis.set_minor_locator(LogLocator(subs='all'))

plt.show()
