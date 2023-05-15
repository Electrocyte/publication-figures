# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:11:14 2023

@author: mangi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dir_ = "D:/Table-S2-Centrifuge-metagenomic-summary.csv"

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
cat_10_TPs = cat_TPs.loc[cat_TPs["conc"] == "10"]
groups = cat_10_TPs.groupby(["name", "sample"]).max()
groups.reset_index(inplace = True)
groups["gs"] = groups["name"].str.split(" ", expand = True)[0] +" "+ groups["name"].str.split(" ", expand = True)[1]
groups = groups.loc[groups["sample"] .str.contains("TC")]


f, ax = plt.subplots(figsize=(15, 15))

ax = sns.scatterplot(data = groups, x="sname", y="score_count", color="blue", hue="gs", s=300)

ax.set_yscale("log")
ax.set_xlabel("Sample", size=40)
ax.set_ylabel("Read Count", size=40)
ax.set_ylim([0, 1e6])

plt.title("Comparison of read counts for 10 CFU / mL samples.\n", size=40)

plt.rcParams['legend.title_fontsize'] = 30
plt.xticks(rotation=90)
plt.tight_layout()
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)

ax.legend(title="Spiked Species", prop={"size": 25}, markerscale = 5)#, bbox_to_anchor=[1,0.7])
plt.show()