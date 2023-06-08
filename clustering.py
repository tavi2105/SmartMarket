import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import AgglomerativeClustering

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("./data/rules.csv", header=None, delimiter=",")

item_distinct = [a for a in set(df[0])]
print(len(item_distinct))

first_column = [item_distinct.index(item) for item in df[0]]
second_column = [item_distinct.index(item) for item in df[1]]
print(len(first_column))

maxim_lift = df[2].max()
matrix = [[0 for x in range(len(item_distinct))] for y in range(len(item_distinct))]
for item1 in item_distinct:
    for item2 in item_distinct:
        if item1 != item2:
            score = df[(df[0] == item1) & (df[1] == item2)][2]
            if not score.empty:
                matrix[item_distinct.index(item1)][item_distinct.index(item2)] = maxim_lift - score.values[0]
            else:
                matrix[item_distinct.index(item1)][item_distinct.index(item2)] = maxim_lift + 1

Agg_hc = AgglomerativeClustering(n_clusters=20, metric='euclidean', linkage='complete')
clusters = Agg_hc.fit_predict(matrix)

product_zone = {}
for index, cluster in enumerate(clusters):
    if cluster in product_zone:
        product_zone[cluster].append(item_distinct[index])
    else:
        product_zone[cluster] = [item_distinct[index]]

for index1 in range(0, 20):
    print(index1, ' -> ', product_zone[index1])

cluster_dist = [[0 for x in range(20)] for y in range(20)]
for index1 in range(0, 20):
    for index2 in range(0, 20):
        if index1 != index2:
            nr = 0
            total = 0
            for item1 in product_zone[index1]:
                for item2 in product_zone[index2]:
                    score = df[(df[0] == item1) & (df[1] == item2)][2]
                    if not score.empty:
                        total = total + score.values[0]
                        nr = nr + 1
                    # else:

            if nr != 0:
                cluster_dist[index1][index2] = total / nr


# print(cluster_dist)

Agg_hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='complete')
clusters_level2 = Agg_hc.fit_predict(cluster_dist)


print(clusters_level2)
