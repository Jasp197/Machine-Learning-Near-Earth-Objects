import warnings

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Set Pandas options to display numbers with 3-digit decimal precision
pd.set_option('display.float_format', '{:.3f}'.format)


# ----------------------------------------
# Phase IV: Clustering and Association
# ----------------------------------------


# K Means
def kMeansAlgo(X):
    # Initialize lists to store silhouette scores and within-cluster variations
    silhouette_scores = []
    within_cluster_variations = []

    # Range of k values to test
    k_values = range(2, 11)  # Change the range as needed

    # Perform K-means clustering for different k values
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=5805)
        kmeans.fit(X)

        # Compute silhouette score for each k
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

        # Compute within-cluster variation (inertia) for each k
        within_cluster_variations.append(kmeans.inertia_)

    # Plotting Silhouette scores for different k values
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('K Means: Silhouette Analysis')

    # Plotting within-cluster variation for different k values
    plt.subplot(1, 2, 2)
    plt.plot(k_values, within_cluster_variations, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-cluster variation')
    plt.title('K Means: Within-Cluster Variation')

    plt.tight_layout()
    plt.show()


# ----------------------------------------

# Apriori Algorithm

# Our data is not suitable for apriori algorithm, but I will provide an example of
# how the apriori algorithm works with binary transaction like data

def aprioriAlgo(X):
    # Define a threshold
    threshold = 0

    # Apply function to convert values to binary
    binary_df = X.applymap(lambda x: 1 if x > threshold else 0)

    # Find frequent itemsets with minimum support threshold
    frequent_itemsets = apriori(binary_df, min_support=0.2, use_colnames=True, verbose=1)
    print("\nFrequent Itemsets:\n", frequent_itemsets)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    print("Association Rules:\n", rules.to_string())
