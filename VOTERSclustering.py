from sklearn.metrics import calinski_harabasz_score, silhouette_score
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, ward
import pandas as pd
import numpy as np
from itertools import combinations

def cluster_performance_plots(silhouette_scores, calinski_harabasz_scores, no_clusters):
    fig, ax1 = plt.subplots()
    plt.title('Clustering Performance Metrics')

    color = 'tab:red'
    ax1.set_xlabel('# Clusters')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(list(range(2,(no_clusters+1))), silhouette_scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0,1])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Calinski-Harabasz Score', color=color)  # we already handled the x-label with ax1
    ax2.plot(list(range(2,(no_clusters+1))), calinski_harabasz_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return plt.show()

def dendrogram_plot(data_to_cluster, no_clusters):
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('distance')
    plt.ylabel('sample index or (cluster size)')
    linkage_array = ward(data_to_cluster)
    dendrogram(linkage_array, p=no_clusters, truncate_mode="lastp", orientation='left')
    return plt.show()

def performance_per_no_clusters(data_to_cluster, no_clusters):
    silhouette_scores = []
    calinski_harabasz_scores = []
    cluster_assignments_list = []
    for k in tqdm(range(2,(no_clusters+1)), desc="Clustering"):
        aggclus = AgglomerativeClustering(linkage = "ward", n_clusters=k)
        aggclus.fit(data_to_cluster)
        cluster_assignments = aggclus.fit_predict(data_to_cluster)
        cluster_assignments_list.append(cluster_assignments)
        silhouette_scores.append(silhouette_score(data_to_cluster, cluster_assignments))
        calinski_harabasz_scores.append(calinski_harabasz_score(data_to_cluster, cluster_assignments))
    cluster_performance_plots(silhouette_scores, calinski_harabasz_scores, no_clusters)

    #dendrogram_plot(data_to_cluster, no_cluster)
    plt.title('Hierarchical Dendrogram (truncated at {} clusters)'.format(no_clusters))
    plt.xlabel('distance')
    plt.ylabel('sample index or (cluster size)')
    linkage_array = ward(data_to_cluster)
    dendrogram(linkage_array, p=no_clusters, truncate_mode="lastp", orientation='left')
    plt.show()

    return silhouette_scores, calinski_harabasz_scores, cluster_assignments_list

def metrics_table(calinski_harabasz_scores, silhouette_scores):
    display_df = pd.DataFrame(data=[calinski_harabasz_scores, silhouette_scores], columns=list(range(2,(len(silhouette_scores)+2))), index=['Calinski-Harabasz', 'Silhouette']).T
    display_df['Silhouette Difference'] = display_df['Silhouette'].diff()
    display_df['Silhouette Difference'] = display_df['Silhouette Difference'].apply(lambda x : "❇️ Increasing" if x > 0 else "😔 Decreasing")
    return display_df

def assign_clusters(voters_data, cluster_assignments):
    clusters = []
    voters_data['Cluster Assignment'] = cluster_assignments
    for cluster_assignment in set(cluster_assignments):
        cluster = voters_data[voters_data['Cluster Assignment']==cluster_assignment]
        cluster = cluster.drop('Cluster Assignment', axis=1)
        clusters.append(cluster)
    voters_data = voters_data.drop('Cluster Assignment', axis=1, inplace=True)
    return clusters

def create_full_data_clusters(reconstituted_voters_data, clusters):
    full_data_clusters = []
    for cluster in clusters:
        full_data_cluster = pd.merge(reconstituted_voters_data, cluster, how='inner', left_index=True, right_index=True)
        full_data_clusters.append(full_data_cluster)
    print("🗃️🔺✨ Cluster assignments merged with original dataset, expanded by weight.")
    return full_data_clusters

def get_cluster_assignments(voters_data, no_of_clusters, dendrogram_generate = True):
    aggclus = AgglomerativeClustering(linkage = "ward", n_clusters=no_of_clusters)
    aggclus.fit(voters_data)
    cluster_assignments = aggclus.fit_predict(voters_data)
    print("✨ Cluster Assignments obtained.")
    if dendrogram_generate == True:
        print("Now drawing dendrogram plot...")
        dendrogram_plot(voters_data, no_of_clusters)
    return cluster_assignments

def display_cluster_sizes(cluster_sizes):
    cluster_sizes_array = np.array(list(cluster_sizes.values())) * 100
    cluster_sizes_round = cluster_sizes_array.round(decimals=2)
    return pd.DataFrame(data = cluster_sizes_round, index = cluster_sizes.keys(), columns=["Size (% of total)"])

def cluster_sizes_dict(clusters, voters_data):
    cluster_sizes = {}
    cluster_no = 0
    for cluster in clusters:
        cluster_no += 1
        cluster_sizes.update({cluster_no : (len(cluster) / len(voters_data))})
    print("📐 Cluster sizes (in terms of percentage of total) calculated.")
    return cluster_sizes

def issue_support_across_clusters(issue, clusters, voters_data, cluster_sizes):
    positive_clusters = []
    positive_size = 0
    negative_clusters = []
    negative_size = 0

    for cluster in cluster_sizes.keys():
        issue_diff = (clusters[cluster - 1].mean() - voters_data.mean())[issue]
        if issue_diff < 0:
            negative_clusters.append(cluster)
            negative_size -= cluster_sizes[cluster]
        if issue_diff > 0:
            positive_clusters.append(cluster)
            positive_size += cluster_sizes[cluster]
    print(issue)
    print("+{}".format(round(positive_size, 2)), positive_clusters)
    print(round(negative_size, 2), negative_clusters)
    print(" ")
    return #perhaps a dataframe?

def generate_cluster_coalitions(cluster_sizes_dict, minimum=2, maximum=3, threshold=.5):
    cluster_coalitions = []
    for combo_size in range(minimum, maximum+1):
        cluster_combos = list(combinations(cluster_sizes_dict, combo_size))
        for cluster_combo in cluster_combos:
            size = 0
            for cluster in cluster_combo:
                size += cluster_sizes_dict[cluster]
            if size > threshold:
                cluster_coalitions.append(cluster_combo)
    return cluster_coalitions

def get_pref_analysis_code(cluster_coalition, sign="-"):
    number = 1
    begin_code = "preferences[0][{}preferences[0]".format(sign)
    end_code = "]"
    intermediary_code = ""
    for pref_no in range(1,len(cluster_coalition)):
        intermediary_code += " & {}preferences[{}]".format(sign, pref_no)
    return begin_code + intermediary_code + end_code

def find_consensus_coalitions(cluster_coalitions, clusters, voters_data, agreements = 1, show_negatives=True):
    for cluster_coalition in cluster_coalitions:
        preferences = []
        for cluster in cluster_coalition:
            preference_profile = (clusters[cluster-1].mean() - voters_data.mean()) > 0
            preferences.append(preference_profile)
        positive_code = get_pref_analysis_code(cluster_coalition, sign="+")
        positive_consensus = eval(positive_code)
        negative_code = get_pref_analysis_code(cluster_coalition, sign="-")
        negative_consensus = eval(negative_code)
        if (len(positive_consensus) + len(negative_consensus)) >= agreements:
            print(cluster_coalition)
            print("Positive Consensus: {}".format(list(positive_consensus.keys())))
            print("Negative Consensus: {}".format(list(negative_consensus.keys())))
            print(" ")
        elif ((len(positive_consensus) + len(negative_consensus)) == 0) and show_negatives == True:
            print(cluster_coalition)
            print("No consensus.")
            print(" ")
    return

def issue_support_across_clusters_dict(issue, clusters, voters_data, cluster_sizes):
    positive_clusters = []
    positive_size = 0
    negative_clusters = []
    negative_size = 0

    for cluster in cluster_sizes.keys():
        issue_diff = (clusters[int(cluster) - 1].mean() - voters_data.mean())[issue]
        if issue_diff < 0:
            negative_clusters.append(cluster)
            negative_size -= cluster_sizes[cluster]
        if issue_diff > 0:
            positive_clusters.append(cluster)
            positive_size += cluster_sizes[cluster]
    #print("{}\n+{} {}\n{} {}".format(issue, round(positive_size, 2), positive_clusters, round(negative_size, 2), negative_clusters))
    issue_support = {"issue" : issue,
                     "positive_size": positive_size,
                     "positive_clusters": positive_clusters,
                     "negative_size": negative_size,
                     "negative_clusters": negative_clusters}
    return issue_support

def issues_by_clustered_advantage(clusters, voters_data, cluster_sizes):
    issue_support_across_clusters_list = []
    for issue in list(voters_data.columns):
        issue_support_stats = issue_support_across_clusters_dict(issue, clusters, voters_data, cluster_sizes)
        issue_support_across_clusters_list.append(issue_support_stats)
    issue_support_across_clusters_sorted = sorted(issue_support_across_clusters_list, key = lambda i : -i['positive_size'])
    for information in range(len(issue_support_across_clusters_sorted)):
        print("{}\n+{} {}\n{} {}\n".format(
                    issue_support_across_clusters_sorted[information]['issue'],
                    round(issue_support_across_clusters_sorted[information]['positive_size'], 2),
                    issue_support_across_clusters_sorted[information]['positive_clusters'],
                    round(issue_support_across_clusters_sorted[information]['negative_size'], 2),
                    issue_support_across_clusters_sorted[information]['negative_clusters']))
    return #issue_support_across_clusters_list
