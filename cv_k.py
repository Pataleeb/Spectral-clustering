from collections import Counter
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
import numpy as np

class PoliticalBlogsClustering:
    def __init__(self):
        self.edges = None
        self.node_labels = {}
        self.num_nodes = 1490

    def data_edges(self, file_path):
        self.edges = np.loadtxt(file_path, dtype=int)
        self.remove_isolated_nodes()
        return self.edges

    def data_nodes(self, file_path):
        node_labels = {}
        with open(file_path, "r") as file:
            for line in file:
                lines = line.strip().split('\t')
                node_id = int(lines[0])
                label = int(lines[2])
                node_labels[node_id] = label
        self.node_labels = node_labels
        return node_labels

    def remove_isolated_nodes(self):
        connected_nodes = set(self.edges.flatten())
        self.node_labels = {node: label for node, label in self.node_labels.items() if node in connected_nodes}
        self.num_nodes = len(self.node_labels)

    def adjacency_matrix(self):
        max_node_index = max(self.edges.max(), max(self.node_labels.keys()))
        self.num_nodes = max_node_index

        i = self.edges[:, 0] - 1
        j = self.edges[:, 1] - 1
        v = np.ones((self.edges.shape[0],))

        A = sparse.coo_matrix((v, (i, j)), shape=(self.num_nodes, self.num_nodes))
        A = (A + A.transpose())
        A = A.toarray()

        degree_sum = np.sum(A, axis=1)
        degree_sum = np.where(degree_sum == 0, 1e-10, degree_sum)

        D = np.diag(1 / np.sqrt(degree_sum).flatten())
        L = D @ A @ D
        return np.asarray(L)

    def eigenvectors(self, L, num_clusters):
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        large_k_eigenvectors = eigenvectors[:, -num_clusters:]

        row_n = np.linalg.norm(large_k_eigenvectors, axis=1, keepdims=True)
        row_n = np.where(row_n == 0, 1e-10, row_n)
        eigenvectors_n = large_k_eigenvectors / row_n

        return eigenvalues, eigenvectors_n

    def kmeans(self, eigenvectors_n, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
        cluster_lab = kmeans.fit_predict(np.asarray(eigenvectors_n))
        return cluster_lab

    def find_majority_labels(self, num_clusters=2, validation=None):
        result_map = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }

        if validation is not None:
            train_nodes = [node for node in self.node_labels if node not in validation]
        else:
            train_nodes = list(self.node_labels.keys())

        train_edges = np.array([edge for edge in self.edges if edge[0] in train_nodes and edge[1] in train_nodes])


        node_to_index = {node: idx for idx, node in enumerate(sorted(train_nodes))}
        i = [node_to_index[edge[0]] for edge in train_edges]
        j = [node_to_index[edge[1]] for edge in train_edges]
        v = np.ones(len(train_edges))

        num_train_nodes = len(train_nodes)
        A = sparse.coo_matrix((v, (i, j)), shape=(num_train_nodes, num_train_nodes))
        A = A + A.transpose()
        A = A.toarray()

        degree_sum = np.sum(A, axis=1)
        degree_sum = np.where(degree_sum == 0, 1e-10, degree_sum)
        D = np.diag(1 / np.sqrt(degree_sum).flatten())
        L = D @ A @ D

        _, eigenvectors_n = self.eigenvectors(L, num_clusters)
        cluster_lab = self.kmeans(eigenvectors_n, num_clusters)

        sorted_train_nodes = sorted(train_nodes)
        cluster_m_labels = {i: [] for i in range(num_clusters)}
        total_mismatches = 0
        total_nodes = 0

        for node_id, cluster_id in enumerate(cluster_lab):
            node_key = sorted_train_nodes[node_id]
            true_label = self.node_labels.get(node_key, -1)

            if true_label != -1:
                cluster_m_labels[cluster_id].append(true_label)
                total_nodes += 1

        for cluster_id, labels in cluster_m_labels.items():
            if labels:
                majority_label = Counter(labels).most_common(1)[0][0]
                tot_mismatch = sum(1 for label in labels if label != majority_label)
                mismatch_rate = round(tot_mismatch / len(labels), 2)
                total_mismatches += tot_mismatch

                result_map["mismatch_rates"].append({
                    "majority_index": majority_label,
                    "mismatch_rate": mismatch_rate
                })
            else:
                result_map["mismatch_rates"].append({
                    "majority_index": None,
                    "mismatch_rate": 0.0
                })

        mismatch_rate_net = round(total_mismatches / total_nodes, 2)
        result_map["overall_mismatch_rate"] = mismatch_rate_net
        return result_map

def cv_k_with_test(k_num, kfolds=5, test_size=0.15):
    pbclustering = PoliticalBlogsClustering()
    pbclustering.data_edges('edges.txt')
    pbclustering.data_nodes('nodes.txt')

    totnodes = list(pbclustering.node_labels.keys())


    train_val_nodes, test_nodes = train_test_split(totnodes, test_size=test_size, shuffle=True)


    kf = KFold(n_splits=kfolds, shuffle=True)

    avg_mmr = []


    for k in k_num:
        fold_mmr = []

        for trainid, validid in kf.split(train_val_nodes):
            valnodes = [train_val_nodes[i] for i in validid]
            result = pbclustering.find_majority_labels(num_clusters=k, validation=valnodes)
            fold_mmr.append(result["overall_mismatch_rate"])

        avg_mismatch = np.mean(fold_mmr)
        avg_mmr.append(avg_mismatch)


    best_k_index = np.argmin(avg_mmr)
    best_k = k_num[best_k_index]
    lowest_mismatch_rate = avg_mmr[best_k_index]

    print(f"\nOptimal k from cross-validation: {best_k}")
    print(f"Lowest Average Mismatch Rate for Optimal k ({best_k}): {lowest_mismatch_rate:.4f}")


    final_result = pbclustering.find_majority_labels(num_clusters=best_k, validation=test_nodes)
    print(f"\nFinal Evaluation on Test Set for k = {best_k}:")
    print(f"Overall Mismatch Rate on Test Set: {final_result['overall_mismatch_rate']:.4f}")


    plt.figure(figsize=(10, 6))
    plt.plot(k_num, avg_mmr, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Mismatch Rate')
    plt.title('Mismatch Rate vs. Number of Clusters')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    k_num = range(2, 50)
    cv_k_with_test(k_num)
