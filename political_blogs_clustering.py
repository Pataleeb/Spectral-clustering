
from collections import Counter
from numpy import dtype
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
##Reference: ISYE6740 demo code
class PoliticalBlogsClustering:
    def __init__(self):
        self.edges=None
        self.node_labels={}
        self.num_nodes=1490
        pass

    def data_edges(self,file_path):
        self.edges=np.loadtxt(file_path,dtype=int)
        self.remove_isolated_nodes()
        return self.edges
        #return np.loadtxt(file_path, dtype=int)



    def data_nodes(self,file_path):
        node_labels={}

        with open(file_path,"r") as file:
            for line in file:
                lines=line.strip().split('\t')
                node_id=int(lines[0])
                label=int(lines[2])
                node_labels[node_id] = label
        self.node_labels=node_labels
        return node_labels

    def remove_isolated_nodes(self):
        connected_nodes=set(self.edges.flatten())
        self.node_labels={node:label for node,label in self.node_labels.items() if node in connected_nodes}
        self.num_nodes=len(self.node_labels)

    def adjacency_matrix(self):

        max_node_index = max(self.edges.max(), max(self.node_labels.keys()))
        self.num_nodes = max_node_index


        i = self.edges[:, 0] - 1
        j = self.edges[:, 1] - 1
        v = np.ones((self.edges.shape[0],))


        A = sparse.coo_matrix((v, (i, j)), shape=(self.num_nodes, self.num_nodes))
        A = (A + A.transpose())
        A = A.todense()


        degree_sum = np.sum(A, axis=1)
        degree_sum = np.where(degree_sum == 0, 1e-10, degree_sum)

        D = np.diag(1 / np.sqrt(degree_sum).flatten())
        L = D @ A @ D
        return np.array(L)

    def eigenvectors(self, L, num_clusters):

        eigenvalues, eigenvectors = np.linalg.eigh(L)
        large_k_eigenvectors = eigenvectors[:, -num_clusters:]

        row_n = np.linalg.norm(large_k_eigenvectors, axis=1, keepdims=True)
        row_n = np.where(row_n == 0, 1e-10, row_n)
        eigenvectors_n = large_k_eigenvectors / row_n

        return eigenvalues, eigenvectors_n

    def kmeans(self,eigenvectors_n,num_clusters):
        kmeans=KMeans(n_clusters=num_clusters,n_init='auto')
        custer_lab=kmeans.fit_predict(eigenvectors_n)
        return custer_lab

    def find_majority_labels(self, num_clusters=2):
        map ={
            "overall_mismatch_rate":None,
            "mismatch_rates":[]
        }
        self.edges=self.data_edges('edges.txt')
        self.node_labels=self.data_nodes('nodes.txt')
        self.remove_isolated_nodes()

        L=self.adjacency_matrix()
        _, eigenvectors_n=self.eigenvectors(L,num_clusters)
        cluster_lab=self.kmeans(eigenvectors_n,num_clusters)

        cluster_m_labels={i: [] for i in range(num_clusters)}
        total_mismatches=0
        total_nodes=0

        for node_id, cluster_id in enumerate(cluster_lab):
            true_label = self.node_labels.get(node_id + 1, -1)

            if true_label!=-1:
                cluster_m_labels[cluster_id].append(true_label)
                total_nodes += 1

        for cluster_id, labels in cluster_m_labels.items():
            if labels:
                majority_label=Counter(labels).most_common(1)[0][0]
                tot_mismatch=sum(1 for label in labels if label != majority_label)
                mismatch_rate=round(tot_mismatch/len(labels),2)
                total_mismatches += tot_mismatch

                map["mismatch_rates"].append({
                    "majority_index": majority_label,
                    "mismatch_rate": mismatch_rate


                })
            else:
                map["mismatch_rates"].append ({
                    "majority_index":None,
                "mismatch_rate":0.0
                })
        mismatch_rate_net=round(total_mismatches/total_nodes,2)
        map["overall_mismatch_rate"]=mismatch_rate_net
        return map

if __name__ == "__main__":
    clustering = PoliticalBlogsClustering()

    for k in [2, 5, 10, 30, 50]:
        result = clustering.find_majority_labels(num_clusters=k)
        print(f"\nResults for k = {k}:")
        print(f"Overall Mismatch Rate: {result['overall_mismatch_rate']:.2f}")
        for i, cluster_info in enumerate(result["mismatch_rates"]):
            majority_index = cluster_info["majority_index"]
            mismatch_rate = cluster_info["mismatch_rate"]
            print(f"  Cluster {i}: Majority Label = {majority_index}, Mismatch Rate = {mismatch_rate:.2f}")

