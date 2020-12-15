import community as cm
import networkx as nx
import numpy as np
import scipy.sparse as sp


# TO DO: comment

def louvain_clustering(adj):

    # Perform Louvain algorithm from adj matrix
    partition = cm.best_partition(nx.from_scipy_sparse_matrix(adj))
    # From dict to lists
    clusters_louvain = list(partition.values())
    # Number of clusters found by Louvain
    nb_clusters_louvain = np.max(clusters_louvain) + 1
    # One-Hot representation
    clusters_louvain_onehot = sp.csr_matrix(np.eye(nb_clusters_louvain)[clusters_louvain])
    # Binary community matrix (adj_louvain[i,j] = 1 if nodes i and j are in the same community)
    adj_louvain = clusters_louvain_onehot.dot(clusters_louvain_onehot.transpose())

    # Remove diagonal and divide by row sum for normalizing
    adj_ = adj_louvain - sp.eye(adj_louvain.shape[0])
    adj_louvain_norm = sp.diags(np.power(np.array(adj_.sum(1)), -1).flatten()).dot(adj_)
    return adj_louvain, adj_louvain_norm, nb_clusters_louvain