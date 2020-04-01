from fastgae.preprocessing import sparse_to_tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import warnings as wn

flags = tf.app.flags
FLAGS = flags.FLAGS

wn.simplefilter('ignore', UserWarning)

def get_distribution(measure, alpha, adj):
    """ Compute the p_i probabilities to pick each node i through the
    node sampling scheme of FastGAE (see subsection 3.2.3. of paper)
    :param measure: node importance measure, among 'degree', 'core', 'uniform'
    :param alpha: alpha scalar hyperparameter for degree and core sampling
    :param adj: sparse adjacency matrix of the graph
    :return: list of p_i probabilities of all nodes
    """
    if measure == 'degree':
        # Degree-based distribution
        proba = np.power(np.sum(adj, axis=0),alpha).tolist()[0]
    elif measure == 'core':
        # Core-based distribution
        G = nx.from_scipy_sparse_matrix(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        proba = np.power(list(nx.core_number(G).values()), alpha)
    elif measure == 'uniform':
        # Uniform distribution
        proba = np.ones(adj.shape[0])
    else:
        raise ValueError('Undefined sampling method!')
    # Normalization
    proba = proba/np.sum(proba)
    return proba

def node_sampling(adj, distribution, nb_node_samples, replace=False):
    """ Sample a subgraph from a given node-level distribution
    :param adj: sparse adjacency matrix of the graph
    :param distribution: p_i distribution, from get_distribution()
    :param nb_node_samples: size (nb of nodes) of the sampled subgraph
    :param replace: whether to sample nodes with replacement
    :return: nodes from the sampled subgraph, and subgraph adjacency matrix
    """
    # Sample nb_node_samples nodes, from the pre-computed distribution
    sampled_nodes = np.random.choice(adj.shape[0], size = nb_node_samples,
                                     replace = replace, p = distribution)
    # Sparse adjacency matrix of sampled subgraph
    sampled_adj = adj[sampled_nodes,:][:,sampled_nodes]
    # In tuple format (useful for optimizers)
    sampled_adj_tuple = sparse_to_tuple(sampled_adj + sp.eye(sampled_adj.shape[0]))
    return sampled_nodes, sampled_adj_tuple, sampled_adj