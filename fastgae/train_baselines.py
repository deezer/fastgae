from __future__ import division
from __future__ import print_function

import time
import os
import pickle

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import community as cm

from fastgae.evaluation import get_roc_score
from fastgae.input_data import load_data, load_label
from fastgae.preprocessing import *

from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_mutual_info_score
from sklearn.manifold import spectral_embedding




####### Settings ######


tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS


# Dataset parameter
flags.DEFINE_string('dataset', 'cora', 'Name of the Dataset')
''' Available datasets:

 - cora: Cora dataset used by Kipf and Welling (2016)

 - citeseer: Citeseer dataset used by Kipf and Welling (2016)

 - pubmed: Pubmed dataset used by Kipf and Welling (2016)

 - google: Google web graph from Stanford's SNAP website

 - patent6 and patent36: US Patent citation network 1975-1999 from Stanford's SNAP website
   (patent6: version with 6 communities, patent36: 36 refined communities - but the graph is the same)

 - sbm: 10-million nodes graph with 10 1-million nodes communities, generated from a Stochastic Block Model
 '''


# Model parameters
flags.DEFINE_string('model', 'spectral', 'Name of the Model')
''' Available Models:
 - spectral: eigendecomposition of Laplacian matrix, and embedding in first eigenvectors space
            (if task=node_clustering, it leads to spectral clustering algorithm)
 - louvain: Louvain algorithm for node clustering from Blondel et al.

 - deepwalk : DeepWalk node embedding model from Perozzi et al.

 - node2vec : node2vec model from Grover and Leskovec

 - line: LINE model (Large-scale Information Network Embedding) from Tang et al.
'''


# Training parameters
flags.DEFINE_integer('nb_run', 1, 'Number of time to run model + test it')

flags.DEFINE_float('prop_train', 0.85, '1/prop_train = Proportion of edges in train set - only if task = link_prediction')

flags.DEFINE_float('prop_test', 0.10, '1/prop_test = Proportion of edges in test set - only if task = link_prediction')

flags.DEFINE_boolean('validation', False, 'Whether to report validation results at each epoch (not implemented for kcore=True)')

flags.DEFINE_boolean('verbose', True, 'Whether to print training progress details.')


# Size of embedding
flags.DEFINE_integer('dimension', 16, "Dimension of embedding")


# Parameters that are specific to Deepwalk and node2vec models

flags.DEFINE_integer('window_size', 5, 'Window size of skipgram model')

flags.DEFINE_integer('number_walks', 10, 'Number of random walks to start at each node')

flags.DEFINE_integer('walk_length', 80, 'Length of the random walk started at each node')


# Parameters that are specific to node2vec model
flags.DEFINE_float('p', 1., 'node2vec p hyperparameter')

flags.DEFINE_float('q', 1., 'node2vec q hyperparameter')

# Parameters that are specific to LINE model
flags.DEFINE_string('proximity', 'second-order', 'first-order or second-order proximity')


# Parameters to load existing preprocessed dataset
# (useful to avoid repeating "edge masking" preprocessing step on large dataset, for link prediction task)
flags.DEFINE_boolean('using_pre_loaded', False, 'Whether to loaded an already masked adjacency matrix')

flags.DEFINE_boolean('save_masked_data', False, 'When computing masked adj matrix, whether to save results')


# Choose ML task to perform on Graph
flags.DEFINE_string('task', 'link_prediction', 'ML talk to perform: link_prediction or node_clustering')




model_str = FLAGS.model
dataset_str = FLAGS.dataset


# Lists to collect average results

if FLAGS.task == 'link_prediction':

    mean_roc = []
    mean_ap = []

else:

    mean_mutual_info = []


mean_time = []




###### Load Data ######


if FLAGS.verbose:
    print("Loading data...")


#Load adjacency matrix
adj_init, features_init = load_data(dataset_str)


# Load ground-truth labels for node clustering task
if FLAGS.task == 'node_clustering':

    labels = load_label(FLAGS.dataset)




# Repeat the entire training process nb_run times (to eventually compute standard errors on final results)
for i in range(FLAGS.nb_run):




    ###### Edge Masking ######


    # For the Link Prediction task, we randomly remove prop_test% of the edges from the graph
    if FLAGS.task == 'link_prediction' :

        if not FLAGS.using_pre_loaded:

            if FLAGS.verbose:

                print("Masking test edges...")

            adj, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_init)

            if FLAGS.save_masked_data:

                sp.save_npz('../data/preprocessed_data/{}_{}.npz'.format(dataset_str,i), adj)
                nx.write_adjlist(nx.from_scipy_sparse_matrix(adj), path='../data/preprocessed_data/{}_{}.adjlist'.format(dataset_str,i))

                np.savetxt('../data/preprocessed_data/test_edges_{}_{}.txt'.format(dataset_str,i), test_edges)
                np.savetxt('../data/preprocessed_data/test_edges_false_{}_{}.txt'.format(dataset_str,i), test_edges_false)

                if FLAGS.validation:

                    np.savetxt('../data/preprocessed_data/val_edges_{}_{}.txt'.format(dataset_str,i), val_edges)
                    np.savetxt('../data/preprocessed_data/val_edges_false_{}_{}.txt'.format(dataset_str,i), val_edges_false)


        else:

            adj = sp.load_npz('../data/preprocessed_data/{}_{}.npz'.format(dataset_str,i))

            test_edges = np.loadtxt("../data/preprocessed_data/test_edges_{}_{}.txt".format(dataset_str,i)).astype(int)
            test_edges_false = np.loadtxt("../data/preprocessed_data/test_edges_false_{}_{}.txt".format(dataset_str,i)).astype(int)

            if FLAGS.validation:
                val_edges = np.loadtxt("../data/preprocessed_data/val_edges_{}_{}.txt".format(FLAGS.model, dataset_str,i)).astype(int)
                test_edges_false = np.loadtxt("../data/preprocessed_data/val_edges_false_{}_{}.txt".format(dataset_str,i)).astype(int)

    else:

        adj = adj_init




    ###### Training ######


    t_start = time.time()




    if FLAGS.model == 'spectral':


        if FLAGS.task == 'node_clustering':

            emb = spectral_embedding(adj.asfptype(), n_components=FLAGS.dimension+1, norm_laplacian=True, drop_first=False)

        elif FLAGS.task == 'link_prediction':

            emb = spectral_embedding(adj, n_components=FLAGS.dimension, norm_laplacian=True)

        t_finish = time.time()




    elif FLAGS.model == 'louvain':

        if FLAGS.task == 'node_clustering':

            partition = cm.best_partition(nx.from_scipy_sparse_matrix(adj))
            print("partition")
            print(partition)

        else:

            print('Louvain algorithm only implemented for node_clustering task !')

        t_finish = time.time()




    elif FLAGS.model == 'deepwalk':

        # Train DeepWalk and save embeddings
        nx.write_adjlist(nx.from_scipy_sparse_matrix(adj), path='../data/preprocessed_data/{}_{}.adjlist'.format(dataset_str,i))
        os.system('cd ../baselines/deepwalk/ ; deepwalk --input ../../data/preprocessed_data/{}_{}.adjlist --output ../../data/output_data/{}_{}.txt \
        --window-size {} --number-walks {} --walk-length {} --representation-size {}'.format(dataset_str,i, dataset_str, i, \
        FLAGS.window_size, FLAGS.number_walks, FLAGS.walk_length, FLAGS.dimension))

        t_finish = time.time()

        # Retrieve embeddings
        emb = np.loadtxt("../data/output_data/{}_{}.txt".format(dataset_str,i), skiprows=1)
        emb = emb[emb[:,0].argsort()][:,1:]

        # Deal with a deepwalk bug
        if emb.shape[0] != adj.shape[0]:
            continue




    elif FLAGS.model == 'node2vec':

        # Train node2vec and save embeddings
        nx.write_adjlist(nx.from_scipy_sparse_matrix(adj), path='../data/preprocessed_data/{}_{}.adjlist'.format(dataset_str,i))
        os.system('cd ../baselines/node2vec/ ; python main.py --input ../../data/preprocessed_data/{}_{}.adjlist --output ../../data/output_data/{}_{}.txt \
        --window-size {} --num-walks {} --walk-length {} --p {} --q {} --dimensions {}'.format(dataset_str,i, dataset_str, i, \
        FLAGS.window_size, FLAGS.number_walks, FLAGS.walk_length, FLAGS.p, FLAGS.q, FLAGS.dimension))

        t_finish = time.time()

        # Retrieve embeddings
        emb = np.loadtxt("../data/output_data/{}_{}.txt".format(dataset_str,i), skiprows=1)
        emb = emb[emb[:,0].argsort()][:,1:]




    elif FLAGS.model == 'line':

        # Save data in pickle format
        nx.write_gpickle(nx.from_scipy_sparse_matrix(adj), path='../baselines/line/input_data/{}_{}.pkl'.format(dataset_str,i))

        t_start = time.time()

        # Train LINE and save embeddings
        os.system('cd ../baselines/line/ ; python line.py --graph_file input_data/{}_{}.pkl  --embedding_dim {} \
                  --proximity {}'.format(dataset_str, i, FLAGS.dimension, FLAGS.proximity))

        t_finish = time.time()

        # Retrieve embeddings
        with open('../baselines/line/output_data/embedding_second-order.pkl', 'rb') as pickle_file:
            loaded_emb = np.load(pickle_file)
            emb = np.vstack(loaded_emb.values())


    # Compute mean total running time
    mean_time.append(t_finish - t_start)




    ####### Test Model ######


    if FLAGS.verbose:

        print("Testing model...")


    # Link Prediction: classification edges/non-edges
    if FLAGS.task == 'link_prediction':

        # Get ROC and AP scores
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)

        # Compute mean scores over all runs
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)


    # Node Clustering in Latent Space (or in graph for Louvain)
    elif FLAGS.task == 'node_clustering':

        # Clustering
        if FLAGS.model == 'louvain':

            clustering_pred = list(partition.values())
            print("clustering_pred")
            print(clustering_pred)
            print(np.max(clustering_pred))

        else:

            clustering_pred = KMeans(n_clusters = len(np.unique(labels)), init = 'k-means++').fit(emb).labels_

        # Compute mean Normalized Mutual Information score over time
        mi_score = adjusted_mutual_info_score(labels, clustering_pred)
        mean_mutual_info.append(mi_score)




###### Report Final Results ######


print("Mean Total Time")
print(mean_time)
print(np.mean(mean_time))
print(np.std(mean_time))


if FLAGS.task == 'link_prediction':

    print("Mean AUC")
    print(mean_roc)
    print(np.mean(mean_roc))
    print(np.std(mean_roc))

    print("Mean AP")
    print(mean_ap)
    print(np.mean(mean_ap))
    print(np.std(mean_ap))


elif FLAGS.task == 'node_clustering':

    print("Mean Mutual Information Score")
    print(mean_mutual_info)
    print(np.mean(mean_mutual_info))
    print(np.std(mean_mutual_info))