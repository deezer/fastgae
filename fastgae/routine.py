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

for features in (False, True):
    print('STARTING :', features, " ------------------------------------------------------------------------------------")
    for gamma3 in (0.,0.001,0.01,0.1,0.5,1.0,2.0):
        print("------------------------------------------------------------------------------------")
        for gamma2 in (0.,0.001,0.01):
            for gamma in (0.,0.001,0.01,0.1,0.5,1.0):
                os.system('python train.py --task=node_clustering --fastgae=True --nb_node_samples=1500 --model=linear_vae --nb_run=8 \
                --verbose=False --features={} --gamma3={} --gamma2={} --gamma={}'.format(features,gamma3,gamma2,gamma))