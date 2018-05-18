from HungarianAlgo import Hungarian
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np

from Bootstrap import *
from HungarianAlgo import *
from Prepare import *
from OptimalClusters import *
from RunningAlgos import *
from Consensus import *


# initialize consts
np.random.seed(666)
MAX_K = 8                     # MAX K used for optimal K sweep
BOOTSTRAP_SIZE = int(159*1)   # size of each bootstrap sample
BOOTSTRAP_N = 20              # number of bootstrap samples (YOU CAN PLAY AROUND WITH THIS)
DATA_START_INDEX = 1          # account for df's named index column 0 (DON'T CHANGE THIS UNLESS YOUR DATASET NEEDS IT)
DO_K_SWEEP = True             # switch to do sweep of K values using K means to find optimal K
OPTIMAL_K = 3                 # Iris dataset has 3 clusters (ground truth), change this for different datasets


# import data
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris['data']], columns= iris['feature_names'] )

# prepare data (add index column 'flower')
prep = Prepare('flower', len(df)).names_join(df)
df = prep['df']
labels = prep['labels']

# generate bootstrap samples
bts = Bootstrap(df, BOOTSTRAP_SIZE, BOOTSTRAP_N).get_bootstraps()

# determine optimal clustering K
kmeans = Bootstrap.kmeans_bootstrap(bts, DO_K_SWEEP, BOOTSTRAP_N, DATA_START_INDEX, OPTIMAL_K, MAX_K)

# max k determined above becomes optimal k
gmm = RunAlgos(3, BOOTSTRAP_N, DATA_START_INDEX, bts, kmeans).run_GMM()
agglomerative = RunAlgos(3, BOOTSTRAP_N, DATA_START_INDEX, bts, kmeans).run_Agglomerative()
kmeans_ = RunAlgos(3, BOOTSTRAP_N, DATA_START_INDEX, bts, kmeans).run_KMeans()
# consensus clustering
cc_init = Consensus(kmeans_, gmm, agglomerative, bts, df, DATA_START_INDEX, 3, labels)
mats = cc_init.combine_results()

# write consensus results to file
cc = cc_init.consensus(mats)

# heatmaps
cc_init.heatmaps(mats)


