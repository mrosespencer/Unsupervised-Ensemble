# Consensus Clustering Demo
# Copyright AQM 2017
# You are free to use, modify, distribute this code AS LONG AS YOU RETAIN THIS HEADER


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
import glob
import pickle
from hungarian import *
from collections import Counter


### constants and initializtion (YOU CAN PLAY AROUND WITH THESE)
np.random.seed(666)
MAX_K = 8 # MAX K used for optimal K sweep
BOOTSTRAP_SIZE = int(159*1) # size of each bootstrap sample
BOOTSTRAP_N = 20 # number of bootstrap samples (YOU CAN PLAY AROUND WITH THIS)
DATA_START_INDEX = 1 # account for df's named index column 0 (DON'T CHANGE THIS UNLESS YOUR DATASET NEEDS IT)
DO_K_SWEEP = True # switch to do sweep of K values using K means to find optimal K
OPTIMAL_K = 3 # we know the Iris dataset has 3 clusters (ground truth), change this for different datasets

# IF using an external CSV
# read each csv in local project folder matching the wildcard
all_dfs = []
path = "*.csv"
for filename in glob.glob(path):
    with open(filename, 'r') as f:
            print("opened csv " + str(f))
            df = pd.read_csv(filename, encoding = 'utf-8')
            all_dfs.append([df, filename])


#df = all_dfs[0][0]

# read Iris dataset
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'] )

# generate names for each row in the Iris dataset
def generate_names(s, size):
    labels = []

    for i in range(size):
        labels.append(s + str(i))

    return labels

# join the names column to the dataframe
labels = generate_names('flower', len(df))
labels = pd.DataFrame(labels)
df.insert(0, 'name', labels)

###

### get bootstrap samples in df format
### size = size of each bootstrap sample, N = number of bootstrap samples
def get_bootstraps(df, size, N):
    bootstrapped_dfs = []
    for i in range(N):
        # randomly select rows from original DF until size is met
        temp = pd.DataFrame()
        for j in range(size):
            # uniformly sample with replacement from dataset (there will be duplicate rows)
            rand_index = np.random.randint(0, len(df))
            # append row to new bootstrap sample dataframe
            temp = temp.append(df.iloc[rand_index, :])[df.columns.tolist()]

        # this list is a collection of the bootstrap sample dfs
        bootstrapped_dfs.append(temp)

    return bootstrapped_dfs

### find optimal K using Pseudo-F for bootstrapped KMeans
### Pseudo-F = (sum(within cluster variance) / (K+1)) / (sum(betwen cluster variance) / 2*(N - K + 1))
### This function also plots an 'Elbow plot' for finding the elbow point for the optimal K
### This function is also 'embarrassingly parallelizable' and quite slow if not parallelized:
### The within cluster variance calculation is the slowest step.
### The exercise is left to the reader to implement a parallel version of this function.
def find_optimal_k(means, bts, max_K=MAX_K, N = BOOTSTRAP_N):
    optimal_k = None
    wit_cluster_var = []
    btw_cluster_var = []

    # find the within cluster variance for each bootstrap set for each K
    for i in range(max_K - 1):
        bts_val = []
        # iterate each bootstrap
        for j in range(N):
            temp = []
            # get cluster centroid
            cents = means[i][j].cluster_centers_
            # calculate each sample's distance from its assigned centroid
            labels = means[i][j].predict(bts[j].values[:,DATA_START_INDEX:])
            # iterate each item in bootstrap sample
            for l in range(len(bts[j])):
                temp.append(np.linalg.norm(bts[j].values[l,DATA_START_INDEX:] - cents[labels[l]]))

            # average the sum of within cluster variance over (N-K)
            val = sum(temp) / (BOOTSTRAP_N - (i+2))
            print("K = " + str(i + 2) + " within cluster var = " + str(val))
            bts_val.append(val)

        wit_cluster_var.append(bts_val)

    # compute the between cluster variances (between the Kmeans centroids)
    for i in range(max_K - 1):
        bts_val = []
        # iterate through each K
        for j in range(N):
            temp = []
            # get cluster centroids
            cents = means[i][j].cluster_centers_
            cents_mean = cents[0]

            # find population vector centroid
            for k in range(len(cents) - 1):
                cents_mean = cents_mean + cents[k+1]
            cents_mean = cents_mean / float(len(cents))

            # calculate btw cluster centroid variance
            for k in range(len(cents)):
                temp.append(np.linalg.norm(cents[k] - cents_mean))

            # average the sum of btw cluster variance over (K - 1)
            val = sum(temp) / ((i + 2) - 1)
            print("K = " + str(i + 2) + " btw cluster var = " + str(val))
            bts_val.append(val)

        btw_cluster_var.append(bts_val)

    pf =[]

    # calcluate F-stat ratio across each bootstrap
    for i in range(len(btw_cluster_var)):
        temp = []
        for j in range(len(btw_cluster_var[i])):
            temp.append(btw_cluster_var[i][j]/wit_cluster_var[i][j])

        pf.append(temp)



    # debug output and save pf to file
    print("Pseudo F:")
    print(pf)
    pickle.dump(pf, open("pf.p", "wb"))

    # pf boxplot
    plt.figure()
    plt.boxplot(pf)

    # scatter plot of points to fit within the boxes
    for n in range(max_K-1):
        for o in range(len(pf[n])):
            y = pf[n][o]
            x = n + 1
            plt.plot(x, y, 'r.', alpha=0.2)

    ticks = list(np.linspace(2, max_K, max_K - 1))
    plt.xticks(list(np.linspace(1, max_K-1, max_K - 1)), [str(int(s)) for s in ticks])
    plt.xlabel('Number of clusters')
    plt.ylabel('Pseudo-F statistic')
    plt.show()

    return None


### pairwise Welch's t-test on each bootstrap for each K for each clustering method
def bts_t_test():

    # TODO: implement your pairwise Welch's t-test here

    return None

#####################################################
#####################################################
#####################################################

bts = get_bootstraps(df, BOOTSTRAP_SIZE, BOOTSTRAP_N)

### run the clustering on each bootstrap for range of K's


# Kmeans clustering
K_means = []
# in the final implementation, K_means_bts contains the labels for all bootstraps for the optimal K
# for all methods
K_means_bts = []
print("running K means")

if(DO_K_SWEEP == True):
    ### Set K = 8 and do the sweep.
    MAX_K = 8
    start = 0

    for i in range(start, MAX_K-1):

        temp = []
        temp2 = []
        for j in range(BOOTSTRAP_N):
            temp.append(KMeans(n_clusters=i+2, init='k-means++', n_jobs = 1, n_init=10, max_iter=400))
            temp2.append(temp[j].fit_predict(bts[j].values[:,DATA_START_INDEX:]))
            print("K = " + str(i+2) + " N = " + str(j))

        K_means.append(temp)
        K_means_bts.append(temp2)


    print("running optimal K")
    find_optimal_k(K_means, bts)

    # exit after optimal K plots have been displayed
    quit()
else:
    ### Set K = 8 and do the sweep.
    MAX_K = OPTIMAL_K
    start = OPTIMAL_K-2

    for i in range(start, MAX_K - 1):

        temp = []
        temp2 = []
        for j in range(BOOTSTRAP_N):
            temp.append(KMeans(n_clusters=i + 2, init='k-means++', n_jobs=1, n_init=10, max_iter=400))
            temp2.append(temp[j].fit_predict(bts[j].values[:, DATA_START_INDEX:]))
            print("K = " + str(i + 2) + " N = " + str(j))

        K_means.append(temp)
        K_means_bts.append(temp2)

### run rest of the clustering algorithms
MAX_K = OPTIMAL_K
start = MAX_K-2

#start = MAX_K
print("running GMM")
# GMM clustering
GMM = []
GMM_bts = []

for i in range(start, MAX_K-1):
    temp = []
    temp2 = []
    for j in range(BOOTSTRAP_N):
        temp.append(GaussianMixture(n_components=i + 2))
        temp[j].fit(bts[j].values[:, DATA_START_INDEX:])
        temp2.append(temp[j].predict(bts[j].values[:, DATA_START_INDEX:]))
        print("K = " + str(i + 2) + " N = " + str(j))

    GMM.append(temp)
    GMM_bts.append(temp2)
    K_means_bts.append(temp2)

    pickle.dump(GMM, open("GMM.p", "wb"))

print("running Ward Agglomerative")
# Ward Agglomerative
wards = []
wards_bts = []

for i in range(start, MAX_K-1):
    temp = []
    temp2 = []
    for j in range(BOOTSTRAP_N):
        temp.append(AgglomerativeClustering(n_clusters=i + 2))
        temp2.append(temp[j].fit_predict(bts[j].values[:, DATA_START_INDEX:]))
        print("K = " + str(i + 2) + " N = " + str(j))

    wards.append(temp)
    wards_bts.append(temp2)
    K_means_bts.append(temp2)

    pickle.dump(wards, open("wards.p", "wb"))

pickle.dump(K_means_bts, open("kmeans_bts.p", "wb"))
#K_means_bts = pickle.load(open("kmeans_bts.p", "rb"))
### do label correspondence between the different bootstraps with the Hungarian algorithm
### set reference label to first Kmeans bootstrap label set
ref_label = K_means_bts[0][0]

final_labels = []
final_labels_cm = []

for i in range(len(K_means_bts)):
    for j in range(len(K_means_bts[i])):
        update_labels = Hungarian(K_means_bts[0][j], ref_label).hungarian()
        print(update_labels['matched_labels'])
        final_labels.append(update_labels['matched_labels'])
        print(update_labels['new_cm'])
        final_labels_cm.append(update_labels['new_cm'])


### Do the actual consensus clustering here:
### Get the matrix of the proportion that each sample in each bootstrap appears
### in Each cluster

labels_with_indices = []

# append the named index to each final label
for i in range(len(K_means_bts[0])):
    indices = bts[i].values[:,0]
    temp = []
    for j in range(len(K_means_bts)):
        temp.append(pd.DataFrame(indices).join(pd.DataFrame(K_means_bts[j][i]), lsuffix='name', rsuffix='class', how = 'inner'))

    labels_with_indices.append(temp)

# iterate through the dfs and find the class labels for each row in the original df

original_df_labels = []
str_list = df.values[:,0]
for i in range(len(df)):
    # iterate across methods (3)
    original_df_labels.append([str_list[i]])
    for j in range(len(K_means_bts)):
        temp2 = []
        # iterate across the bootstraps
        for k in range(len(K_means_bts[0])):
            foo = labels_with_indices[k][j][labels_with_indices[k][j]['0name'].str.contains(str_list[i])].values[:,DATA_START_INDEX]
            if foo.size != 0:
                temp2.append(foo[0])

        original_df_labels[i].append(temp2)

# count the class labels for each row in original df
original_df_labels_freqs = []
for i in range(len(df)):
    original_df_labels_freqs.append([str_list[i]])
    for j in range(len(K_means_bts)):
        # iterate across methods (3)
        temp = Counter(original_df_labels[i][j+1]).keys()
        temp2 = Counter(original_df_labels[i][j+1]).values()
        original_df_labels_freqs[i].append(dict(zip(temp, temp2)))

# calculate proportions here
mats = []

for j in range(len(K_means_bts)):
    temp_mat = []
    for i in range(len(df)):
        total_occurences = float(sum(original_df_labels_freqs[i][j+1].values()))
        temp = []
        for k in range(OPTIMAL_K):
            if k in original_df_labels_freqs[i][j+1]:
                temp.append(original_df_labels_freqs[i][j+1][k] / total_occurences)
            else:
                temp.append(0.0)
        temp_mat.append(temp)
    mats.append(temp_mat)

## heatmap/Matrix plots for each algorithm
titles = ['KMeans', 'GMM Clustering', 'Ward Agglomerative']

for i in range(len(mats)):
    plt.figure()
    plt.title(titles[i])
    plt.xlabel('First 20 samples in numeric order')
    plt.ylabel('Class label')
    plt.yticks([0,1,2])
    plt.imshow(np.asarray(mats[i][1:20]).transpose(), interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.axes().set_aspect('auto')
plt.show()

# finally, get the consensus from each algorithm
consensus = []
for i in range(len(df)):
    temp = []
    for j in range(OPTIMAL_K):
        temp2 = []
        for k in range(len(mats)):
            temp2.append(mats[k][i][j])

        temp2 = sum(temp2)/float(OPTIMAL_K)
        temp.append(temp2)

    consensus.append(temp)
# save final consensus to CSV
consensus = pd.DataFrame(consensus)
consensus.insert(0, 'name', labels)
consensus.to_csv('iris_consensus.csv')




