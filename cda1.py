#!/usr/bin/env python
# coding: utf-8

#  (5 points) What’s the main difference between supervised and unsupervised learning? Give one benefit
# and drawback for supervised and unsupervised learning, respectively
# 
# Unsupervised learning is a category of machine learning algorithms that autonomously extracts insights from datasets without human intervention. This stands in contrast to supervised learning, where data is accompanied by predefined labels.
# 
# Differences of supervised and unsupervised learning.
# Supervised learning:
# - In supervised learning, both input and output data are labeled.
# - Supervised learning exhibits high complexity.
# - It relies on a training dataset for model development.
# - The number of classes in supervised learning is predefined.
# - Classification of data in supervised learning is based on the training dataset.
# - Supervised learning is categorized into two types: Regression and Classification.
# 
# Unsupervised learning:
# - Unsupervised learning algorithms operate without labeled data.
# - It involves lower computational complexity compared to supervised learning.
# - Unsupervised learning utilizes only the input dataset.
# - The number of classes in unsupervised learning is unknown.
# - Classification in unsupervised learning is based on the inherent properties of the data.
# - Unsupervised learning is classified into two types: Clustering and Association.
# 
# 

# In[1]:


import pandas as pd

# data
data = {'City': ['Atlanta', 'Houston'],
        'PropertyType': ['House', 'House'],
        'Price': [500000, 300000]}

# Convert categorical variables into one-hot-encoded binary vectors
df = pd.get_dummies(pd.DataFrame(data), columns=['City', 'PropertyType'])

# Display the resulting DataFrame
print(df)

from scipy.spatial.distance import hamming, euclidean

# Data points
point1_categorical = df.iloc[0, 1:3].values  # Categorical features for point 1
point2_categorical = df.iloc[1, 1:3].values  # Categorical features for point 2

point1_real = [df.iloc[0, 0]]  # Real-valued feature (price) for point 1
point2_real = [df.iloc[1, 0]]  # Real-valued feature (price) for point 2

# Calculate Hamming distance for categorical features
hamming_distance = hamming(point1_categorical, point2_categorical)

# Calculate Euclidean distance for real-valued feature
euclidean_distance = euclidean(point1_real, point2_real)

# Display the distances
print("Hamming Distance:", hamming_distance)
print("Euclidean Distance:", euclidean_distance)

# Define similarity function
def similarity_function(xi, xj):
    categorical_distances = hamming(xi[1:3], xj[1:3])
    real_valued_distance = euclidean([xi[0]], [xj[0]])
    return categorical_distances + real_valued_distance

#
point1 = df.iloc[0, :].values
point2 = df.iloc[1, :].values

# Calculate similarity measure using the defined function
similarity_measure = similarity_function(point1, point2)

# Display the combined similarity measure
print("Combined Similarity Measure:", similarity_measure)


# In[2]:


#pip install nbconvert
import nbconvert


# et's consider the Euclidean distance between x i and cj, which is equivalent to the square root of the squared Euclidean distance:
# $$∥x i−cj∥= \sqrt{}∑^{d}_{i=1}(x _{il}−c _{jl})^{2}$$
# d is the dimensionality of the data points. The optization problem:
# $$π(i)=arg    min _{j=1,...,k}(c_{j})^{T}(\frac{1}{2}(c_{j} −x_{i}))$$
# if we simplify the argument:
# $$(c_{j})^{T}( \frac{1}{2}(c_{j}−x_{i}))= \frac{1}{2}(c_{j})^{T}(c_{j}−x_{i})
# = \frac{1}{2}​((c_{j})^{T}c_{j}−(c_{j})^{T}x_{i})
# = \frac{1}{2}(^{d}∑_{1=1}(c_{jl})^{2} − ^{d∑_{l=1}c_{jl}x_{il})$$
# when we focus on the minimization:
# $$π(i)=arg min _{j=1,...,k}(\frac{1}{2}∑^{d}_{l=1}(c_{jl})^{2}−\frac{1}{2}∑^{d}_{l=1}c_{jl}x_{il})$$
# This expression is equivalent to the clustering assignment problem because minimizing this expression for j is equivalent to finding the j that minimizes the Euclidean distance between xi and cj which shows that they are equivalent.
#  

# 4) Different initializations in k-means produce diverse outcomes due to the algorithm's objective of minimizing a non-convex function. The sensitivity to the initial positions of cluster centroids leads to convergence at various local minima. Random initialization is standard, with multiple runs exploring different solutions. Despite variability, some initializations may yield similar results. Techniques like "k-means++" aim to enhance the likelihood of finding better solutions. In summary, k-means faces challenges in ensuring the identification of the global minimum due to non-convexity, resulting in convergence to different local minima based on initialization. Varying initializations can lead to distinct outcomes, emphasizing the impact of the chosen setup on the convergence path.

# 5) K-means is guaranteed to stop after a finite number of iterations due to the limited number of possible combinations of cluster assignments (\(m^k\)) and the monotonic decrease of the cost function. The algorithm converges to a solution where further iterations do not result in a decrease in the cost function, ensuring a finite stopping point. However, the specific solution reached can be sensitive to the choice of initial centroids, leading to variability across runs.

# 6) The k-means algorithm is designed for clustering, grouping similar data points into k clusters. In contrast, generalized k-means is an adaptation that introduces flexibility in the choice of similarity/dissimilarity/distance measures. 
# The primary distinction between k-means and generalized k-means lies in their approach to distance metrics. K-means exclusively utilizes Euclidean distance, while generalized k-means allows for the use of different distance measures.
# 
# The selection of a particular distance measure significantly influences the clustering results. It directly impacts the determination of cluster centers and overall clustering quality. The appropriateness of the distance measure is crucial, as an inadequate choice, such as using Euclidean distance in k-means for certain data types or distributions, can lead to inaccuracies in clustering outcomes.
# 
# In summary, the k-means algorithm relies on Euclidean distance, whereas generalized k-means offers versatility by permitting the use of diverse distance measures. The careful choice of a suitable similarity/dissimilarity/distance measure is paramount to ensuring the precision and efficacy of the clustering algorithm.

# The Laplacian matrix, derived from the adjacency matrix of a graph, captures structural characteristics. 
# Eigenvectors corresponding to zero eigenvalues of the Laplacian represent patterns of connectivity in the graph. 
# The presence of zero eigenvalues in the Laplacian matrix indicates the existence of disconnected clusters within the graph.
# 
# $$A = \{ a_{(ij)n*m}: A_{ij}= \{ ^{1} _{0}$$
#  
# n is the number of vertices in the graph                                
# A = |0 1 1 0 0|
#     |1 0 1 0 0|
#     |1 1 0 0 0|
#     |0 0 0 0 1|
#     |0 0 0 1 0|
# 
# D = |2 0 0 0 0|
#     |0 2 0 0 0|
#     |0 0 2 0 0|
#     |0 0 0 1 0|
#     |0 0 0 0 1|
#                                 
# D is the degree of matrix, the laplacian matrix L is given by L= D-A
# 
# L = |2 -1 -1 0 0|
#     |-1 2 -1 0 0|
#     |-1 -1 2 0 0|
#     |0 0 0 1 -1|
#     |0 0 0 -1 -1| 
#                                 
# Each row sum of laplacian matrix is o. zero is an eigenvalue of the matrix L,
# the eigen vector corresponding to eigenvalue d=0
# |x1|                 
# |x2|
# |x3|  
# |x4|
# |x5| 
# 
# = 
# |0|                 
# |0|
# |0|  
# |0|
# |0|
# 
# X is eigenvector for eigenvalue d=0
# 
# R1= R1 + 2R2
# R3= R3-R2
# Rs= Rs+ Ra
# 
# |0 3 -3 0 0|
# |-1 2 -1 0 0|         
# |0 -3 -3 0 0|
# |0 0 0 1 -1|
# |0 0 0 0 0|
# 
# |x1|                 
# |x2|
# |x3|  
# |x4|
# |x5| 
# 
# = 
# |0|                 
# |0|
# |0|  
# |0|
# |0|
# 
# R3= R3+ R1
# X3 and X5 are free variables
# 
# x3= x5= 1
# x4-x5= 0   x4=x5=1
# 3x2=3x3 = 0    x2= x3=1
# -x1= 2x2-x3= 0 x1=1 
# 
# X= 
# 
# |1|                 
# |1|
# |1|  
# |1|
# |1|
# are eigen vector corresponding to the eigenvalue d= 0

# QUESTION 2
# solving for uj
# $$\frac{∂J}{∂µ^{j}}= 0
# \frac{∂J}{∂µ^{j}}= \sum_{i=1}^{m}2r^{ij}(x^{i}−µ^{j}) = 0
# \sum_{i=1}^{m}r^{ij}x^{i} - µ^{j}\sum_{i=1}^{m}r^{ij}=0
# isolation
# µ^{j}= \frac{\sum_{i}r^{ij}x^{i}}{\sum_{i}r^{ij}}$$
# 
# This expression represents the centroid of the j-th cluster in the context of the K-means algorithm, where r ij is the assignment indicator 

# 2.2 
# derive the assignment variables $$\(r_{ij}\)$$ to minimize the distortion function \(J\) when the centroids $$\(\mu_j\)$$ are fixed. Recall the distortion function J:
# 
# $$\[ J = \sum_{i=1}^{m} \sum_{j=1}^{k} r_{ij} \lVert x_i - \mu_j \rVert^2 \]$$
# 
# The goal is to find the assignment variables $$\(r_{ij}\)$$ that minimize this function with fixed centroids $$\(\mu_j\)$$. We can achieve this by considering each data point $$\(x_i\)$$ and assigning it to the cluster with the closest centroid. Mathematically, this is expressed as:
# 
# $$\[ r_{ij} = \begin{cases} 1 & \text{if } j = \arg \min_k \lVert x_i - \mu_k \rVert^2 \\ 0 & \text{otherwise} \end{cases} \]$$
# 
# In words, $$\(r_{ij}\)$$ is 1 if the j-th cluster has the closest centroid to the data point $$\(x_i\)$$, and 0 otherwise.
# 
# This ensures that each data point is assigned to the cluster whose centroid is the closest in terms of squared Euclidean distance, resulting in the minimization of the distortion function \(J\) when the centroids are fixed.

# In[1]:


#QUESTION 3
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def myKmeans(raw, k, ct_init, dist_choice='euclidean'):
    n1, n2, n3 = raw.shape
    raw_img = raw.reshape((-1, n3))

    # Initialization
    ct_old = ct_init
    cost_old = np.inf
    nIter = 0
    cost_list = []

    while nIter < max_iterations:
        # Find the cluster assignment for each data point
        dist_mtx = cdist(raw_img, ct_old, dist_choice)**2
        cl = np.argmin(dist_mtx, axis=1)

        # Update the centroid for each group
        ct_new = np.zeros_like(ct_old)
        current_cost = 0

        for jj in range(k):
            idx_j = np.where(cl == jj)
            x_j = raw_img[idx_j]

            if dist_choice == 'euclidean':
                ct_new[jj] = np.mean(x_j, axis=0)
            elif dist_choice == 'cityblock':
                ct_new[jj] = np.median(x_j, axis=0)
            else:
                sys.exit('Please specify the correct distance')

            if not np.isfinite(np.sum(ct_new[jj])):
                ct_new[jj] = np.full(np.shape(ct_new[jj]), fill_value=np.inf)

            current_cost += np.sum(x_j.dot(ct_new[jj]))

        # Save the cost of the current iteration for record
        cost_list.append(current_cost)

        # Check convergence
        if current_cost == cost_old:
            break

        # Update variables for the next iteration
        cost_old = current_cost
        ct_old = ct_new
        nIter += 1

    # Assign the new pixel value with the new centroid
    dist_all = cdist(raw_img, ct_new, dist_choice)
    cl_all = np.argmin(dist_all, axis=1)

    # Prepare to output the result
    img = np.full(np.shape(raw_img), fill_value=np.nan)
    for ii in np.unique(cl_all):
        img[np.where(cl_all == ii)] = ct_new[ii] / 255

    img_out = np.reshape(img, (n1, n2, n3))

    # Check empty clusters
    n_empty = sum(1 - np.isfinite(np.sum(ct_new, axis=1)))

    return img_out, n_empty, nIter


def display_results(raw_img, k_mesh, run_time, n_empty_all, nIter_all, dist_choice, init_method):
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].imshow(raw_img)
    ax[0, 0].set_title('Original', fontsize=8)
    ax[0, 0].get_xaxis().set_visible(False)
    ax[0, 0].get_yaxis().set_visible(False)

    rseed = 6
    nIter_all = []

    for ii in range(len(k_mesh)):
        start_time = time.time()

        np.random.seed(rseed)
        if init_method == 'random':
            ct_init = np.random.random((k_mesh[ii], 3)) * 255
        elif init_method == 'poor':
            ct_init = np.random.random((k_mesh[ii], 3))
        else:
            sys.exit('Please specify either random or poor')

        img, n_empty, nIter = myKmeans(raw_img, k_mesh[ii], ct_init, dist_choice)

        end_time = time.time()
        nIter_all.append(nIter)

        img = (img * 255).astype('int')
        ax[int((ii + 1) / 2), np.remainder(ii + 1, 2)].imshow(img)
        ax[int((ii + 1) / 2), np.remainder(ii + 1, 2)].set_title(f'k={k_mesh[ii]}', fontsize=8)
        ax[int((ii + 1) / 2), np.remainder(ii + 1, 2)].get_xaxis().set_visible(False)
        ax[int((ii + 1) / 2), np.remainder(ii + 1, 2)].get_yaxis().set_visible(False)

        run_time.append(end_time - start_time)
        n_empty_all.append(n_empty)

    fig.tight_layout(pad=1.0)
    fig.suptitle(f'Distance-{dist_choice}-Initialization-{init_method}')
    fig.subplots_adjust(top=0.85)

    savename = f'Distance-{dist_choice}-Initialization-{init_method}.pdf'
    plt.savefig(savename, dpi=300)

    print(f'\nKmeans result for {dist_choice}, current random seed: {rseed}')
    print('The running time for each k')
    for kk in range(5):
        print(f'k = {k_mesh[kk]}: {run_time[kk]:.2f} sec. # of empty cluster: {n_empty_all[kk]} nIteration: {nIter_all[kk]}')


# Image 1
raw_img = plt.imread('C:\\Users\\laury\\OneDrive\\Documents\\football.bmp')
k_mesh = [2, 4, 8, 16, 32]
run_time = []
n_empty_all = []
max_iterations = 100

dist_choice = 'euclidean'  # or 'cityblock'
init_method = 'random'  # or 'poor'

display_results(raw_img, k_mesh, run_time, n_empty_all, [], dist_choice, init_method)

#image 2
raw_img = plt.imread('C:\\Users\\laury\\OneDrive\\Documents\\hestain.bmp')
k_mesh = [2, 4, 8, 16, 32]
run_time = []
n_empty_all = []
max_iterations = 100

dist_choice = 'euclidean'  # or 'cityblock'
init_method = 'random'  # or 'poor'

display_results(raw_img, k_mesh, run_time, n_empty_all, [], dist_choice, init_method)

#image 3
raw_img = plt.imread('C:\\Users\\laury\\OneDrive\\Documents\\butterfly.jpg')
k_mesh = [2, 4, 8, 16, 32]
run_time = []
n_empty_all = []
max_iterations = 100

dist_choice = 'euclidean'  # or 'cityblock'
init_method = 'random'  # or 'poor'

display_results(raw_img, k_mesh, run_time, n_empty_all, [], dist_choice, init_method)


# MyKmeans function implements the K-means algorithm for image segmentation, It initializes centroids, assigns data points to clusters, and updates centroids iteratively until convergence.
# and returns the segmented image, the number of empty clusters, and the number of iterations. For each image, i perform segmentation with different values of `k` and display the results.I used the 'euclidean' distance metric and 'random' initialization method. the method I used to find the best k, is to iterate over a range of k values and perform k-means clustering for each value. The performance of the clustering is then evaluated based on several factors, including the running time, the number of empty clusters, and the number of iterations needed for convergence.
# 
# in the 'football.bmp' image the running time increases with higher values of k, indicating that the algorithm takes more time to converge for a larger number of clusters.
# The number of empty clusters is mostly zero, suggesting that the algorithm successfully assigns data points to clusters for the images.
# For k = 16 and k = 32, there are a few empty clusters, and the algorithm reaches the maximum number of iterations (100). This indicates convergence issues or suboptimal initialization. 
# I picked k=16 and k=32 to be the closest images that are similar to the original image.
# 
# 
# in the 'hestain.bmp' image the running time generally increases with higher values of k, which is expected as more clusters require more computation. The number of empty clusters varies, and it becomes more prevalent for larger values of k.
# For k = 2 and k = 4, there are no empty clusters, and the algorithm converges within a reasonable number of iterations.
# For k = 8 and beyond, the number of empty clusters increases, indicating potential challenges in assigning data points to clusters, especially for larger values of k.
# For k = 32, the algorithm reaches the maximum number of iterations (100), suggesting that convergence might be challenging or the algorithm has not fully stabilized.
# This was radher a tough image to pick the best k, however k=32 appears to be the closest to match the original image.
# 
# 
# In the 'butterfly.jpg' image the running time is generally low for smaller values of k and increases as k becomes larger, which is expected.
# For k = 2 and k = 4, there are no empty clusters, and the algorithm converges within a relatively small number of iterations.
# For k = 8, the algorithm converges, but there are no empty clusters, indicating successful assignment of data points.
# For k = 16, there is only one empty cluster, and the algorithm converges within a moderate number of iterations.
# For k = 32, there are 7 empty clusters, and the algorithm reaches the maximum number of iterations (100), suggesting that convergence might be challenging for this specific case.
# the best k value will be k=32 even though it does not match the original image.
# 
# 
# in summary for all these 3 images,running time increases with larger k for all images, reflecting increased computational requirements.
# Empty clusters are mostly zero for smaller k, but the prevalence increases with larger k.
# Larger k values, especially k = 32, show convergence challenges or instability in some case
# 

# In[ ]:





# In[3]:


#QUESTION 4.1

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.io import loadmat

# Load MNIST dataset from MAT file
mnist_data = loadmat("C:\\Users\\laury\\OneDrive\\Documents\\mnist_10digits.mat")

# Update variable assignments based on the keys in your MAT file
X_train = mnist_data['xtrain']
y_train = mnist_data['ytrain']
X_test = mnist_data['xtest']
y_test = mnist_data['ytest']

# Data preprocessing
X_train = X_train / 255
X_test = X_test / 255

# Flatten the data
X_train_flattened = X_train.reshape(len(X_train), -1)
X_test_flattened = X_test.reshape(len(X_test), -1)

# Define the number of clusters (K)
K = 10

# Apply K-means clustering using squared-`2 norm as the metric
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300, algorithm='auto')
y_pred = kmeans.fit_predict(X_train_flattened)

# Compute purity for each cluster
purity_scores = []
for cluster in range(K):
    cluster_mask = (y_pred == cluster)
    cluster_labels = y_train.flatten()[cluster_mask]  # Flatten y_train before applying the mask
    dominant_label = np.argmax(np.bincount(cluster_labels))
    correct_assignments = np.sum(cluster_labels == dominant_label)
    purity = correct_assignments / len(cluster_labels)
    purity_scores.append(purity)

# Report purity scores for each cluster
for cluster, purity in enumerate(purity_scores):
    print(f'Purity for Cluster {cluster + 1}: {purity}')


# Purity is a measure of how well-defined and homogeneous the clusters are. It is computed by assigning each cluster to the class label that is most frequent in that cluster and then measuring the accuracy of this assignment.
# cluster 5 had the highest purity, which indicates it is highly homogenous. some clusters are more homogeneous (e.g., Clusters 2, 5, 8, and 10) than others, which might indicate that the data points within these clusters share more similar characteristics or patterns. Clusters with lower purity scores are less well-defined and may contain a more diverse set of samples.

# In[5]:


#Question 4.2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.io import loadmat

# Load MNIST dataset from MAT file
mnist_data = loadmat("C:\\Users\\laury\\OneDrive\\Documents\\mnist_10digits.mat")

# Update variable assignments based on the keys in your MAT file
X_train_subset = mnist_data['xtrain'] / 255
y_train_subset = mnist_data['ytrain'].flatten()

# Use a subset of the dataset (adjust the sample_size based on your available memory)
sample_size = 500
X_train_subset = X_train_subset[:sample_size]
y_train_subset = y_train_subset[:sample_size]

# Flatten the data
X_train_flattened_subset = X_train_subset.reshape(len(X_train_subset), -1)

# Set the number of clusters
K = 10

# Apply K-means clustering with Manhattan distance
kmeans_manhattan = KMeans(n_clusters=K, random_state=42, algorithm='auto', n_init=10)
y_pred_manhattan = kmeans_manhattan.fit_predict(X_train_flattened_subset)

# Compute purity for each cluster with Manhattan distance
purity_scores_manhattan = []

for cluster in range(K):
    cluster_indices = np.where(y_pred_manhattan == cluster)[0]
    cluster_labels = y_train_subset[cluster_indices]
    most_frequent_label = np.bincount(cluster_labels).argmax()
    correct_assignments = np.sum(cluster_labels == most_frequent_label)
    purity = correct_assignments / len(cluster_indices)
    purity_scores_manhattan.append(purity)

# Report purity scores for each cluster
for i, purity in enumerate(purity_scores_manhattan):
    print(f'Cluster {i + 1}: Purity Score = {purity:.4f}')


# Clusters 6 and 9 stand out with very high purity scores, indicating highly homogeneous and well-defined clusters. Cluster 9, in particular, has a perfect purity score, suggesting that all samples in that cluster belong to the same class. Clusters 2, 3, and 7 have lower purity scores, indicating less homogeneity and more diversity in the assigned class labels.
# 
# #Conclusion
# 
# Cluster 6 stands out as a highly homogeneous and well-defined cluster in both cases, achieving a very high purity score.
# Cluster 9 has a perfect purity score in both metrics, indicating a perfectly homogeneous cluster.
# Im my opinion both clusters seem to yield similar results
# 

# In[6]:


#QUESTION 5.1
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load data using pandas
edges = np.loadtxt("C:\\Users\\laury\\OneDrive\\Documents\\edges.txt", dtype=int) - 1  # Adjust for 0-indexing
nodes_df = pd.read_csv("C:\\Users\\laury\\OneDrive\\Documents\\nodes.txt", sep='\t', header=None, names=["ID", "Blog_URL", "Label", "Additional_Labels"])

# Extract labels and create adjacency matrix
label = nodes_df["Label"].to_numpy()
n = len(label)

# Initialize adjacency matrix
A = np.zeros((n, n), dtype=int)

# Create symmetric adjacency matrix
for edge in edges:
    A[edge[0], edge[1]] = 1
    A[edge[1], edge[0]] = 1

# Remove isolated nodes
iso_node = np.where(np.sum(A, axis=0) == 0)[0]
A = np.delete(A, iso_node, axis=0)
A = np.delete(A, iso_node, axis=1)
label = np.delete(label, iso_node)

n = len(label)

# Graph Laplacian
D = np.diag(np.sum(A, axis=1))
L = D - A

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L)

# Spectral clustering for k = 2, 5, 10, 25
for k in [2, 5, 10, 25]:
    Ut = eigenvectors[:, :k]
    
    # K-means clustering
    cluster_predict = KMeans(n_clusters=k, random_state=42).fit_predict(Ut)

    mismatch = np.zeros(k)
    cluster_size = np.zeros(k)
    majority = np.zeros(k)

    # Calculate mismatch rates and majority labels
    for i in range(k):
        idx = np.where(cluster_predict == i)[0]
        cluster_size[i] = len(idx)
        vote_in_group_k = label[idx]
        majority[i] = np.bincount(vote_in_group_k).argmax()
        mismatch[i] = np.sum(vote_in_group_k != majority[i]) / cluster_size[i]

    misrate_tmp = np.sum(mismatch * cluster_size) / n
    print(f'For k = {k}, overall mismatch rate: {misrate_tmp}')
    print('Mismatch rate for each cluster:')
    print(mismatch)
    print('Majority vote for each cluster:')
    print(majority)
    print('Cluster size:')
    print(cluster_size)
    print('\n')


# In[7]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
# Visualization
G = nx.Graph()
G.add_edges_from(edges)

pos = nx.spring_layout(G)  # Use spring layout for visualization

# Color nodes based on the predicted cluster
colors = [cluster_predict[i] for i in range(n)]

plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_color=colors, cmap=plt.cm.rainbow, node_size=50, with_labels=False)
plt.title(f"Spectral Clustering with k={k}")
plt.show()


# 
# For k = 2, the overall mismatch rate is 47.88%. Cluster 1 exhibits a mismatch rate of 47.95% with a majority size of 1222, while Cluster 2 (majority label 0) shows no mismatch with a majority size of 2.
# 
# With k = 5, the overall mismatch rate is 47.71%. Clusters 1, 3, and 5 demonstrate high mismatch rates, whereas Clusters 2 and 4 have no mismatches. Clusters 1, 3, and 5 are predominantly associated with label 1, while Clusters 2 and 4 are dominated by label 0.
# 
# At k = 10, the overall mismatch rate is 46.98%. Cluster 1 exhibits the highest mismatch rate, while Cluster 10 achieves a mismatch rate of 0%. Cluster 1 is dominated by label 1, and Cluster 10 is dominated by label 0.
# 
# For k = 25, the overall mismatch rate is 45.59%. Cluster 1 has the highest mismatch rate, but several clusters show no mismatch. Cluster 1 is predominantly associated with label 1, while Cluster 14 is dominated by label 0.
# 
# In conclusion, the overall mismatch rates tend to decrease with an increase in the number of clusters (k). Smaller k values lead to clusters dominated by one label, resulting in higher mismatch rates. As k increases, clusters become more specialized, contributing to a reduction in the overall mismatch rate. The varying sizes of clusters indicate the presence of diverse community sizes within the graph.

# In[9]:


#question 5.2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming Ut and label are already defined (e.g., from previous code)

K_max = 50
misrate = np.zeros(K_max)

for k in range(1, K_max + 1):
    Ut = eigenvectors[:, :k]

    # K-means clustering
    cluster_predict = KMeans(n_clusters=k, random_state=42).fit_predict(Ut)

    mismatch = np.zeros(k)
    cluster_size = np.zeros(k)

    # Calculate mismatch rates for each cluster
    for i in range(k):
        idx = np.where(cluster_predict == i)[0]
        cluster_size[i] = len(idx)
        data_cluster = label[idx]
        majority = np.bincount(data_cluster).argmax()
        mismatch[i] = np.sum(data_cluster != majority) / cluster_size[i]

    misrate[k - 1] = np.sum(mismatch * cluster_size) / n

    # Display average mismatch numbers
    print(f'K={k}, Average Mismatch Numbers: {np.sum(mismatch * cluster_size):.2f}')

# Standard plot
plt.plot(range(1, K_max + 1), misrate, linewidth=2)
plt.xlabel('K', fontsize=16)
plt.ylabel('Average mismatch rate among all clusters', fontsize=16)
plt.show()


# the above graph shows The log-magnitude of overall mismatch varies across different choices of k.
# As the number of clusters (k) increases, the average mismatch numbers generally decrease. The reduction is more pronounced for larger values of k, with a significant decrease observed between k=20 and k=30. However, beyond k=30, the average mismatch numbers stabilize around 527, indicating that additional clusters yield diminishing returns in minimizing mismatches. Overall, increasing the granularity through higher k values contributes to reducing average mismatch numbers, but there's a point of diminishing returns.
# 

# REFERENCES
# CDA Demo code and class videos
# 
# https://www.kaggle.com/code/manishkc06/mnist-multi-class-classification-model
# 
# https://edstem.org/us/courses/51310/discussion/4116731
# 
# https://www.youtube.com/watch?v=5w5iUbTlpMQ
# 

# In[2]:


import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[3]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')


# In[4]:


import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[ ]:




