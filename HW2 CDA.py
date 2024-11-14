#!/usr/bin/env python
# coding: utf-8

# #1.1
# The optimization problem for Principal Component Analysis (PCA) can be formulated as follows: maximize the expression $$w^TCw$$ subject to the constraint $$\|w\| \leq 1$$. Here, $$w$$ represents a direction in the feature space,$$C$$ is the covariance matrix, and $$\lambda$$ is an eigenvalue. Solving for the optimal $$w$$ leads to the eigendecomposition problem $$Cw = \lambda w$$.
# 
# Through this eigendecomposition, we recognize that $$w^TCw = \lambda w^Tw = \lambda \|w\|^2$$. It is evident that for the optimal solution, where the constraint is binding $$(\|w\| = 1)$$, we aim to maximize $$\lambda$$ since it corresponds to the variance. This implies that the weight vector required for the first principal component direction is the eigenvector associated with the largest eigenvalue of the sample covariance matrix. In essence, this largest eigenvalue maximizes the objective function, which represents variance.

# #1.2
# The optimization problem in Principal Component Analysis (PCA) involves maximizing $$w^TCw$$ subject to the constraint $$\|w\| \leq 1$$, where $$w$$ is a direction in the feature space, $$C$$ is the covariance matrix, and $$\lambda$$ is an eigenvalue. The solution involves finding the eigenvector associated with the largest eigenvalue, as it maximizes the objective function related to variance.
# 
# To find the third-largest principal component direction:
# 1. Perform eigenvalue decomposition on $$C$$ to obtain $$Cw_i = \lambda_iw_i$$.
# 2. Identify and select the third-largest eigenvalue $$\lambda_3$$.
# 3. Extract the corresponding eigenvector $$w_3$$.
# 4. Normalize $$w_3$$ to ensure $$\|w_3\| = 1$$.
# 
# The resulting $$w_3$$ represents the third principal component direction, capturing the variance in the data along this axis.

# #1.3
# 
# consider the probability density of a Gaussian distribution $$N(\mu, \sigma) \in \mathbb{R}$$:
# 
# $$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
# 
# Next, assuming our sample data is independent and identically distributed (i.i.d), the likelihood can be expressed as:
# 
# $$\mathcal{L} = \prod_{i=1}^{m} f(x_i) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$
# 
# Simplifying, we find the log-likelihood, using natural logarithm properties:
# 
# $$\log(\mathcal{L}) = -\frac{m}{2} \log(2\pi\sigma^2) - \sum_{i=1}^{m} \frac{(x_i - \mu)^2}{2\sigma^2}$$
# 
# Now, let's take the partial derivatives of $$log(\mathcal{L})$$ with respect to $$\mu\$$  and σ, and set them equal to 0.
# 
# For $$\mu\$$:
# 
# $$\frac{1}{\sigma^2} \sum_{i=1}^{m} (x_i - \mu) = 0 \implies \mu = \frac{1}{m} \sum_{i=1}^{m} x_i$$
# 
# For $$\sigma^2$$:
# 
# $$-\frac{m}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^{m} (x_i - \mu)^2 = 0 \implies \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$
# 
# Furthermore, we show that the Hessian for $$\log(\mathcal{L})$$ can be written as:
# 
# $$\text{Hessian} = \sum_{i=1}^{m} \left(-\frac{1}{\sigma^2}\right) - \sum_{i=1}^{m} \frac{(x_i - \mu)^2}{\sigma^4}$$
# 
# Since $4\mu = \bar{x}$$ and $$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2$$, we substitute these Maximum Likelihood Estimators (MLEs) and obtain:
# 
# $$\text{Hessian} = -\frac{m}{\hat{\sigma}^2} - \frac{m}{\hat{\sigma}^2} = -\frac{2m}{\hat{\sigma}^2}$$
# 
# Since this Hessian matrix is negative definite (as $$m, \hat{\sigma} > 0$$), and the determinant of the Hessian is negative, we conclude that $$\hat{\mu} = \bar{x}$$ and $$\hat{\sigma}^2$$ are Maximum Likelihood Estimators.

# #1.4
# 
# 
# ISOMAP incorporates three fundamental concepts, which can be viewed as key elements or sequential stages:
# 
# 1. Neighborhood Determination:
#    - The first idea involves establishing a set of neighbors on a manifold M. This can be achieved by either selecting each point's \(K\) nearest neighbors or by connecting each point to all others within a fixed radius $$\epsilon$$.
# 
# 2. Geodesic Distance Estimation:
#    - The second concept entails calculating the shortest path distance between all pairs of points on the manifold. This results in the estimation of geodesic distances between all data pairs, represented as a graph matrix D.
# 
# 3. Multi-Dimensional Scaling (MDS):
#    - The third idea involves the application of Multi-Dimensional Scaling (MDS). This step aims to discover a low-dimensional representation that preserves the distance information derived from the graph matrix D.
# 
# In summary, ISOMAP progresses by first defining neighborhood relationships on the manifold, then estimating geodesic distances through shortest paths, and finally obtaining a lower-dimensional representation via MDS, ensuring the preservation of distance relationships.

# #1.5
# 
# To decide the number of principal components k from data in Principal Component Analysis (PCA):
# 
# 1. Scree Plot:
#    - Look for the "elbow" point where adding more components results in diminishing returns in explained variance.
# 
# 2. Cumulative Explained Variance:
#    - Choose k such that a desired percentage (e.g., 95% or 99%) of cumulative explained variance is retained.
# 
# 3. Cross-Validation:
#    - Evaluate model performance with different k values using cross-validation and select the one with the best performance on a validation set.
# 
# 4. Information Criteria:
#    - Minimize information criteria (e.g., AIC, BIC) to balance model fit and simplicity.
# 
# 5. Eigenvalue Threshold:
#    - Choose k as the number of eigenvalues above a specified threshold, assuming significant principal components have significantly greater eigenvalues.
# 
# 6. Cross-Validation for Model Performance:
#    - Assess predictive performance for different k values and select the one yielding the best performance on unseen data.
# 
# 7. Domain Knowledge:
#    - Consider any prior knowledge or expectations about the data and its underlying structure.
# 
# 

# #1.6
# 
# Outliers can significantly impact the performance of Principal Component Analysis (PCA) by influencing the estimation of the covariance matrix and, consequently, the principal components. Outliers may distort the relationships between variables, leading to unreliable results. Example;

# In[2]:


# A dataset with two variables, X and Y are positively correlated, and we introduce an outlier:

import numpy as np
from sklearn.decomposition import PCA

# Original Data
data_original = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]])

# Data with Outliers
data_with_outliers = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15], [6, 30, 45]])

# PCA on Original Data
pca_original = PCA(n_components=3)
pca_original.fit(data_original)
variance_explained_original = pca_original.explained_variance_ratio_

# PCA on Data with Outliers
pca_outliers = PCA(n_components=3)
pca_outliers.fit(data_with_outliers)
variance_explained_outliers = pca_outliers.explained_variance_ratio_

print("Variance Explained (Original Data):", variance_explained_original)
print("Variance Explained (Data with Outliers):", variance_explained_outliers)



# the introduction of outliers in variables B and C may distort the covariance matrix and alter the variance explained by each principal component. This illustrates how outliers can impact the performance of PCA and the interpretation of underlying structures in the data.

# In[1]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')


# In[2]:


import os
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[13]:


#Question 2a
import csv
import numpy as np
from matplotlib import pyplot as plt

# Load data from CSV file with a relative path
raw = []
with open('food-consumption.csv', newline='') as file:
    filereader = csv.reader(file, delimiter=',')
    header = next(filereader)
    for row in filereader:
        raw.append(row)

raw = np.array(raw)

# Extract country names, food names, and numerical data
country_names = raw[:, 0]
food_names = np.array(header[1:])
data = raw[:, 1:].astype(float)

# Centering the data
mu = np.mean(data, axis=0)
demean = data - np.tile(mu, [len(data), 1])

# Performing PCA
C = demean.T @ demean / len(data)
_, s, u = np.linalg.svd(C)

# Extracting the first two principal components
k = 2
pc = demean @ u[:, :k] @ np.diag(1/np.sqrt(s[:k]))

# Scatter plot of two-dimensional representations
plt.figure()
plt.scatter(pc[:, 0], pc[:, 1], s=20, facecolors='none', edgecolors='r')
plt.axis('equal')

# Annotating countries on the plot
for ii, txt in enumerate(country_names):
    plt.annotate(txt, (pc[ii, 0], pc[ii, 1]), fontsize=5)

plt.grid()
plt.title('PCA Plot for Food Consumption by Countries')
plt.show()


# The scatter plot visually represents the countries in a two-dimensional space based on their food consumption patterns.
# Proximity of countries in the plot suggests similarity in their food consumption profiles.
# Outliers or distinct clusters may indicate countries with unique dietary habits
# From the plot we see relative consumption patterns from different countries, Spain and Portugal have almost similar patterns. it appears that Switzerland, Iceland and Denmark have unique dietary habits.

# In[16]:


#Question 2b
import csv
import numpy as np
from matplotlib import pyplot as plt
 

# Extract country names, food names, and numerical data
country_names = raw[:, 0]
food_names = np.array(header[1:])
data = raw[:, 1:].astype(float)

# Transpose data to treat country consumptions as features for each food item
data_transposed = data.T
mu_food = np.mean(data_transposed, axis=0)
demean_food = data_transposed - np.tile(mu_food, [len(food_names), 1])

# Performing PCA for food items
C_food = demean_food.T @ demean_food / len(food_names)
u_food, s_food, _ = np.linalg.svd(C_food)
pc_food = demean_food @ u_food[:, :2] @ np.diag(1/np.sqrt(s_food[:2]))

# Scatter plot of two-dimensional representations for food items
plt.figure()
plt.scatter(pc_food[:, 0], pc_food[:, 1], s=20, facecolors='none', edgecolors='r')
plt.axis('equal')

# Annotating food items on the plot
for ii, txt in enumerate(food_names):
    plt.annotate(txt, (pc_food[ii, 0], pc_food[ii, 1]), fontsize=8)

plt.grid()
plt.title('PCA Plot for Food Items')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In the depicted plot, garlic and olive exhibit the most distinctive consumption patterns, with countries either strongly favoring or disfavoring these foods. Additionally, a notable negative correlation in preference is observed between real coffee and tin soup, suggesting that countries tend to either prefer real coffee while disliking tin soup, or vice versa.

# In[4]:


pip install numpy scikit-learn scipy


# In[7]:


#Question 3a
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.utils.graph import graph_shortest_path
import scipy.io

# Load the data
mat_data = scipy.io.loadmat('isomap.mat')
I = mat_data['images']

# Number of images
num_images = I.shape[1]

# Chunk size to reduce memory usage
chunk_size = 100

# Initialize the weighted matrix
weighted_matrix = np.zeros((num_images, num_images))

# Define the epsilon parameter for ϵ-ISOMAP
epsilon = 10.0  # You may need to adjust this parameter based on your data

# Construct weighted matrix in chunks
for i in range(0, num_images, chunk_size):
    chunk_end = min(i + chunk_size, num_images)
    distances_chunk = np.linalg.norm(I[:, :, None] - I[:, None, i:chunk_end], axis=0)
    weighted_matrix[:, i:chunk_end] = np.exp(-(distances_chunk / epsilon) ** 2)

# Visualize the weighted adjacency matrix
plt.figure(figsize=(10, 10))
plt.imshow(weighted_matrix, cmap='viridis', interpolation='none')
plt.title('Weighted Adjacency Matrix Visualization')
plt.colorbar(label='Weight')
plt.show()

# Construct the binary adjacency matrix based on a threshold (e.g., 0.1)
binary_adjacency_matrix = (weighted_matrix > 0.1).astype(int)

# Visualize the binary adjacency matrix as a graph
G = nx.from_numpy_matrix(binary_adjacency_matrix)
pos = nx.spring_layout(G)

plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, font_size=8, node_size=10)
plt.title('Binary Adjacency Matrix Graph Visualization')
plt.show()


# In[12]:


pip install networkx


# In[15]:


#question 3a(to keep)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io

# Load the data
mat_data = scipy.io.loadmat('isomap.mat')
I = mat_data['images']

# Number of images
num_images = I.shape[1]

# Chunk size to reduce memory usage
chunk_size = 100

# Initialize the distance matrix
D = np.zeros((num_images, num_images))

# Construct distance matrix D in chunks
for i in range(0, num_images, chunk_size):
    chunk_end = min(i + chunk_size, num_images)
    D[:, i:chunk_end] = np.linalg.norm(I[:, :, None] - I[:, None, i:chunk_end], axis=0)

# Define the epsilon parameter for ϵ-ISOMAP
epsilon = 10.0  # You may need to adjust this parameter based on your data

# Construct the weighted adjacency matrix using Gaussian weights
weights_matrix = np.exp(-(D / epsilon) ** 2)

# Construct the adjacency matrix for the nearest neighbor graph
adjacency_matrix = (D < epsilon).astype(int)

# Visualize the weighted adjacency matrix
plt.figure(figsize=(10, 10))
plt.imshow(weights_matrix, cmap='viridis', interpolation='none')
plt.title('Weighted Adjacency Matrix Visualization')
plt.colorbar(label='Weight')
plt.show()

# Visualize the nearest neighbor graph
G = nx.from_numpy_matrix(adjacency_matrix)
pos = nx.spring_layout(G)  # You can use other layout algorithms as well

plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, font_size=8, node_size=10)
plt.title('Nearest Neighbor Graph Visualization')
plt.show()

# Use networkx to compute shortest paths
shortest_paths = dict(nx.shortest_path_length(G))

# Randomly select a few nodes for illustration
selected_nodes = np.random.choice(num_images, size=5, replace=False)

# Plot images corresponding to selected nodes
plt.figure(figsize=(15, 5))
for i, node in enumerate(selected_nodes):
    plt.subplot(1, len(selected_nodes), i + 1)
    plt.imshow(I[:, node].reshape(64, 64), cmap='gray')
    plt.title(f'Node {node}')
    plt.axis('off')

plt.show()


# In the visual representation above, we observe the weighted adjacency matrix, where each entry signifies the distance in the nearest neighbor graph. The color intensity reflects the weight of the connections, with darker shades indicating stronger associations. The dots pinpoint the positions of randomly selected images, providing insights into their relationships within the context of the nearest neighbor graph

# In[13]:


# Question 3b
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
import scipy.io

# Load the data
mat_data = scipy.io.loadmat('isomap.mat')
I = mat_data['images']

# Number of images
num_images = I.shape[1]

# Chunk size to reduce memory usage
chunk_size = 100

# Initialize the distance matrix
D = np.zeros((num_images, num_images))

# Construct distance matrix D in chunks
for i in range(0, num_images, chunk_size):
    chunk_end = min(i + chunk_size, num_images)
    D[:, i:chunk_end] = np.linalg.norm(I[:, :, None] - I[:, None, i:chunk_end], axis=0)

# Define the epsilon parameter for ϵ-ISOMAP
epsilon = 10.0  # You may need to adjust this parameter based on your data

# Construct the weighted adjacency matrix using Gaussian weights
weights_matrix = np.exp(-(D / epsilon) ** 2)

# Construct the adjacency matrix for the nearest neighbor graph
adjacency_matrix = (D < epsilon).astype(int)

# Visualize the weighted adjacency matrix
plt.figure(figsize=(10, 10))
plt.imshow(weights_matrix, cmap='viridis', interpolation='none')
plt.title('Weighted Adjacency Matrix Visualization')
plt.colorbar(label='Weight')
plt.show()

# Use networkx to compute shortest paths
shortest_paths = dict(nx.shortest_path_length(nx.from_numpy_matrix(adjacency_matrix)))

# Compute the ISOMAP embedding
n_components = 2
model = Isomap(n_neighbors=5, n_components=n_components)
embedding = model.fit_transform(I.T)

# Scatter plot of the ISOMAP embedding
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], s=20, c='b', marker='o', edgecolors='k')

# Annotate random images on the plot
selected_nodes = np.random.choice(num_images, size=5, replace=False)
for i, node in enumerate(selected_nodes):
    plt.scatter(embedding[node, 0], embedding[node, 1], s=50, c='r', marker='x', label=f'Node {node}')

# Add labels and title
plt.xlabel('ISOMAP Dimension 1')
plt.ylabel('ISOMAP Dimension 2')
plt.title('ISOMAP Embedding')
plt.legend()
plt.show()

# Plot images corresponding to selected nodes
plt.figure(figsize=(15, 5))
for i, node in enumerate(selected_nodes):
    plt.subplot(1, len(selected_nodes), i + 1)
    plt.imshow(I[:, node].reshape(64, 64), cmap='gray')
    plt.title(f'Node {node}')
    plt.axis('off')

plt.show()



# Scatter plot of the ISOMAP embedding with embedded images
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(embedding[:, 0], embedding[:, 1], s=20, c='b', marker='o', edgecolors='k')

# Annotate images on the plot
for i in range(num_images):
    img = I[:, i].reshape(64, 64)
    imagebox = OffsetImage(img, zoom=0.2, cmap='gray')
    ab = AnnotationBbox(imagebox, embedding[i], frameon=False, pad=0)
    ax.add_artist(ab)

# Add labels and title
ax.set_xlabel('ISOMAP Dimension 1')
ax.set_ylabel('ISOMAP Dimension 2')
ax.set_title('ISOMAP Embedding with Embedded Images')
plt.show()


# The plot above reveals noticeable trends in the results. As we traverse along the manifold, the facial poses exhibit gradual changes, indicating a smooth transition between different orientations. On a global scale, there is a discernible 'drifting' pattern, suggesting a cohesive transformation across the entire set of faces.

# In[17]:


#Question 3C
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io

# Load the data
mat_data = scipy.io.loadmat('isomap.mat')
I = mat_data['images']

# Number of images
num_images = I.shape[1]

# Chunk size to reduce memory usage
chunk_size = 100

# Initialize the distance matrix
D = np.zeros((num_images, num_images))

# Construct distance matrix D in chunks
for i in range(0, num_images, chunk_size):
    chunk_end = min(i + chunk_size, num_images)
    D[:, i:chunk_end] = np.linalg.norm(I[:, :, None] - I[:, None, i:chunk_end], axis=0)

# Define the epsilon parameter for ϵ-ISOMAP
epsilon = 10.0  # You may need to adjust this parameter based on your data

# Construct the weighted adjacency matrix using Gaussian weights
weights_matrix = np.exp(-(D / epsilon) ** 2)

# Construct the adjacency matrix for the nearest neighbor graph
adjacency_matrix = (D < epsilon).astype(int)

# Use networkx to compute shortest paths
shortest_paths = dict(nx.shortest_path_length(nx.from_numpy_matrix(adjacency_matrix)))

# Compute the ISOMAP embedding
n_components = 2
model = Isomap(n_neighbors=5, n_components=n_components)
embedding_isomap = model.fit_transform(I.T)

# Compute PCA projection
X = I.reshape((-1, num_images)).T
pca_model = PCA(n_components=2)
projection_pca = pca_model.fit_transform(X)

# Scatter plot of the ISOMAP embedding
plt.figure(figsize=(15, 8))

# Plot ISOMAP
plt.subplot(1, 2, 1)
plt.scatter(embedding_isomap[:, 0], embedding_isomap[:, 1], s=20, c='b', marker='o', edgecolors='k')
plt.title('ISOMAP Embedding')
plt.xlabel('ISOMAP Dimension 1')
plt.ylabel('ISOMAP Dimension 2')

# Annotate images on the ISOMAP plot
for i in range(num_images):
    img = I[:, i].reshape(64, 64)
    imagebox = OffsetImage(img, zoom=0.2, cmap='gray')
    ab = AnnotationBbox(imagebox, embedding_isomap[i], frameon=False, pad=0)
    plt.gca().add_artist(ab)

# Plot PCA
plt.subplot(1, 2, 2)
plt.scatter(projection_pca[:, 0], projection_pca[:, 1], s=20, c='g', marker='s', edgecolors='k')
plt.title('PCA Projection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Annotate random images on the PCA plot
for i, node in enumerate(selected_nodes):
    plt.scatter(projection_pca[node, 0], projection_pca[node, 1], s=50, c='b', marker='x', label=f'Node {node}')

plt.legend()
plt.show()


# In contrast to the ISOMAP embedding, the PCA projection fails to capture the inherent structure of the manifold, revealing a limitation in representing the global relationships among the data points. The PCA plot exhibits local similarities among faces, lacking the ability to encapsulate the broader, interconnected changes observed in the ISOMAP embedding. As a result, the faces in the PCA projection appear similar only within localized regions, emphasizing the superior capability of ISOMAP in preserving the global structure of the dataset.

# In[23]:


pip install pillow


# In[25]:


#QUestion 4a
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the images
datadir = 'yalefaces/'
with_svd = False

# Function to perform downsampling
def downsample_image(image):
    return np.asarray(image)[::4, ::4].reshape(1, -1)

# Load images and perform downsampling
s01 = np.empty((0, 16*16))
s02 = np.empty((0, 16*16))

# Determine the target image size
target_size = 16 * 16

for f in os.listdir(datadir):
    if f.endswith('.gif'):
        im = downsample_image(Image.open(datadir + f))

        # Check and resize images to the target size
        if im.shape[1] != target_size:
            im = im[:, :target_size]

        if f.startswith('subject01'):
            if f.endswith('test.gif'):
                s01_test = im
            else:
                s01 = np.row_stack((s01, im))
        elif f.startswith('subject02'):
            if f.endswith('test.gif'):
                s02_test = im
            else:
                s02 = np.row_stack((s02, im))

# Perform PCA analysis
mu01 = np.mean(s01, axis=0)
mu02 = np.mean(s02, axis=0)

x01 = s01 - np.tile(mu01, (s01.shape[0], 1))
x02 = s02 - np.tile(mu02, (s02.shape[0], 1))

if with_svd:
    # SVD on the data matrix
    u01, _, _ = np.linalg.svd(x01.T)
    u02, _, _ = np.linalg.svd(x02.T)
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]
else:
    a01 = np.cov(x01, rowvar=False)
    a02 = np.cov(x02, rowvar=False)
    _, evec01 = np.linalg.eigh(a01)
    _, evec02 = np.linalg.eigh(a02)
    u01 = np.real(evec01[:, ::-1])
    u02 = np.real(evec02[:, ::-1])
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]

# Reshape and visualize eigenfaces
fig1, ax1 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax1[ii, jj].imshow(np.reshape(eface01[:, flag], (16, 16)), cmap='gray')
        ax1[ii, jj].xaxis.set_visible(False)
        ax1[ii, jj].yaxis.set_visible(False)
        ax1[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig1.tight_layout(pad=0.5)

fig2, ax2 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax2[ii, jj].imshow(np.reshape(eface02[:, flag], (16, 16)), cmap='gray')
        ax2[ii, jj].xaxis.set_visible(False)
        ax2[ii, jj].yaxis.set_visible(False)
        ax2[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig2.tight_layout(pad=0.5)

plt.show()


# In[31]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the images
datadir = 'yalefaces/'
with_svd = False

# Function to perform downsampling
def downsample_image(image):
    return np.asarray(image)[::4, ::4].reshape(1, -1)

# Load images and perform downsampling
s01 = np.empty((0, 16*16))
s02 = np.empty((0, 16*16))

# Determine the target image size
target_size = 16 * 16

for f in os.listdir(datadir):
    if f.endswith('.gif'):
        im = downsample_image(Image.open(datadir + f))

        # Check and resize images to the target size
        if im.shape[1] != target_size:
            im = im[:, :target_size]

        if f.startswith('subject01'):
            if f.endswith('test.gif'):
                s01_test = im
            else:
                s01 = np.row_stack((s01, im))
        elif f.startswith('subject02'):
            if f.endswith('test.gif'):
                s02_test = im
            else:
                s02 = np.row_stack((s02, im))

# Perform PCA analysis
mu01 = np.mean(s01, axis=0)
mu02 = np.mean(s02, axis=0)

x01 = s01 - np.tile(mu01, (s01.shape[0], 1))
x02 = s02 - np.tile(mu02, (s02.shape[0], 1))

if with_svd:
    # SVD on the data matrix
    u01, _, _ = np.linalg.svd(x01.T)
    u02, _, _ = np.linalg.svd(x02.T)
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]
else:
    a01 = np.cov(x01, rowvar=False)
    a02 = np.cov(x02, rowvar=False)
    _, evec01 = np.linalg.eigh(a01)
    _, evec02 = np.linalg.eigh(a02)
    u01 = np.real(evec01[:, ::-1])
    u02 = np.real(evec02[:, ::-1])
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]

# Definition of num_eface_batch
num_eface_batch = [1, 2, 3, 4, 5, 6]

# Function to calculate residual score
def residual_score(x, ef):
    if len(ef.shape) == 1:
        ef = ef.reshape(-1, 1)
    residual = x - ef @ (ef.T @ x)
    return np.linalg.norm(residual)**2

# Perform face recognition task
print('\nWith mean subtracted:')
for num_eface in num_eface_batch:
    s11 = residual_score((s01_test - mu01).T, eface01[:, :num_eface])
    s12 = residual_score((s02_test - mu01).T, eface01[:, :num_eface])
    s21 = residual_score((s01_test - mu02).T, eface02[:, :num_eface])
    s22 = residual_score((s02_test - mu02).T, eface02[:, :num_eface])

    print(f'\nThe residual score with {num_eface} eigenfaces included:')
    print('s11:', format(s11, "5.4e"))
    print('s12:', format(s12, "5.4e"))
    print('s21:', format(s21, "5.4e"))
    print('s22:', format(s22, "5.4e"))


# As the number of eigenfaces increases, the reconstruction errors decrease, leading to better face recognition.
# Subject 2 generally has lower reconstruction errors than Subject 1, indicating better recognition and potentially more distinctive facial features.
# The decreasing trend in reconstruction errors suggests that the selected eigenfaces capture important features for recognition.
# 
# in conclusion, the choice of eigenfaces significantly influences the face recognition task. Lower residual scores, especially for Subject 2, indicate that the selected eigenfaces are effective in representing and recognizing facial features. The results demonstrate the power of PCA in extracting relevant information for face recognition.

# In[36]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the images
datadir = 'yalefaces/'
with_svd = False

# Function to perform downsampling
def downsample_image(image, target_size):
    return np.asarray(image.resize(target_size, Image.ANTIALIAS)).reshape(1, -1)

# Load images and perform downsampling
s01 = np.empty((0, 16*16))
s02 = np.empty((0, 16*16))

# Determine the target image size
target_size = (16, 16)

for f in os.listdir(datadir):
    if f.endswith('.gif'):
        im = Image.open(datadir + f)

        # Check and resize images to the target size
        im = downsample_image(im, target_size)

        if f.startswith('subject01'):
            if f.endswith('test.gif'):
                s01_test = im
            else:
                s01 = np.vstack((s01, im))
        elif f.startswith('subject02'):
            if f.endswith('test.gif'):
                s02_test = im
            else:
                s02 = np.vstack((s02, im))

# Perform PCA analysis
mu01 = np.mean(s01, axis=0)
mu02 = np.mean(s02, axis=0)

x01 = s01 - np.tile(mu01, (s01.shape[0], 1))
x02 = s02 - np.tile(mu02, (s02.shape[0], 1))

if with_svd:
    # SVD on the data matrix
    u01, _, _ = np.linalg.svd(x01.T)
    u02, _, _ = np.linalg.svd(x02.T)
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]
else:
    a01 = np.cov(x01, rowvar=False)
    a02 = np.cov(x02, rowvar=False)
    _, evec01 = np.linalg.eigh(a01)
    _, evec02 = np.linalg.eigh(a02)
    u01 = np.real(evec01[:, ::-1])
    u02 = np.real(evec02[:, ::-1])
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]

# Reshape and visualize eigenfaces for Subject 1
fig1, ax1 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax1[ii, jj].imshow(np.reshape(eface01[:, flag], target_size), cmap='gray')
        ax1[ii, jj].xaxis.set_visible(False)
        ax1[ii, jj].yaxis.set_visible(False)
        ax1[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig1.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 1')

# Reshape and visualize eigenfaces for Subject 2
fig2, ax2 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax2[ii, jj].imshow(np.reshape(eface02[:, flag], target_size), cmap='gray')
        ax2[ii, jj].xaxis.set_visible(False)
        ax2[ii, jj].yaxis.set_visible(False)
        ax2[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig2.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 2')

plt.show()


# In[42]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Read in the images
datadir = 'yalefaces/'
with_svd = False

# Function to perform downsampling
def downsample_image(image, target_size):
    return np.mean(image.reshape(target_size[0], -1, target_size[1]), axis=(1, 2)).reshape(1, -1)
# Load images and perform downsampling
s01 = np.empty((0, 16*16))
s02 = np.empty((0, 16*16))

# Determine the target image size
target_size = (16, 16)

for f in os.listdir(datadir):
    if f.endswith('.gif'):
        im = np.asarray(Image.open(datadir + f))

        # Check and resize images to the target size
        im = downsample_image(im, target_size)

        if f.startswith('subject01'):
            if f.endswith('test.gif'):
                s01_test = im
            else:
                s01 = np.vstack((s01, im))
        elif f.startswith('subject02'):
            if f.endswith('test.gif'):
                s02_test = im
            else:
                s02 = np.vstack((s02, im))

# Perform PCA analysis
mu01 = np.mean(s01, axis=0)
mu02 = np.mean(s02, axis=0)

x01 = s01 - np.tile(mu01, (s01.shape[0], 1))
x02 = s02 - np.tile(mu02, (s02.shape[0], 1))

if with_svd:
    # SVD on the data matrix
    u01, _, _ = np.linalg.svd(x01.T)
    u02, _, _ = np.linalg.svd(x02.T)
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]
else:
    a01 = np.cov(x01, rowvar=False)
    a02 = np.cov(x02, rowvar=False)
    _, evec01 = np.linalg.eigh(a01)
    _, evec02 = np.linalg.eigh(a02)
    u01 = np.real(evec01[:, ::-1])
    u02 = np.real(evec02[:, ::-1])
    eface01 = u01[:, :6]
    eface02 = u02[:, :6]

# Reshape and visualize eigenfaces for Subject 1
fig1, ax1 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax1[ii, jj].imshow(np.reshape(eface01[:, flag], target_size), cmap='gray')
        ax1[ii, jj].xaxis.set_visible(False)
        ax1[ii, jj].yaxis.set_visible(False)
        ax1[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig1.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 1')

# Reshape and visualize eigenfaces for Subject 2
fig2, ax2 = plt.subplots(2, 3)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax2[ii, jj].imshow(np.reshape(eface02[:, flag], target_size), cmap='gray')
        ax2[ii, jj].xaxis.set_visible(False)
        ax2[ii, jj].yaxis.set_visible(False)
        ax2[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig2.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 2')

plt.show()


# In[46]:


pip install scikit-image


# In[54]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from skimage import util as sm
#from skimage.measure import block_reduce
from skimage.util import view_as_blocks

def flat_image(image_path):
    # Open the image from the working directory
    image = Image.open(os.path.abspath(image_path))
    image = np.array(image)
    
    # Determine the block shape based on the target size
    target_size = (61, 80)
    block_shape = (image.shape[0] // target_size[0], image.shape[1] // target_size[1])

    # Use view_as_blocks to create non-overlapping blocks
    blocks = view_as_blocks(image, block_shape)

    # Compute the mean value in each block
    image = np.mean(blocks, axis=(2, 3))

    # Flatten the resulting image
    image = image.flatten()

    return image
    
    # Downsampling by a factor of 4
    #image = sm.block_reduce(image, (4, 4), np.median)
    #from scipy.ndimage import block_reduce
    #image = block_reduce(image, (4, 4), np.median)
    #blocks = view_as_blocks(image, (4, 4))
    #image = np.median(blocks, axis=(2, 3))
    

    #image = image.flatten()
    
    #return image

def create_image_matrix(image_paths):
    image_list = []
    for image_path in image_paths:
        image_list.append(flat_image(image_path))
    images = tuple(image_list)
    subject = np.vstack(images)
    return subject

# Define image paths for Subject 1 and Subject 2
image1_paths = ["yalefaces/subject01.glasses.gif", "yalefaces/subject01.happy.gif",
                "yalefaces/subject01.leftlight.gif", "yalefaces/subject01.noglasses.gif",
                "yalefaces/subject01.normal.gif", "yalefaces/subject01.rightlight.gif",
                "yalefaces/subject01.sad.gif", "yalefaces/subject01.sleepy.gif",
                "yalefaces/subject01.surprised.gif", "yalefaces/subject01.wink.gif"]

image2_paths = ["yalefaces/subject02.glasses.gif", "yalefaces/subject02.happy.gif",
                "yalefaces/subject02.leftlight.gif", "yalefaces/subject02.noglasses.gif",
                "yalefaces/subject02.normal.gif", "yalefaces/subject02.rightlight.gif",
                "yalefaces/subject02.sad.gif", "yalefaces/subject02.sleepy.gif",
                "yalefaces/subject02.wink.gif"]

# Create image matrices for Subject 1 and Subject 2
subject_1 = create_image_matrix(image1_paths)
subject_2 = create_image_matrix(image2_paths)

# Checking the performance of the PCA components for Subject 1
pca_full = PCA(n_components=10)
pca_full.fit(subject_1)

plt.grid()
plt.plot(np.cumsum(pca_full.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

# Reshape and visualize eigenfaces for Subject 1
fig1, ax1 = plt.subplots(2, 3, figsize=(10, 7))
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax1[ii, jj].imshow(np.reshape(eface01[:, flag], (16, 16)), cmap='gray')
        ax1[ii, jj].xaxis.set_visible(False)
        ax1[ii, jj].yaxis.set_visible(False)
        ax1[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig1.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 1')

# Reshape and visualize eigenfaces for Subject 2
fig2, ax2 = plt.subplots(2, 3, figsize=(10, 7))
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        ax2[ii, jj].imshow(np.reshape(eface02[:, flag], (16, 16)), cmap='gray')
        ax2[ii, jj].xaxis.set_visible(False)
        ax2[ii, jj].yaxis.set_visible(False)
        ax2[ii, jj].set_title('eigenface: ' + str(flag + 1), fontsize=8)
fig2.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 2')

plt.show()


# Getting the first 6 eigenfaces for Subject 1
pca_1 = PCA(n_components=6)
pca_1.fit(subject_1)
eigenfaces = pca_1.components_[:6]

plt.figure()
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        axes[ii, jj].imshow(eigenfaces[flag].reshape(61, 80), cmap='gray')
        axes[ii, jj].xaxis.set_visible(False)
        axes[ii, jj].yaxis.set_visible(False)
        axes[ii, jj].set_title('Eigenface: ' + str(flag + 1), fontsize=8)
fig.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 1')
plt.show()

# Checking the performance of the PCA components for Subject 2
pca_full2 = PCA(n_components=9)
pca_full2.fit(subject_2)

plt.figure()
plt.grid()
plt.plot(np.cumsum(pca_full2.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

# Getting the first 6 eigenfaces for Subject 2
pca_2 = PCA(n_components=6)
pca_2.fit(subject_2)
eigenfaces2 = pca_2.components_[:6]

plt.figure()
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
for ii in range(2):
    for jj in range(3):
        flag = ii * 3 + jj
        axes[ii, jj].imshow(eigenfaces2[flag].reshape(61, 80), cmap='gray')
        axes[ii, jj].xaxis.set_visible(False)
        axes[ii, jj].yaxis.set_visible(False)
        axes[ii, jj].set_title('Eigenface: ' + str(flag + 1), fontsize=8)
fig.tight_layout(pad=0.5)
plt.suptitle('Eigenfaces for Subject 2')
plt.show()

# Reducing dimensionality for Subject 2
pca_2 = PCA(n_components=7)
sub_2_reduced = pca_2.fit_transform(subject_2)
sub_2_recovered = pca_2.inverse_transform(sub_2_reduced)

# Displaying first 6 compressed faces for Subject 2
plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sub_2_recovered[i, :].reshape([61, 80]), cmap='gray')
    plt.title(f'Compressed image Subject02 {i + 1}', fontsize=15, pad=15)
plt.show()


# In[4]:


from skimage.measure import block_reduce


# In[5]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from skimage import util as sm
#from skimage.measure import block_reduce
from skimage.util import view_as_blocks

def flat_image(image_path):
    # Open the image from the working directory
    image = Image.open(os.path.abspath(image_path))
    image = np.array(image)
    
    #downsampling by a factor of 4
    #image = sm.block_reduce(image, (4,4), np.median)
    image = block_reduce(image, (4, 4), np.median)
    image = image.flatten()
    return image



def create_image_matrix(image_paths):
    image_list = []
    for image_path in image_paths:
        image_list.append(flat_image(image_path))
    images = tuple(image_list)
    subject = np.vstack(images)
    return subject

# Define image paths for Subject 1 and Subject 2
image1_paths = ["yalefaces/subject01.glasses.gif", "yalefaces/subject01.happy.gif",
                "yalefaces/subject01.leftlight.gif", "yalefaces/subject01.noglasses.gif",
                "yalefaces/subject01.normal.gif", "yalefaces/subject01.rightlight.gif",
                "yalefaces/subject01.sad.gif", "yalefaces/subject01.sleepy.gif",
                "yalefaces/subject01.surprised.gif", "yalefaces/subject01.wink.gif"]

image2_paths = ["yalefaces/subject02.glasses.gif", "yalefaces/subject02.happy.gif",
                "yalefaces/subject02.leftlight.gif", "yalefaces/subject02.noglasses.gif",
                "yalefaces/subject02.normal.gif", "yalefaces/subject02.rightlight.gif",
                "yalefaces/subject02.sad.gif", "yalefaces/subject02.sleepy.gif",
                "yalefaces/subject02.wink.gif"]

# Create image matrices for Subject 1 and Subject 2
subject_1 = create_image_matrix(image1_paths)
subject_2 = create_image_matrix(image2_paths)


# Checking the performance of the PCA components for Subject 1
pca_full = PCA(n_components=10)
pca_full.fit(subject_1)

plt.grid()
plt.plot(np.cumsum(pca_full.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

#Getting the first 6 eigenfaces for subject 1
pca_1 = PCA(n_components= 6)
pca_1.fit(subject_1)
eigenfaces = pca_1.components_[:6]
plt.figure()
fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes[0][0].imshow(eigenfaces[0].reshape(61,80))
axes[0][1].imshow(eigenfaces[1].reshape(61,80))
axes[0][2].imshow(eigenfaces[2].reshape(61,80))
axes[1][0].imshow(eigenfaces[3].reshape(61,80))
axes[1][1].imshow(eigenfaces[4].reshape(61,80))
axes[1][2].imshow(eigenfaces[5].reshape(61,80))
plt.show()
sub_1_reduced = pca_1.fit_transform(subject_1)
sub_1_recovered = pca_1.inverse_transform(sub_1_reduced)

#First 6 faces for Subject 1
plt.figure()
image_11 = sub_1_recovered[0,:].reshape([61,80])
plt.imshow(image_11)
plt.title('Compressed image Subject01 Glasses', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject01 Glasses.png')

plt.figure()
image_12 = sub_1_recovered[1,:].reshape([61,80])
plt.imshow(image_12)
plt.title('Compressed image Subject01 Happy', fontsize=15, pad=15)
plt.savefig('Compressed image Subject01 Happy.png')

plt.figure()
image_13 = sub_1_recovered[2,:].reshape([61,80])
plt.imshow(image_13)
plt.title('Compressed image Subject01 Leftlight', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject01 Leftlight.png')

plt.figure()
image_14 = sub_1_recovered[3,:].reshape([61,80])
plt.imshow(image_14)
plt.title('Compressed image Subject01 No Glasses', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject01 No Glasses.png')

plt.figure()
image_15 = sub_1_recovered[4,:].reshape([61,80])
plt.imshow(image_15)
plt.title('Compressed image Subject01 Normal', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject01 Normal.png')

plt.figure()
image_16 = sub_1_recovered[5,:].reshape([61,80])
plt.imshow(image_16)
plt.title('Compressed image Subject01 Rightlight', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject01 Rightlight.png')

#Checking to see the performance of the PCA components for subjact 2
pca_full2 = PCA(n_components=9)
pca_full2.fit(subject_2)

plt.figure()
plt.grid()
plt.plot(np.cumsum(pca_full2.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

#Getting the first 6 eigenfaces for subject 1
pca_2 = PCA(n_components= 6)
pca_2.fit(subject_2)
eigenfaces2 = pca_2.components_[:6]
plt.figure()
fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes[0][0].imshow(eigenfaces2[0].reshape(61,80))
axes[0][1].imshow(eigenfaces2[1].reshape(61,80))
axes[0][2].imshow(eigenfaces2[2].reshape(61,80))
axes[1][0].imshow(eigenfaces2[3].reshape(61,80))
axes[1][1].imshow(eigenfaces2[4].reshape(61,80))
axes[1][2].imshow(eigenfaces2[5].reshape(61,80))
plt.show()

pca_2 = PCA(n_components= 7)
sub_2_reduced = pca_2.fit_transform(subject_2)
sub_2_recovered = pca_2.inverse_transform(sub_2_reduced)

#First 6 faces for Subject 2
plt.figure()
image_21 = sub_2_recovered[0,:].reshape([61,80])
plt.imshow(image_21)
plt.title('Compressed image Subject02 Glasses', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 Glasses.png')

plt.figure()
image_22 = sub_2_recovered[1,:].reshape([61,80])
plt.imshow(image_22)
plt.title('Compressed image Subject02 Happy', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 Happy.png')

plt.figure()
image_23 = sub_2_recovered[2,:].reshape([61,80])
plt.imshow(image_23)
plt.title('Compressed image Subject02 Leftlight', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 Leftlight.png')

plt.figure()
image_24 = sub_2_recovered[3,:].reshape([61,80])
plt.imshow(image_24)
plt.title('Compressed image Subject02 No Glasses', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 No Glasses.png')

plt.figure()
image_25 = sub_2_recovered[4,:].reshape([61,80])
plt.imshow(image_25)
plt.title('Compressed image Subject02 Normal', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 GNormal.png')

plt.figure()
image_26 = sub_2_recovered[5,:].reshape([61,80])
plt.imshow(image_26)
plt.title('Compressed image Subject02 Rightlight', fontsize=15, pad=15)
#plt.savefig('Compressed image Subject02 Right.png')


# In[6]:


test_1_list = ["yalefaces/subject01-test.gif"]
test_2_list = ["yalefaces/subject02-test.gif"]

test_1 = create_image_matrix(test_1_list)
test_2 = create_image_matrix(test_2_list)
test = [test_1,test_2]

eg_face_1 = eigenfaces[0].reshape(-1,1)
eg_face_2 = eigenfaces2[0].reshape(-1,1)
eg_face = [eg_face_1,eg_face_2]

S = np.zeros((2,2))

for i in range(2):
    for j in range(2):
        a = test[j]
        b = eg_face[i] * eg_face[i].T
        c = b * a
        S[i][j] = np.linalg.norm(a - c)**2
print(S)


# In[9]:


test_1_list = ["yalefaces/subject01-test.gif"]
test_2_list = ["yalefaces/subject02-test.gif"]

# Create image matrices for the test images
test_1 = create_image_matrix(test_1_list)
test_2 = create_image_matrix(test_2_list)

# Define the eigenfaces for Subject 1 and Subject 2
eg_face_1 = eigenfaces_subject1[0].reshape(-1, 1)
eg_face_2 = eigenfaces_subject2[0].reshape(-1, 1)

# Create a list of eigenfaces
eg_faces = [eg_face_1, eg_face_2]

# Initialize a matrix to store the scores
S = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        # Compute the residual score
        projection = np.dot(test[j], eg_faces[i])
        reconstruction = np.dot(projection, eg_faces[i].T)
        residual = np.linalg.norm(test[j] - reconstruction) ** 2
        S[i][j] = residual

print("Residual scores:")
print(S)


# In[10]:


test_1_list = ["yalefaces/subject01-test.gif"]
test_2_list = ["yalefaces/subject02-test.gif"]

# Create image matrices for the test images
test_1 = create_image_matrix(test_1_list)
test_2 = create_image_matrix(test_2_list)

# Define the eigenfaces for Subject 1 and Subject 2
eg_face_1 = eigenfaces_subject1[0].reshape(-1, 1)
eg_face_2 = eigenfaces_subject2[0].reshape(-1, 1)

# Create a list of eigenfaces
eg_faces = [eg_face_1, eg_face_2]

# Initialize a matrix to store the scores
S = np.zeros((2, 2))

# Compute the mean of eigenface subject 1 and subject 2
mean_subject1 = np.mean(subject_1, axis=0)
mean_subject2 = np.mean(subject_2, axis=0)

for i in range(2):
    for j in range(2):
        # Subtract the mean of the eigenface subject from the test photo
        test_j = test_1 if j == 0 else test_2
        mean_subject = mean_subject1 if j == 0 else mean_subject2
        test_j_centered = test_j - mean_subject
        
        # Compute the residual score
        projection = np.dot(test_j_centered, eg_faces[i])
        reconstruction = np.dot(projection, eg_faces[i].T)
        residual = np.linalg.norm(test_j_centered - reconstruction) ** 2
        S[i][j] = residual

print("Residual scores:")
print(S)


# In[ ]:




