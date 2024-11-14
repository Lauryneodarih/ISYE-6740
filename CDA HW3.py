#!/usr/bin/env python
# coding: utf-8

# #Question 1.1
# Advantages of KDE over histograms:
# 1. Flexibility: KDEs offer greater flexibility due to the ability to dynamically adjust parameters such as bandwidth and kernel shape and size. This adaptability makes them more suitable for cases where the underlying distributions are largely unknown, allowing for better exploration of data properties.
# 2. Information preservation: KDEs tend to retain more information compared to histograms since they don't involve binning, thereby minimizing loss of data and providing a clearer representation of the distribution.
# 
# Advantages of histograms over KDE:
# 1. Computational efficiency: Histograms require less computational power compared to KDEs, making them more efficient, especially for large datasets. This efficiency can be crucial in scenarios where computational resources are limited or time constraints are present.
# 2. Simplicity and transparency: Histograms are simpler to understand and visualize, making them more accessible to a wider audience, including those without extensive statistical expertise. Their straightforward representation of data distribution allows for easier interpretation and communication of findings.

# #question 1.2
# 
# 
# The inclusion of multiple components complicates the analytical tractability of the likelihood function. Additionally, Bayesian methods face limitations due to the exponential increase in terms within the expanded likelihood, rendering them impractical. Moreover, the presence of multiple modes in the likelihood poses challenges for straightforward numerical maximization or the application of simpler Monte Carlo algorithms like Gibbs sampling or Metropolis-Hastings.
# 
# The Expectation-Maximization (EM) algorithm is commonly used for estimating parameters in Gaussian Mixture Models (GMMs). It operates in two main steps:
# 
# a) Expectation step (E-step): Given initial parameter estimates, the algorithm calculates the expected values of latent variables (cluster assignments) using the current model parameters. This step involves computing the probabilities of data points belonging to each cluster.
# 
# b) Maximization step (M-step): Using the expected values obtained in the E-step, the algorithm updates the model parameters to maximize the expected log-likelihood. Parameters updated include cluster means, covariances, and mixture weights.
# 
# The algorithm iterates between the E-step and M-step until convergence, typically determined by predefined criteria such as small changes in parameter estimates or reaching a maximum number of iterations.
# 
# The EM algorithm is well-suited for GMM estimation due to its ability to handle non-convex likelihood functions and model complexity. It iteratively improves parameter estimates and provides a framework for incorporating prior information or regularization techniques to enhance estimation performance.

# #Question 1.3
# 
# 
# The Expectation-Maximization (EM) algorithm for Gaussian mixtures is an iterative approach that begins with initial parameter values and progresses towards convergence, which may result in either a global or local minimum. It iteratively updates the parameter set Θ until convergence criteria are met. The algorithm consists of alternating E-step and M-step iterations.
# 
# In the E-step, the algorithm computes the posterior probabilities or weights for each component of the Gaussian mixture model given the observed data points. These weights represent the uncertainty associated with assigning each data point to a particular component. The posterior probabilities can be easily computed for individual samples using Bayes' rule, which states:
# 
# $$P(\text{Component } k | \text{Data point } X_i) = \frac{P(X_i | \text{Component } k) \times \text{Weight of component } k}{\sum_{j=1}^{K} P(X_i | \text{Component } j) \times \text{Weight of component } j} $$
# 
# where $$ P(\text{Component } k | \text{Data point } X_i) $$ represents the posterior probability of data point $$ X_i $$ belonging to component $$ k $$, $$ P(X_i | \text{Component } k) $$ is the likelihood of data point $$ X_i $$ given component $$ k $$, and $$ \text{Weight of component } k $$ is the weight of component $$ k $$ in the mixture.
# 
# The M-step involves updating the parameters of the Gaussian mixture model based on the computed posterior probabilities. This step typically includes updating the means, covariances, and mixture weights to maximize the expected log-likelihood of the observed data.
# 
# Overall, the EM algorithm for Gaussian mixtures provides a robust framework for estimating model parameters and capturing the uncertainty associated with component assignments for each data point.

# # Question 1.4
# 
# 
# 
# Some common choices of kernel function:
# - the normal distribution: N(0,1)
# - the uniform rectangle: =0.5 for |x|<1, else =0
# 
# -the Epanechnikov kernel: =¾(1-x2) for |x|<1, else =0 It can be shown that the Epanechnikov kernel is “optimal” in 1D in a particular sense, but on the other hand it turns out that it makes very little difference which kernel you use. Many people like to use normal (Gaussian) distributions for simplicity.
# 
# Choosing an appropriate bandwidth is the key. Of course to do this perfectly you’d need to know what the underlying distribution is.
# 
# The basic principles are similar to those behind choosing the best binning for a histogram. Narrow bandwidth: allows you to sample narrow features of the distribution, but leaves you susceptible to random scatter. Generally the more points you have, the narrower you can make the bandwidth.
# 
# Wide bandwidth: smooths out statistical fluctuations, but may bias the result by smearing out narrower features.
# 
# Choosing the kernel bandwidth for Kernel Density Estimation (KDE) is crucial for balancing bias and variance in estimating the probability density function. Several methods exist for selecting the bandwidth:
# 
# 1. Rule of Thumb: Utilize rules like Silverman's or Scott's, which estimate bandwidth based on sample standard deviation and data points' count.
# 
# 2. Cross-Validation: Employ techniques like leave-one-out or k-fold cross-validation to evaluate bandwidths and select the one minimizing a chosen criterion, such as log-likelihood.
# 
# 3. Sheather-Jones Method: Improve bandwidth estimation by considering the kurtosis of the data distribution, offering robustness for non-Gaussian distributions.
# 
# 4. Maximum Likelihood Estimation: Directly optimize the bandwidth to fit the data distribution by maximizing the likelihood function.
# 
# 5. Grid Search: Perform an exhaustive search over bandwidth values and select the one maximizing a chosen criterion, such as likelihood or cross-validation score.
# 
# 6. Adaptive Bandwidth: Allow bandwidth to vary across data space regions based on local characteristics, providing more accurate density estimates, especially in regions with varying data density.
# 
# The selection method depends on data characteristics, desired smoothness, and computational considerations, with comparison between methods recommended for robust and reliable KDE estimation.
# 

# In[4]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[4]:


#Question 2a
import csv
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn import preprocessing

#part a
data =pd.read_csv("n90pol.csv")
data =pd.DataFrame(data)

fig, axes =plt.subplots(1, 2)
#plot histograms
data.hist('acc', bins=20, ax=axes[0],edgecolor ='black')
data.hist('amygdala', bins=20, ax=axes[1],facecolor ='orange',edgecolor ='black')
plt.savefig('Q2a_Histogram.png')
plt.show()

acc =data['acc']
amygdala =data['amygdala']
#plot kde 
df =data[['acc','amygdala']]
ax =df.plot.kde()
plt.savefig('Q2a_KDE.png')
plt.show()


# In[5]:


#Question 2b
import seaborn as sns
from scipy.stats import spearmanr 
from scipy import stats

# Create a 2-dimensional histogram for amygdala and acc
plt.figure(figsize=(8, 6))
plt.hist2d(amygdala, acc, bins=20, cmap='Blues')
plt.colorbar(label='Frequency')
plt.xlabel('Amygdala')
plt.ylabel('ACC')
plt.title('2D Histogram: Amygdala vs. ACC')
plt.savefig('Q2b_Histogram.png')
plt.show()


#Bivariate 2D KDE plot 
sns.kdeplot(data['acc'],data['amygdala'])
plt.title("2D KDE")
plt.savefig('Q2b_KDE.png')
plt.show()


# After visually inspecting the data, it appears that the dominant values or ranges are concentrated, and the values seem to align towards one peak. I would argue that this distribution is unimodal, with a few outliers. For part C, we need to test for independence between acc and amygdala. This can be determined by checking whether the joint probability  $$p(\text{amygdala}, \text{acc})$$ is equal to the product of the marginal probabilities $$ p(\text{amygdala}) $$ and  $$p(\text{acc}) $$. If $$ p(\text{amygdala}, \text{acc}) = p(\text{amygdala}) \times p(\text{acc}) $$, it suggests independence between the variables.
# 

# In[6]:


#Question 2c

from sklearn.neighbors import KernelDensity

acc_np =np.array(data['acc'])
amygdala_np =np.array(data['amygdala'])

stacked_data = np.vstack([amygdala_np,acc_np])
xmin, xmax = np.min(amygdala_np), np.max(amygdala_np)
ymin, ymax = np.min(acc_np), np.max(acc_np)

X, Y =np.mgrid[2*xmin:2*xmax:200j, 2*ymin:2*ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])

x_d =X[:,1]
x =amygdala_np
kde =KernelDensity(bandwidth=0.01, kernel='gaussian')
kde.fit(x[:, None])
logprob =kde.score_samples(x_d[:, None])
amygdala_kde =np.exp(logprob)

## find kde of acc
y_d =Y[0]
y =acc_np
kde =KernelDensity(bandwidth=0.01, kernel='gaussian')
kde.fit(y[:, None])
logprob =kde.score_samples(y_d[:, None])
acc_kde =np.exp(logprob)

### find the products of the marginal distributions
product =amygdala_kde[:,None] *acc_kde[None,:]

### find the joint kde

kde =KernelDensity(bandwidth=0.01, kernel='gaussian')
kde.fit(df)
logprob =kde.score_samples(positions.T)
joint =np.reshape(np.exp(logprob), X.shape)

#plot results
plt.imshow(joint, cmap='cividis')
plt.colorbar()
plt.title("Joint Kernel Distribution")
plt.savefig('Q2C_Joint.png')
plt.show()

plt.imshow(product, cmap='cividis')
plt.colorbar()
plt.title("Product")
plt.savefig('Q2C_Product.png')
plt.show()

plt.imshow(np.abs(joint -product), cmap='cividis')
plt.colorbar()
plt.title("Error")
plt.savefig('Q2C_Error.png')
plt.show()


# The generated plots illustrate the product of the 1D marginal distributions, the joint distribution, and their disparity. These visualizations reveal that the difference between the product of the marginal distributions and the joint distribution doesn't consistently equate to zero.
# 
# To ascertain the independence of *amygdala* and *acc*, we rely on the following principle: 
# 
# f(X,Y)(x, y) = f(X)(x) * f(Y)(y) 
# 
# Where f(X)(x) and f(Y)(y) represent the density functions estimated using 1D Kernel Density Estimation (KDE) for *amygdala* and *acc* respectively, and f(X,Y)(x, y) denotes the joint probability distribution function estimated via 2D KDE. It's crucial to note that this property should hold true across the entirety of the domain.
# 
# Notably, at certain x and y values, this disparity is notably substantial. Consequently, we can infer that our two variables are not independent.

# In[8]:


# Question 2d
X_plot =np.linspace(-0.1, 0.1, 1000)[:, np.newaxis]
acc_lst =[]
amy_lst =[]

for i in range(2,6):
    print("\nOrientation: " +str(i))
    
    orient =data[data['orientation'] ==i]
    acc =np.array(orient['acc']).reshape(-1, 1)
    amygdala =np.array(orient['amygdala']).reshape(-1, 1)
    
    kde_acc =KernelDensity(kernel="gaussian", bandwidth=0.01).fit(acc)
    kde_amy =KernelDensity(kernel="gaussian", bandwidth=0.01).fit(amygdala)
    
    fig, ax =plt.subplots(figsize=(6, 4))
    
    log_dens_acc =kde_acc.score_samples(X_plot)
    plt.plot(np.exp(log_dens_acc))
    plt.title('ACC_C' +str(i) +'.png')
    plt.savefig('ACC_C' +str(i) +'.png')
    plt.show()
    plt.close()
    
    log_dens_amy =kde_amy.score_samples(X_plot)
    plt.plot(np.exp(log_dens_amy))
    plt.title('AMY_C' +str(i) +'.png')
    plt.savefig('AMY_C' +str(i) +'.png')
    plt.show()
    plt.close()
    
    acc_mean =np.mean(acc)
    amy_mean =np.mean(amygdala)
    acc_lst.append(acc_mean)
    amy_lst.append(amy_mean)
    
    print("acc_mean = " +str(acc_mean))
    print("amy_mean = " +str(amy_mean))
    
df =pd.DataFrame(list(zip(acc_lst,amy_lst)),columns =["ACC","Amygdala"],index=['2', '3', '4','5'])
display(df)


# From the KDE graphs, it's evident that there exists a correlation between brain structure and political views. The distribution of the amygdala varies depending on political orientation.
# 
# 
# Amygdala Analysis:
# - The mean values of each conditional sample are depicted above in the table and also indicated in each KDE plot with a dotted line.
# - Across different orientations, the relationship concerning amygdala varies significantly. Some distributions are right-skewed, while others are left-skewed. This suggests that the influence of the brain structure amygdala differs among various orientations.
# - Notably, for the orientation labeled as "very liberal" (Orientation 5), the mean value is negative, in contrast to the positive means observed for all other orientations.
# 
# ACC Analysis:
# - Similar to the analysis for the amygdala, the situation for ACC shows variability across different orientations.
# - The mean values of ACC are negative for the "very conservative" orientation (Orientation 2), indicating a different impact of ACC across various political orientations.
#    
# this implies that the brain structures of amygdala and ACC have differing impacts on individuals across the political spectrum. The varying means across different orientations suggest that political ideology may indeed be related to differences in brain structure and functioning. For instance, the negative mean for ACC in the "very conservative" orientation suggests a distinct cognitive processing pattern compared to other orientations. These findings shed light on the complex interplay between brain structure and political views.

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity

# Define the range for the grid
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.1, 0.1
X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

# Initialize lists to store KDE results
kde_results = []

# Iterate through each orientation
for orientation_value in range(2, 6):
    print("\nOrientation: " + str(orientation_value))
    
    # Filter data for the current orientation
    orientation_data = data[data['orientation'] == orientation_value]
    acc = np.array(orientation_data['acc']).reshape(-1, 1)
    amygdala = np.array(orientation_data['amygdala']).reshape(-1, 1)
    
    # Fit the two-dimensional KDE
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    stacked_data = np.column_stack((acc, amygdala))
    kde.fit(stacked_data)
    
    # Evaluate the KDE on the grid
    log_density = kde.score_samples(positions.T)
    density = np.exp(log_density).reshape(X.shape)
    
    # Append the results to the list
    kde_results.append(density)
    
    # Plot the two-dimensional KDE (e.g., heatmap or contour plot)
    plt.imshow(density, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('C= ' + str(orientation_value))
    plt.xlabel('ACC')
    plt.ylabel('Amygdala')
    plt.show()



# The differences observed in the conditional joint distributions indicate variations in brain structure corresponding to political views.
# 

# In[5]:


# Question 3a
#loading libraries

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.io
import pandas as pd
from sklearn import preprocessing
from scipy.stats import multivariate_normal as mvn
import scipy.sparse.linalg as ll
from scipy import ndimage
import seaborn as sns


# In[7]:



data = scipy.io.loadmat('data.mat')['data']
label = scipy.io.loadmat('label.mat')['trueLabel']

data = np.array(data).T
label = np.array(label)
mu_original = np.mean(data, axis=0, keepdims=True)
ndata = data - mu_original

m, n = data.shape

C = np.matmul(data.T, data) / m

# PCA the data
d = 5 
V, Sig, _ = np.linalg.svd(C)
V = V[:, :d]
Sig = np.diag(Sig[:d])

# Project the data to the top 5 principal directions
pdata = np.dot(ndata, V)

K = 2
seed = 5

# Initialize prior
np.random.seed(seed)
pi = np.random.random(K)
pi = pi / np.sum(pi)

# Initial mean and covariance
mu = np.random.randn(K, d)
mu_old = mu.copy()

sigma = []
for ii in range(K):
    # To ensure the covariance is positive semi-definite
    seed = 1 if ii == 0 else 4
    np.random.seed(seed)
    dummy = np.random.randn(d, d)
    sigma.append(dummy @ dummy.T + np.eye(d))

# Initialize the posterior
tau = np.full((m, K), fill_value=0.)


# Initialization
maxIter = 100
tol = 1e-3

# Metrics for log-likelihood
log_likelihood = []
iter_count = []

plt.ion()

for ii in range(100):
    # E-step
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])

    # Normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (m, 1)
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))

    # M-step
    for kk in range(K):
        # Update prior
        pi[kk] = np.sum(tau[:, kk]) / m

        # Update component mean
        mu[kk] = pdata.T @ tau[:, kk] / np.sum(tau[:, kk], axis=0)

        # Update cov matrix
        dummy = pdata - np.tile(mu[kk], (m, 1))  # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:, kk]) @ dummy / np.sum(tau[:, kk], axis=0)

    # Metrics for graph
    log_likelihood.append(np.sum(np.log(sum_tau)))
    iter_count.append(ii)

    print('-----iteration--- ', ii)
    plt.scatter(pdata[:, 0], pdata[:, 1], c=tau[:, 0], cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("EM Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.draw()
    plt.pause(0.1)

    if np.linalg.norm(mu - mu_old) < tol:
        print('Training converged')
        break
    mu_old = mu.copy()

    if ii == 99:
        print('Max iteration reached')
        break

plt.plot(iter_count, log_likelihood, color='blue')
plt.title("Log-Likelihood Function versus Iterations")
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.savefig("Q3a_Log-Likelihood_function_versus_Iterations.png")
plt.show()


# The EM algorithm converged after 23 iterations

# In[4]:


# Question 3b
first_mean =(V @mu[0] +mu_original).reshape((28,28)).T
plt.imshow(first_mean, cmap ='gray')
plt.title('Mean of the First Component with Weight = ' +str(pi[0]))
plt.savefig('Q3b_Mu1')
plt.show()

second_mean =(V @mu[1] +mu_original).reshape((28,28)).T
plt.imshow(second_mean, cmap ='gray')
plt.title('Mean of the Second Component with Weight = ' +str(pi[1]))
plt.savefig('Q3b_Mu2')
plt.show()


# In[5]:


cov1 =sns.heatmap(sigma[0], cmap ='cividis', yticklabels=False)
plt.title('Covariance Matrix of the First Component')
fig =cov1.get_figure()
fig.savefig("Q3b_Cov1")


# In[6]:


cov2 =sns.heatmap(sigma[1], cmap ='cividis', yticklabels=False)
plt.title('Covariance Matrix of the Second Component')
fig =cov2.get_figure()
fig.savefig("Q3b_Cov2")

print("Weights of each Component",pi)
print("Mean of each compnent",mu)


# In[7]:


# Question 3C
from sklearn.cluster import KMeans
labels_em =np.argmax(tau,axis =1)
labels =label.T.flatten('F')

labels_em[labels_em ==0] =2
labels_em[labels_em ==1] =6

mismatch_count =np.sum(labels_em !=labels)
total_elements =len(labels_em)
mismatch_ratio_gmm =mismatch_count /total_elements


# In[8]:


np.random.seed(seed)
kmeans =KMeans(n_clusters =2).fit(data)
kmeans_labels =kmeans.labels_

kmeans_centers =kmeans.cluster_centers_

# Check cluster for mismatch calculation
center_0 =kmeans_centers[0].reshape(28,28)
center_1 =kmeans_centers[1].reshape(28,28)

center_0 =np.fliplr(center_0)
center_0 =ndimage.rotate(center_0, 90, reshape=False)
plt.imshow(center_0, cmap='gray')
fig.savefig("Q3c_KMeans1")


# In[9]:


center_1 =np.fliplr(center_1)
center_1 =ndimage.rotate(center_1, 90, reshape=False)
plt.imshow(center_1, cmap='gray')
fig.savefig("Q3c_KMeans2")


# In[17]:


# Replace labels based on corresponding digit: 0 = 2, 1 = 6
kmeans_labels[kmeans_labels ==0] =2
kmeans_labels[kmeans_labels ==1] =6

mismatch_count =np.sum(kmeans_labels !=labels)
total_elements =len(kmeans_labels)
mismatch_ratio_kmeans =mismatch_count /total_elements

print("Mismatch ratio for GMM:", mismatch_ratio_gmm)
print("Mismatch ratio for KMeans:", mismatch_ratio_kmeans)


# The lower mismatch ratio for the GMM suggests that it has identified clusters that are closer to the true labels compared to KMeans. which means that the GMM performed better than the KMeans

# #Question 4.1a
# 
# 
# Since the random variables \(Y(pr)\) and \(Z(pr)\) are independent, and $$\epsilon \sim N(0, \sigma^2)$$ is independent Gaussian noise, we have:
# $$ \text{Cov}(Y(pr), Z(pr)) = 0 $$
# $$ \text{Cov}(Y(pr), \epsilon) = 0 $$
# $$ \text{Cov}(Z(pr), \epsilon) = 0 $$
# 
# $$x(pr)$$ can be written as:
# $$ x(pr) = y(pr) + z(pr) + \epsilon $$
# 
# Using the properties of covariance and the fact that $$Y(pr)$$ and $$\epsilon$$ are independent:
# $$ \text{Cov}(Y(pr), X(pr)) = \text{Cov}(Y(pr), Y(pr)) + \text{Cov}(Y(pr), Z(pr)) + \text{Cov}(Y(pr), \epsilon) = \sigma^2_p $$
# 
# using the properties of covariance and the fact that $$Z(pr)$$ and $$\epsilon$$ are independent:
# $$ \text{Cov}(Z(pr), X(pr)) = \text{Cov}(Z(pr), Y(pr)) + \text{Cov}(Z(pr), Z(pr)) + \text{Cov}(Z(pr), \epsilon) = \tau^2_r $$
# 
# By the property of the sum of independent normally distributed random variables, we have:
# $$ X(pr) \sim N(\mu_p + \nu_r, \sigma^2 + \sigma^2_p + \tau^2_r) $$
# 
# The joint distribution of $$Y(pr)$$, $$Z(pr)$$, and $$X(pr)$$ follows a multivariate Gaussian distribution:
# $$ \begin{bmatrix} Y(pr) \\ Z(pr) \\ X(pr) \end{bmatrix} \sim N \left( \begin{bmatrix} \mu_p \\ \nu_r \\ \mu_p + \nu_r \end{bmatrix}, \begin{bmatrix} \sigma^2_p & 0 & \sigma^2_p \\ 0 & \tau^2_r & \tau^2_r \\ \sigma^2_p & \tau^2_r & \sigma^2_p + \tau^2_r + \sigma^2 \end{bmatrix} \right) $$
# 
# From the property of bivariate Gaussian distribution, the conditional mean is:
# $$\begin{bmatrix} \mu(pr)_1 \\ \mu(pr)_2 \end{bmatrix} = \text{E}[Y(pr), Z(pr) | X(pr)] = \begin{bmatrix} \mu_p \nu_r + \frac{\sigma^2_p \tau^2_r}{\sigma^2_p + \tau^2_r + \sigma^2} \cdot x(pr) - (\mu_p + \nu_r) \frac{\sigma^2_p}{\sigma^2_p + \tau^2_r + \sigma^2} \end{bmatrix} $$
# 
# The conditional covariance matrix is:
# $$ \begin{bmatrix} \sigma(pr)_{11} & \sigma(pr)_{12} \\ \sigma(pr)_{21} & \sigma(pr)_{22} \end{bmatrix} = \begin{bmatrix} \sigma^2_p & 0 \\ 0 & \tau^2_r - \frac{\sigma^4_p}{\sigma^2_p + \tau^2_r + \sigma^2} \end{bmatrix} $$
# 
# 

# #Question 4.1b
# 
# For the E-step, we aim to compute the expectation of the latent variables given the observed $$x(pr)$$ and the parameters $$\theta$$.
# 
# The joint distribution $$P(y(pr), z(pr), x(pr))$$ can be decomposed as:
# $$ P(y(pr), z(pr), x(pr)) = P(x(pr) | y(pr), z(pr))P(y(pr), z(pr)) $$
# 
# Taking the logarithm, we get:
# $$ \log P(y(pr), z(pr), x(pr)) = -\frac{3}{2} \log(2\pi) - \frac{1}{2} \log(\sigma^2 (\sigma'^2_p) (\tau'^2_r)) - \frac{1}{2} \left( \frac{(x(pr) - y(pr) - z(pr))^2}{2\sigma^2} + \frac{(y(pr) - \mu'^2_p)^2}{2(\sigma'^2_p)} + \frac{(z(pr) - \nu'^2_r)^2}{2(\tau'^2_r)} \right) $$
# 
# 
# Since $$x(pr) = y(pr) + z(pr) + \epsilon$$, we have:
# $$ y(pr) + z(pr) | x(pr) \sim N(x(pr), \sigma^2) $$
# 
# 
# Using variance decomposition, we find:
# $$ E[(y(pr) - \mu'^2_p)^2 (\sigma'^2_p) x(pr)] = \frac{1}{(\sigma'^2_p)} \left( (\sigma(pr)_{11} + (\mu(pr)_1 - \mu'^2_p)^2) \right) $$
# $$ E[(z(pr) - \nu'^2_r)^2 (\tau'^2_r) x(pr)] = \frac{1}{(\tau'^2_r)} \left( (\sigma(pr)_{22} + (\mu(pr)_2 - \nu'^2_r)^2) \right) $$
# 
# 
# Therefore, the expectation becomes:
# $$ Q(pr)(\theta'|\theta) = -\frac{3}{2} \log(2\pi) - \frac{1}{2} \log(\sigma^2 (\sigma'^2_p) (\tau'^2_r)) - \frac{1}{2} \left( \frac{(\sigma'^2_p) (\sigma(pr)_{11} + (\mu(pr)_1 - \mu'^2_p)^2)}{2} + \frac{(\tau'^2_r) (\sigma(pr)_{22} + (\mu(pr)_2 - \nu'^2_r)^2)}{2} \right) $$
# 
# 

# #Question 4.2
# 
# For $$\mu'_p$$:
#    $$ (\mu'_p)^* = \frac{1}{R} \sum_{r=1}^{R} \sum_{p=1}^{P} \mu(pr)_1 $$
# 
# For $$\nu'_r$$:
#    $$ (\nu'_r)^* = \frac{1}{P} \sum_{p=1}^{P} \sum_{r=1}^{R} \mu(pr)_2 $$
# 
# For $$(\sigma'^2_p)$$:
#    $$ (\sigma'^2_p)^* = \frac{1}{R} \sum_{r=1}^{R} \sum_{p=1}^{P} (\sigma(pr)_{11} + (\mu(pr)_1 - \mu'^2_p)^2) $$
# 
# For $$(\tau'^2_r)$$:
#    $$ (\tau'^2_r)^* = \frac{1}{P} \sum_{p=1}^{P} \sum_{r=1}^{R} (\sigma(pr)_{22} + (\mu(pr)_2 - \nu'^2_r)^2) $$
# 
# 

# #References
# Class code and videos
# 
# https://towardsdatascience.com/kernel-density-estimation-explained-step-by-step-7cc5b5bc4517
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
#     
# https://www.sciencedirect.com/science/article/pii/S0960982211002892?ref=cra_js_challenge&fr=RR-1
# 
# #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
# 
# #https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html

# In[ ]:




