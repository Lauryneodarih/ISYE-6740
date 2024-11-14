#!/usr/bin/env python
# coding: utf-8

# #Question 1.1
# 
# 
# Given the cost function:
# 
# $$ \ell(\theta) = -\sum_{i=1}^{m} \left[ \log(1 + \exp(-\theta^Tx^{(i)})) + (y^{(i)} - 1)\theta^Tx^{(i)} \right] $$
# 
# I want to find the gradient of this cost function with respect to $$ \theta $$:
# 
# $$ \frac{\partial \ell(\theta)}{\partial \theta} $$
# 
# Let's compute the derivative of $$ -\log(1 + \exp(-\theta^Tx^{(i)})) $$ with respect to $$ \theta $$:
# 
# $$ \frac{\partial}{\partial \theta} \left( -\log(1 + \exp(-\theta^Tx^{(i)})) \right) $$
# 
# I will use the chain rule here:
# 
# $$ = -\frac{1}{1 + \exp(-\theta^Tx^{(i)})} \cdot \frac{\partial}{\partial \theta} \left( 1 + \exp(-\theta^Tx^{(i)}) \right) $$
# 
# $$ = -\frac{1}{1 + \exp(-\theta^Tx^{(i)})} \cdot \exp(-\theta^Tx^{(i)}) \cdot \frac{\partial}{\partial \theta} (-\theta^Tx^{(i)}) $$
# 
# $$ = \frac{\exp(-\theta^Tx^{(i)})}{1 + \exp(-\theta^Tx^{(i)})} \cdot x^{(i)} $$
# 
# $$ = \frac{1}{1 + \exp(\theta^Tx^{(i)})} \cdot x^{(i)} $$
# 
# $$ = \frac{1}{1 + \exp(\theta^Tx^{(i)})} \cdot x^{(i)} $$
# 
# Compute the derivative of $$ (y^{(i)} - 1)\theta^Tx^{(i)} $$ with respect to $$ \theta $$:
# 
# $$ \frac{\partial}{\partial \theta} \left( (y^{(i)} - 1)\theta^Tx^{(i)} \right) = (y^{(i)} - 1)x^{(i)} $$
# 
# Sum these derivatives over all training examples $$ i $$:
# 
# $$ \frac{\partial \ell(\theta)}{\partial \theta} = -\sum_{i=1}^{m} \left[ \frac{1}{1 + \exp(\theta^Tx^{(i)})} \cdot x^{(i)} + (y^{(i)} - 1)x^{(i)} \right] $$
# 
# $$ = -\sum_{i=1}^{m} \left[ \frac{1}{1 + \exp(\theta^Tx^{(i)})} + y^{(i)} - 1 \right] x^{(i)} $$
# 
# $$ = -\sum_{i=1}^{m} \left[ \frac{1 + \exp(\theta^Tx^{(i)}) + y^{(i)} - 1 + 1 - 1}{1 + \exp(\theta^Tx^{(i)})} \right] x^{(i)} $$
# 
# $$ = -\sum_{i=1}^{m} \left[ \frac{\exp(\theta^Tx^{(i)}) + y^{(i)}}{1 + \exp(\theta^Tx^{(i)})} \right] x^{(i)} $$
# 
# $$ = -\sum_{i=1}^{m} \left[ \frac{1}{1 + \exp(-\theta^Tx^{(i)})} \right] x^{(i)} $$
# 
# $$ = -\sum_{i=1}^{m} \left[ \frac{1}{1 + \exp(-y^{(i)}\theta^Tx^{(i)})} \right] x^{(i)} $$
# 
# writing it more explicitly using the sigmoid function $$ g(z) = \frac{1}{1 + \exp(-z)} $$:
# 
# $$ \frac{\partial \ell(\theta)}{\partial \theta} = -\sum_{i=1}^{m} \left[ g(y^{(i)}\theta^Tx^{(i)}) \right] x^{(i)} $$
# 

# In[4]:


#Question 1.2
import numpy as np

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

def compute_gradient(theta, x, y):
    """Compute the gradient of the cost function"""
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    gradient = np.dot(x.T, (h - y)) / m
    return gradient

def gradient_descent(x, y, tol=0.001, step=0.01):
    """Gradient Descent to find optimal theta"""
    # Initialize theta with random values between 0 and 1
    np.random.seed(0)
    theta = np.random.rand(x.shape[1])

    k = 0  # Iteration counter
    while True:
        theta_old = theta.copy()
        gradient = compute_gradient(theta, x, y)
        theta -= step * gradient
        k += 1

        if np.linalg.norm(theta - theta_old) < tol:
            break

    return theta


#sample data
x = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = np.array([0, 1, 0, 1])

# Call the gradient descent function
optimal_theta = gradient_descent(x, y)

# Print the optimized theta
print("Optimal theta:", optimal_theta)


# This pseudo-code outlines the process of gradient descent for logistic regression training. It initializes the parameters, iteratively updates them in the direction of the negative gradient, and stops when the parameters converge or the maximum number of iterations is reached.

# In[12]:


#Question 1.3
import numpy as np

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

def compute_gradient(theta, x, y):
    """Compute the gradient of the cost function"""
    m = len(y)
    h = sigmoid(np.dot(x, theta))
    gradient = np.dot(x.T, (h - y)) / m
    return gradient

def stochastic_gradient_descent(x, y, max_iterations=50, step=0.01):
    """Stochastic Gradient Descent for logistic regression"""
    np.random.seed(0)
    theta = np.random.rand(x.shape[1])
    m = len(y)

    for k in range(max_iterations):
        for i in range(m):
            random_index = np.random.randint(m)  # Randomly select an index
            x_i = x[random_index, :].reshape(1, -1)  # Reshape to (1, n)
            y_i = y[random_index].reshape(1, 1)  # Reshape to (1, 1)
            gradient = compute_gradient(theta, x_i, y_i)
            theta -= step * gradient.flatten()  # Flatten gradient to match theta shape

    return theta


# sample data
x = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])
y = np.array([0, 1, 0, 1])

# Call the stochastic gradient descent function
optimal_theta = stochastic_gradient_descent(x, y, max_iterations=50, step=0.01)

# Print the optimized theta
print("Optimal theta:", optimal_theta)


# 
# Gradient Descent (GD):
# In Gradient Descent, we calculate the gradient of the cost function using all the training examples, which can be computationally expensive for large datasets. It involves summing up the gradients of all individual training examples to update the parameters θ. This means iterating through the entire dataset for each parameter update.
# 
# Stochastic Gradient Descent (SGD):
# Stochastic Gradient Descent, on the other hand, is a more efficient approach for large datasets. Instead of using all the training examples, SGD randomly selects one example at a time to update the estimate of the objective function F. This random selection and update process is done iteratively for a number of iterations. It allows for faster parameter updates since it does not require calculating gradients for the entire dataset.
# 
# 
# For large datasets, the standard Gradient Descent (GD) approach becomes computationally expensive. In GD, the algorithm calculates the gradient of the cost function using all n training examples, which requires summing up the gradients of each individual training example to update the parameters θ. This process can be time-consuming and memory-intensive, especially when dealing with large n's.
# 
# In contrast, Stochastic Gradient Descent (SGD) offers a more efficient alternative. Instead of using all the training examples, SGD randomly selects a single observation (or a small batch of observations) to update the estimate of the objective function F. This random selection and parameter update process is done iteratively for a number of iterations, making it much faster compared to GD for large datasets. The randomness in SGD allows for quicker convergence to a solution while avoiding the computational overhead of processing the entire dataset in each iteration.

# 
# #Question 1.4
# 
# To show that the training problem in basic logistic regression is concave, we will derive the Hessian matrix of the log-likelihood function ℓ(θ) and demonstrate that it is negative semi-definite. 
# 
# The log-likelihood function in logistic regression is given by:
# 
# $$ ℓ(θ) = \sum_{i=1}^{m} -\log(1 + \exp(-θ^Tx_i)) + (y_i - 1)\theta^Tx_i $$
# 
# where $$theta $$ is the parameter vector, $$ x_i $$ are the feature vectors, and $$ y_i $$ are the corresponding labels.
# 
# 
# The Hessian matrix of $$ ℓ(θ) $$ with respect to $$ θ $$ is calculated as follows:
# 
# $$ H_{ij} = \frac{\partial^2 ℓ(θ)}{\partial θ_i \partial θ_j} $$
# 
# Let's compute the Hessian matrix:
# 
# $$ H_{ij} = \frac{\partial^2 ℓ(θ)}{\partial θ_i \partial θ_j} = - \frac{\partial}{\partial θ_j} \left( \frac{\partial ℓ(θ)}{\partial θ_i} \right) $$
# 
# Since the gradient of $$ ℓ(θ) $$ is:
# 
# $$ \frac{\partial ℓ(θ)}{\partial θ_i} = -\sum_{i=1}^{m} \frac{x_i \exp(-θ^Tx_i)}{1 + \exp(-θ^Tx_i)} + (y_i - 1)x_i $$
# 
# We have:
# 
# $$ H_{ij} = \frac{\partial^2 ℓ(θ)}{\partial θ_i \partial θ_j} = \sum_{i=1}^{m} \left( \frac{x_i^2 \exp(-θ^Tx_i)}{(1 + \exp(-θ^Tx_i))^2} \right) $$
# 
# Simplifying further, we get:
# 
# $$ H_{ij} = \sum_{i=1}^{m} x_i x_i^T \frac{\exp(-θ^Tx_i)}{(1 + \exp(-θ^Tx_i))^2} $$
# 
# This is the Hessian matrix for the logistic regression log-likelihood function.
# 
# 
# For a function to be concave, its Hessian matrix must be negative semi-definite. 
# 
# Let's analyze the Hessian matrix:
# 
# $$ H = \sum_{i=1}^{m} x_i x_i^T \frac{\exp(-θ^Tx_i)}{(1 + \exp(-θ^Tx_i))^2} $$
# 
# Since $$ x_i x_i^T $$ is a positive semi-definite matrix and $$ \frac{\exp(-θ^Tx_i)}{(1 + \exp(-θ^Tx_i))^2} $$ is always non-negative, the Hessian $$ H $$ is a sum of positive semi-definite matrices, making it negative semi-definite.
# 
# Thus, the log-likelihood function $$ ℓ(θ) $$ is concave.
# 
# 
# Now, since the log-likelihood function is concave, the optimization problem is well-behaved. Gradient Descent (GD) or its variants can efficiently find the global optimum. 
# 
# Gradient Descent and many of its variants such as Newton's method can efficiently converge to the global solution.
#   
# A unique global optimizer is achieved due to the concavity of the function.
#   
# Gradient Descent moves along the function until the derivative is zero or the slope of the curve is zero, making it efficient for optimization.
#   

# 
# 
# #Question 2.1
# 
# 
# The class prior $$ P(y = 0) $$ for spam messages would be $$ \frac{3}{7} $$ because there are 3 spam messages out of 7 total messages. Similarly, $$ P(y = 1) $$ for non-spam messages would be $$ \frac{4}{7} $$ because there are 4 non-spam messages out of 7 total messages.
# 
# 
# Spam Messages:
# 1. "million dollar offer for today":
# Feature vector $$ x^{(1)} $$: [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 2. "secret offer today":
# Feature vector $$ x^{(2)} $$: [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# 3. "secret is secret":
# Feature vector $$ x^{(3)} $$: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# 
# Non-Spam Messages:
# 1. "low price for valued customer today":
# Feature vector $$ x^{(4)} $$: [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
# 2. "play secret sports today":
# Feature vector $$ x^{(5)} $$: [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
# 3. "sports is healthy":
# Feature vector $$ x^{(6)} $$: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
# 4. "low price pizza today":
# Feature vector $$ x^{(7)} $$: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# 
# #These feature vectors represent each message, where each entry $$ x^{(i)}_j $$ denotes the number of times the $$ j $$-th word in the vocabulary $$ V $$ occurs in the $$ i $$-th message.
# 

# 
# #Question 2.2
# 
# 
# Assuming the keywords follow a multinomial distribution, the likelihood of a sentence with its feature vector $$ x $$ given a class $$ c $$ is given by:
# 
# $$ P(x|y = c) = \frac{n!}{x_1! \cdot x_2! \cdot \ldots \cdot x_d!} \prod_{k=1}^{d} \theta_{c,k}^{x_k}, \quad c \in \{0, 1\} $$
# 
# where $$ n = x_1 + x_2 + \ldots + x_d $$, $$ 0 \leq \theta_{c,k} \leq 1 $$ is the probability of word $$ k $$ appearing in class $$ c $$, and $$ \sum_{k=1}^{d} \theta_{c,k} = 1 \) for \( c \in \{0, 1\} $$.
# 
# Given this, the complete log-likelihood function for our training data is given by:
# 
# $$ \mathcal{L}(\theta_{0,1}, \ldots, \theta_{0,d}, \theta_{1,1}, \ldots, \theta_{1,d}) = \sum_{i=1}^{m} \sum_{k=1}^{d} x^{(i)}_k \log \theta_{y^{(i)},k} $$
# 
# From the previously derived formulas, we can extract the following specific values for $$ \theta_{0,1} $$, $$ \theta_{0,7} $$, $$ \theta_{1,1} $$, and $$ \theta_{1,15} $$:
# 
# - $$ \theta_{0,1} = \frac{1}{3} $$
# - $$ \theta_{0,7} = \frac{1}{9} $$
# - $$ \theta_{1,1} = \frac{1}{15} $$
# - $$ \theta_{1,15} = \frac{1}{15} $$
# 
# These values represent the probabilities of each word appearing in the respective classes based on the provided spam and non-spam messages and the given vocabulary.

# 
# 
# # Question 2.3
# 
# 
# With the previously calculated parameters, we can compute the necessary values:
# 
# For class 0 (Spam):
# - $$ \theta_{0,1} = \frac{1}{3} $$
# - $$ \theta_{0,7} = \frac{1}{9} $$
# - $$ \theta_{0,15} = \frac{1}{15} $$
# 
# For class 1 (Non-Spam):
# - $$ \theta_{1,1} = \frac{1}{15} $$
# - $$ \theta_{1,7} = \frac{1}{9} $$
# - $$ \theta_{1,15} = \frac{1}{15} $$
# 
# Now, we are given a test paragraph with the data vector:
# $$ \mathbf{x} = [x_1 = 1, x_7 = 1, x_{15} = 1] $$
# 
# We can compute the conditional probabilities:
# 
# $$ p(\mathbf{x}|y=0) = \left(\frac{1}{3}\right)^1 \cdot \left(\frac{1}{9}\right)^1 \cdot \left(\frac{1}{15}\right)^1 = \frac{1}{3 \times 9 \times 15} $$
# $$ p(\mathbf{x}|y=1) = \left(\frac{1}{15}\right)^1 \cdot \left(\frac{1}{9}\right)^1 \cdot \left(\frac{1}{15}\right)^1 = \frac{1}{15 \times 9 \times 15} $$
# 
# By Bayes' rule:
# $$ p(y=0|\mathbf{x}) = \frac{p(\mathbf{x}|y=0) \cdot p(y=0)}{p(\mathbf{x}|y=1) \cdot p(y=1) + p(\mathbf{x}|y=0) \cdot p(y=0)} $$
# 
# Substituting the values:
# $$ p(y=0|\mathbf{x}) = \frac{\frac{1}{3 \times 9 \times 15} \cdot \frac{3}{7}}{\frac{1}{15 \times 9 \times 15} \cdot \frac{4}{7} + \frac{1}{3 \times 9 \times 15} \cdot \frac{3}{7}} $$
# 
# Simplifying:
# $$ p(y=0|\mathbf{x}) = \frac{\frac{1}{3 \times 9}}{\frac{1}{15 \times 9} + \frac{1}{3 \times 9}} = \frac{1}{1 + 5} = \frac{1}{6} $$
# 
# Setting the threshold to 0.5, we can classify this message as spam since the probability of it being spam $$( p(y=0|\mathbf{x})) $$ is less than 0.5.

# In[15]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[3]:


import numpy as np
from pathlib import Path
import scipy.io
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

np.random.seed(2)

raw = []
file2open = 'marriage.csv'

with open(file2open) as cf:
    readcsv = csv.reader(cf, delimiter=',')
    for row in readcsv:
        raw.append(row)

data = np.array(raw).astype(np.float)
x = data[:, 0:-1]
y = data[:, -1]

def threeClassifier(target_x, target_y, noise_level, ntrials):
    rate_nb, rate_lr, rate_kn = np.zeros(ntrials), np.zeros(ntrials), np.zeros(ntrials)
    
    for ii in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(target_x, target_y, test_size=0.2)
        ntest = len(y_test)
        
        # Naive Bayes
        nb = GaussianNB(var_smoothing=noise_level)
        y_pred_nb = nb.fit(X_train, y_train).predict(X_test)
        rate_nb[ii] = sum(y_pred_nb == y_test) / ntest
        
        # Logistic Regression
        lr = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        rate_lr[ii] = sum(y_pred_lr == y_test) / ntest
        
        # KNN
        kn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
        y_pred_kn = kn.predict(X_test)
        rate_kn[ii] = sum(y_pred_kn == y_test) / ntest
    
    acc_nb = rate_nb.mean()
    acc_lr = rate_lr.mean()
    acc_kn = rate_kn.mean()
    
    return acc_nb, acc_lr, acc_kn

def plot_decision_boundary(model, title, x_train, x_test, y_train):
    h = 0.01
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    x_min, x_max = x_train[:, 0].min(), x_train[:, 0].max()
    y_min, y_max = x_train[:, 1].min(), x_train[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

def pca2(x_):
    xc = (x_ - x_.mean(axis=0))
    u, s, _ = np.linalg.svd(xc.T @ xc / len(xc))
    xt = xc @ u[:, 0:2] @ np.diag(s[0:2]**-1 / 2)
    return xt, u, s

# Part 1a: Accuracy for marriage data
mrg_nb, mrg_lr, mrg_kn = threeClassifier(x, y, 1e-9, 1)
print('Result for marriage data')
print('Test accuracy of Naive Bayes: ', mrg_nb)
print('Test accuracy of Logistic Regression: ', mrg_lr)
print('Test accuracy of KNN: ', mrg_kn)


# I executed the program for 100 trials and computed the average accuracy of the results. The dataset appears to be relatively straightforward for classification, as indicated by the nearly linear separability observed in the two-feature plots. Consequently, the performance of all three classifiers—Naive Bayes, Logistic Regression, and KNN—was found to be quite similar.

# In[14]:


# Part 1b: Boundary plots for marriage data
X_tr, X_te, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Perform PCA
X_train, u, s = pca2(X_tr)
X_test = (X_te - X_tr.mean(axis=0)) @ u[:, 0:2] @ np.diag(s[0:2]**-1 / 2)

# Naive Bayes
nb = GaussianNB().fit(X_train, y_train)
plot_decision_boundary(nb, 'Marriage: Naive Bayes', X_train, X_test, y_train)

# Logistic Regression
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
plot_decision_boundary(lr, 'Marriage: Logistic Regression', X_train, X_test, y_train)

# KNN
kn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
plot_decision_boundary(kn, 'Marriage: KNN', X_train, X_test, y_train)

plt.show()


# In the decision boundary plots, we observe the expected patterns for each classifier:
# 
# Naive Bayes: The boundary appears as the intersection of two ellipses. This is due to the assumption of Gaussian distribution for each class, resulting in elliptical decision boundaries.
# 
# Logistic Regression: With a linear link function, the boundary is represented by a straight line. In higher-dimensional cases, this translates to a hyperplane separating the classes.
# 
# KNN (K-Nearest Neighbors): The boundary appears "rugged" with varying shapes and contours. The smoothness of this boundary depends on the choice of the number of neighbors (k). A higher k value leads to a smoother boundary, while a lower k value results in a more intricate and potentially noisier boundary.

# 
# 
# # REFERENCES
# 
# - Weekly videos and sample code
# 
# - https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
#     
# - https://machinelearningmastery.com/a-gentle-introduction-to-hessian-matrices/
#     
# - https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function

# In[ ]:




