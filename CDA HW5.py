#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Question 1

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time
import scipy.io as sio

########## LOADING DATASET ##########
data = sio.loadmat("mnist_10digits.mat")
xtrain = data["xtrain"]
xtest = data["xtest"]
ytrain = data["ytrain"].reshape(-1,)
ytest = data["ytest"].reshape(-1,)

ntest = ytest.shape[0]
ndownsample = 5000
xtrain = xtrain / 255
xtest = xtest / 255

# ~~~~~~~~~~~~~~~ logistic regression ~~~~~~~~~~~~~~~
time0 = time.time()
lr = LogisticRegression(max_iter=1000).fit(xtrain, ytrain)
y_pred_lr = lr.predict(xtest)
acc_lr = np.mean(y_pred_lr == ytest)
time1 = time.time()

conf_lr = confusion_matrix(ytest, y_pred_lr)
plt.imshow(conf_lr, cmap="hot")
plt.colorbar()
plt.title('Logistic Regression')
print('Report the test score for Logistic Regression')
print(classification_report(ytest, y_pred_lr))
print('Running time: ', round((time1 - time0), 2), 'seconds')

# ~~~~~~~~~~~~~~~~ SVM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ kernel SVM ~~~~~~~~~~~~~~ 
time0 = time.time()
sv_ker = SVC(kernel='rbf', gamma='scale').fit(xtrain[0:ndownsample], ytrain[0:ndownsample])
y_pred_sv_ker = sv_ker.predict(xtest)
time1 = time.time()

conf_ksvm = confusion_matrix(ytest, y_pred_sv_ker)
plt.imshow(conf_ksvm, cmap="hot")
plt.colorbar()
plt.title('Kernel SVM')
print('Report the test score for Kernel SVM')
print(classification_report(ytest, y_pred_sv_ker))
print('Running time: ', round((time1 - time0), 2), 'seconds')

# ~~~~~~~~~~~~~~ linear SVM ~~~~~~~~~~~~~~ 
time0 = time.time()
sv = SVC(kernel='linear').fit(xtrain[0:ndownsample], ytrain[0:ndownsample])
y_pred_sv = sv.predict(xtest)
time1 = time.time()

conf_Lsvm = confusion_matrix(ytest, y_pred_sv)
plt.imshow(conf_Lsvm, cmap="hot")
plt.colorbar()
plt.title('Linear SVM')
print('Report the test score for Linear SVM')
print(classification_report(ytest, y_pred_sv))
print('Running time: ', round((time1 - time0), 2), 'seconds')

# ~~~~~~~~~~~~~~~~ Neural network ~~~~~~~~~~~~~~~~~~~
time0 = time.time()
nn = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500).fit(xtrain, ytrain)
y_pred_nn = nn.predict(xtest)

conf_nn = confusion_matrix(ytest, y_pred_nn)
plt.imshow(conf_nn, cmap="hot")
plt.colorbar()
plt.title('Neural Network')
print('Report the test score for Neural Network')
print(classification_report(ytest, y_pred_nn))
time1 = time.time()
print('Running time: ', round((time1 - time0), 2), 'seconds')

# ~~~~~~~~~~~~~~~~ KNN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
time0 = time.time()
kn = KNeighborsClassifier(n_neighbors=3).fit(xtrain[0:ndownsample], ytrain[0:ndownsample])
y_pred_kn = kn.predict(xtest)
time1 = time.time()

conf_knn = confusion_matrix(ytest, y_pred_kn)
plt.imshow(conf_knn, cmap="hot")
plt.colorbar()
plt.title('K-Nearest Neighbor')
print('Report the test score for KNN')
print(classification_report(ytest, y_pred_kn))
print('Running time: ', round((time1 - time0), 2), 'seconds')


# #question 1b
# 
# 
# | Classifier         | Accuracy | Running Time (seconds) |
# |--------------------|----------|-------------------------|
# | Logistic Regression| 93%      | 185.65                  |
# | Kernel SVM         | 95%      | 25.48                   |
# | Linear SVM         | 91%      | 9.68                    |
# | Neural Network     | 95%      | 686.12                  |
# | KNN                | 93%      | 5.04                    |
# 
# Kernel SVM and the Neural Network tend to perform better achieving the highest accuracy of 95%, this is due to the ability to capture complex relationships in the data.It  followed closely by Logistic Regression and KNN at 93%. The Linear SVM lags slightly behind with an accuracy of 91%. The differences in performance can be attributed to the inherent properties of each classifier, such as their ability to capture complex patterns, handle non-linear data, and computational efficiency.
# 

# #Question 2.1
# 
# 
# The optimization problem aims to maximize the margin \(2c\) between the two classes while ensuring that all data points lie on or above the margin boundary \(y_i(wx_i + b) = c\). Since the magnitude of \(c\) scales the parameters \(w\) and \(b\) without altering the relative effectiveness of different classifiers, we can set \(c = 1\) to simplify the formulation without losing generality.

# # Question 2.2
# In the dual representation of the Support Vector Machine (SVM) optimization function, the objective is to minimize the expression $$1^T w + \frac{1}{2}w^Tw$$, while satisfying the constraints $$1 - y_i(wx_i + b) \leq 0$$ for all $$i$$.
# 
# The Lagrangian function for this problem is given by:
# 
# $$L(w, b, \alpha) = 1^T w + \sum_{i=1}^{n} \alpha_i (1 - y_i(wx_i + b))$$
# 
# Here, $$w$$ and $$b$$ are the SVM parameters, and $$\alpha_i$$ are the Lagrange multipliers.
# 
# When we take the derivative of $$L$$ with respect to $$w$$ and set it to zero, we obtain:
# 
# $$\frac{\partial L}{\partial w} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0$$
# 
# This leads to the conclusion:
# 
# $$w = \sum_{i=1}^{n} \alpha_i y_i x_i$$
# 
# This result reveals that the optimal separation vector $$w$$ is a weighted sum of the input data points $$x_i$$, multiplied by their corresponding labels $$y_i$$, and scaled by the Lagrange multipliers $$\alpha_i$$. 
# 
# In simpler terms, $$w = \sum_{i=1}^{n} \alpha_i y_i x_i$$ signifies that the optimal weight vector $$w$$ is determined by combining the input data points $$x_i$$ according to their labels $$y_$$, with weights provided by the Lagrange multipliers $$\alpha_i$$. This formulation serves to define the SVM's decision boundary, which effectively separates the classes within the input space.

# # Question 2.3
# The Lagrangian function  learnt in class;
# 
# $$L(w, b, \alpha) = f(w) + \sum_{i=1}^{n} \alpha_i g(w)$$
# 
# where $$f(w)$$ is the objective function and $$g(w) = (1 - y_i(wx_i + b))$$, subject to the condition $$\alpha_i \geq 0$$, we can derive the Karush-Kuhn-Tucker (KKT) conditions:
# 
# 1. $$\alpha_i g(w) = 0$$ for all $$i$$
# 2. $$\sum_{i=1}^{n} \alpha_i (1 - y_i(wx_i + b)) = 0$$
# 3. $$\alpha_i = 0$$ for $$1 - y_i(wx_i + b) < 0$$
# 4. $$\alpha_i > 0$$ for $$1 - y_i(wx_i + b) = 0$$
# 
# These conditions reveal that the Lagrange multipliers $$\alpha$$ can only be non-zero for points lying on the margin. Combined with the earlier result:
# 
# $$w = \sum_{i=1}^{n} \alpha_i y_i x_i$$
# 
# we can conclude that only the points lying on the margin contribute to the sum in the optimization. This means that the SVM decision boundary is defined by a weighted sum of the support vectors $$(x_i$$ with non-zero $$\alpha_i$$, which are precisely the points lying on the margin. 
# 
# In essence, the KKT conditions show that the points contributing to the optimization are those that are either on the margin or misclassified (with $$\alpha_i > 0$$), and these points are the ones that define the decision boundary of the SVM.

# Question 1.4(a)
# 
# 
# The range of $$h$$ for linear separability is $$0 < h < 1$$ or $$h > 4$$. This means that $$h$$ should be less than 1 to allow $$x_3$$ to lie below the line $$y = x$$, or $$h$$ should be greater than 4 to allow $$x_3$$ to lie to the right of $$x_2$$. 
# 
# In mathematical notation:
# $$0 < h < 1 \quad \text{or} \quad h > 4$$

# # Question 2.4 (b)
# 
# Yes,
# When $$h > 4$$, the orientation of the maximum margin decision boundary can be expressed as a function of $$h$$ using the vectors $$\vec{a}$$, $$\vec{b}$$, and $$\vec{c}$$ corresponding to the three points in the image.
# 
# The orientation of the decision boundary can be written as:
# 
# $$\vec{w}(h) = \vec{c} - \frac{1}{2}(\vec{a} + \vec{b})$$
# 
# Expanding this expression, we get:
# 
# $$\vec{w}(h) = \frac{2}{2} - \frac{1}{2}(0, 3) + (h, 1)$$
# 
# Simplifying, we find:
# 
# $$\vec{w}(h) = (1, 2) - \left(\frac{h}{2}, \frac{1}{2}\right)$$
# 
# So, the orientation of the decision boundary as a function of $$h$$ when $$h > 4$$ can be expressed as:
# 
# $$\vec{w}(h) = \left(1 - \frac{h}{2}, 2 - \frac{1}{2}\right)$$

# # Question 3.1
# 
# Write $$ u_i = w^T z_i $$ and differentiate the cost function with respect to $$ w $$:
#    
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial w} = \sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \frac{\partial}{\partial w} (y_i - \sigma(u_i))$$
# 
# Using the chain rule, we'll differentiate $$ (y_i - \sigma(u_i)) $$ with respect to $$ u_i $$ and then $$ u_i $$ with respect to $$ w $$:
# 
# Derivative of $$ (y_i - \sigma(u_i)) $$ with respect to $$ u_i $$:
#       
# $$\frac{\partial}{\partial u_i} (y_i - \sigma(u_i)) = -2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i))$$
# 
# Derivative of \( u_i \) with respect to $$ w $$:
#       
# $$\frac{\partial u_i}{\partial w} = z_i$$
# 
# Putting these together, we get the gradient with respect to $$ w $$:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial w} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) z_i$$
# 
# So, we have:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial w} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) z_i$$
# This is the gradient of the cost function with respect to $$ w $$:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial w} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) z_i$$

# # Question 3.2
# 
# Gradient with respect to $$ \alpha $$:
# 
# We have:
# 
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \alpha} = \sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \frac{\partial}{\partial \alpha} (y_i - \sigma(u_i))$$
# 
# Using the chain rule:
# 
# Derivative of $$ (y_i - \sigma(u_i)) $$ with respect to $$ u_i $$:
# 
# $$\frac{\partial}{\partial u_i} (y_i - \sigma(u_i)) = -2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i))$$
# 
# Differentiate $$ u_i = \alpha^T x_i $$ with respect to $$ \alpha $$:
# 
# 
# $$\frac{\partial u_i}{\partial \alpha} = x_i$$
# 
# Put these together:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \alpha} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) x_i$$
# 
# So, the gradient with respect to $$ \alpha $$ is:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \alpha} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) x_i$$
# 
# Gradient with respect to $$ \beta $$:
# 
# Similarly:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \beta} = \sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \frac{\partial}{\partial \beta} (y_i - \sigma(u_i))$$
# 
# Using the chain rule:
# 
# Derivative of $$ (y_i - \sigma(u_i)) $$ with respect to $$ u_i $$:
# 
# $$\frac{\partial}{\partial u_i} (y_i - \sigma(u_i)) = -2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i))$$
# 
# differentiate $$ u_i = \beta^T x_i $$ with respect to $$ \beta $$:
# 
# $$\frac{\partial u_i}{\partial \beta} = x_i$$
# 
# Putting these together:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \beta} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) x_i$$
# 
# The gradient with respect to $$ \beta $$ is:
# 
# $$\frac{\partial \ell(w, \alpha, \beta)}{\partial \beta} = -\sum_{i=1}^{m} 2(y_i - \sigma(u_i)) \sigma(u_i)(1 - \sigma(u_i)) x_i$$
# 

# # Question 4.1
# 
# I will use the following formula:
# 
# $$ I(\text{Keyword}; \text{Spam}) = \sum_{\text{Keyword}} \sum_{\text{Spam}} p(\text{Keyword}, \text{Spam}) \log_2 \left( \frac{p(\text{Keyword}, \text{Spam})}{p(\text{Keyword}) \cdot p(\text{Spam})} \right) $$
# 
# Let's calculate for each keyword:
# 
# For the keyword "prize":
# Counts:
# "prize" = 1, "spam" = 1: 150
# "prize" = 1, "spam" = 0: 10
# "prize" = 0, "spam" = 1: 1000
# "prize" = 0, "spam" = 0: 15000
# 
# Probabilities:
# $$p(\text{"prize"}=1) = \frac{150 + 10}{150 + 10 + 1000 + 15000} = \frac{160}{17160} $$
# $$p(\text{"prize"}=0) = \frac{1000 + 15000}{150 + 10 + 1000 + 15000} = \frac{15150}{17160} $$
# $$p(\text{"spam"}=1) = \frac{150 + 1000}{150 + 10 + 1000 + 15000} = \frac{1150}{17160} $$
# $$p(\text{"spam"}=0) = \frac{10 + 15000}{150 + 10 + 1000 + 15000} = \frac{15150}{17160} $$
# 
# Mutual Information:
# $$ I(\text{"prize"}) = \left( \frac{150}{17160} \right) \log_2 \left( \frac{150 \cdot 17160}{160 \cdot 1150} \right) + \left( \frac{10}{17160} \right) \log_2 \left( \frac{10 \cdot 17160}{15150 \cdot 160} \right) $$
# $$ I(\text{"prize"}) = 0.142137 $$
# 
# For the keyword "hello":
# Counts:
# "hello" = 1, "spam" = 1: 145
# "hello" = 1, "spam" = 0: 15
# "hello" = 0, "spam" = 1: 11000
# "hello" = 0, "spam" = 0: 5000
# 
# Probabilities:
# $$ p(\text{"hello"}=1) = \frac{145 + 15}{145 + 15 + 11000 + 5000} = \frac{160}{16260} $$
# $$ p(\text{"hello"}=0) = \frac{11000 + 5000}{145 + 15 + 11000 + 5000} = \frac{16000}{16260} $$
# $$ p(\text{"spam"}=1) = \frac{145 + 11000}{145 + 15 + 11000 + 5000} = \frac{11145}{16260} $$
# $$ p(\text{"spam"}=0) = \frac{15 + 5000}{145 + 15 + 11000 + 5000} = \frac{5015}{16260} $$
# 
# Mutual Information:
# $$ I(\text{"hello"}) = \left( \frac{145}{16260} \right) \log_2 \left( \frac{145 \cdot 16260}{160 \cdot 11145} \right) + \left( \frac{15}{16260} \right) \log_2 \left( \frac{15 \cdot 16260}{16000 \cdot 5015} \right) $$
# $$ I(\text{"hello"}) = -3.012 $$
# 
# Comparing the mutual information values:
# $$ I(\text{"prize"}) = 0.142137 $$
# $$ I(\text{"hello"}) = -3.012 $$
# 
# The positive and higher mutual information value for the keyword "prize" suggests that it provides more information for deciding whether an email is spam or not, compared to the keyword "hello". Hence, "prize" is a more informative indicator for spam classification.

# # Question 4.2
# 
# 
# In the context of change detection using the Cumulative Sum (CUSUM) statistic:
# 
# The null hypothesis $$ H_0 $$ represents the assumption of no change in the distribution of the data.
# 
# The alternate hypothesis $$ H_1 $$ represents the presence of a change in the distribution of the data.
# 
# Specifically, $$ H_0 $$ states that the data follows the initial distribution $$ f_0 $$, while $$ H_1 $$ suggests that the data transitions to a different distribution $$ f_1 $$.
# 
# The CUSUM statistic helps monitor the cumulative difference between observed data and expected values under both distributions, aiding in the detection of this change.
# 
# $$ H_0 $$ assumes no change in the data distribution, while $$ H_1 $$ suggests a transition to a new distribution, and the CUSUM statistic is used to detect this change.
# 
# To calculate the Cumulative Sum (CUSUM) statistic for change detection, we compute the log-likelihood ratio at each step based on the observed data. 
# 
# #For $$ f_0 $$ to $$ f_1 $$:
# Initialize:
# $$ \text{CUSUM0}(0) = 0 $$
# 
# At each step $$ t $$:
# Compute the log-likelihood ratio:
# $$\text{CUSUM0}(t) = \max(0, \text{CUSUM0}(t-1) + \log\left(\frac{f_1(x_t)}{f_0(x_t)}\right)) $$
# 
# #For \( f_1 \) to \( f_0 \):
# Initialize:
# $$ \text{CUSUM1}(0) = 0 $$
# At each step $$ t $$:
# Compute the log-likelihood ratio:
# $$ \text{CUSUM1}(t) = \max(0, \text{CUSUM1}(t-1) + \log\left(\frac{f_0(x_t)}{f_1(x_t)}\right)) $$
# 
# Where:
# $$ f_0(x) $$ and $$ f_1(x) $$ are the probability density functions (PDFs) of the respective distributions.
# $$ x_t $$ is the observed data point at time $$ t $$.
# 
# 
# The CUSUM statistic at each step is given by:
# $$\text{CUSUM}(t) = \max(\text{CUSUM0}(t), \text{CUSUM1}(t)) $$
# 
# This statistic captures the cumulative difference between the two distributions.
# 
# Let's generate a sequence of randomly generated samples and plot the CUSUM statistic for $$ x_1, \ldots, x_{100} $$ from $$ f_0 $$ and $$ x_{101}, \ldots, x_{200} $$ from $$ f_1 $$.
# 
# 

# In[1]:


#.... continue....

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate random samples from f0 and f1
np.random.seed(0)
f0_samples = np.random.normal(0, 1, 100)
f1_samples = np.random.normal(0.5, 1.5, 100)

# Calculate the log-likelihood ratio
log_likelihood_ratio = np.log(norm.pdf(f1_samples, 0.5, 1.5) / norm.pdf(f1_samples, 0, 1))

# Calculate the CUSUM statistic
cusum_statistic = np.maximum(0, np.cumsum(log_likelihood_ratio))

# Plot the CUSUM statistic
plt.plot(np.arange(1, 101), cusum_statistic)
plt.xlabel('Time')
plt.ylabel('CUSUM Statistic')
plt.title('CUSUM Statistic for Change Detection')
plt.grid(True) 
plt.show()


# In[1]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# In[3]:


#Question 5a

# Lasso Regression
import scipy.io
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
from sklearn.linear_model import LassoCV
from PIL import Image
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt

# Load the data 
mat = scipy.io.loadmat('cs.mat')
data = mat['img']

# Display the original image 
plt.imshow(data, interpolation='nearest')
plt.title('Original Image')
plt.show()

ny, nx = data.shape

# extract small sample of signal
k = round(nx * ny * 0.5) # 50% sample
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = data.T.flat[ri]
b = np.expand_dims(b, axis=1)

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:] # subsample the matrix

# Function for inverse DCT transform
def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho',axis=0).T, norm='ortho',
axis=0)
# Initialize LassoCV model
lasso = LassoCV(cv = 10)
lasso.fit(A, b)

# reconstruct the image from the coefficients
Xat = np.array(lasso.coef_).reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

# Display the reconstructed image
plt.imshow(Xa)
plt.show()

# Plot the cross-validation error curves
plt.semilogx(lasso.alphas_, lasso.mse_path_, ":")
plt.plot(
    lasso.alphas_ ,lasso.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    lasso.alpha_, linestyle="--", color="k", label="lambda: CV estimate"
)

plt.legend()
plt.xlabel("lambda")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")


# In[4]:


#Question 5b
# Ridge regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import validation_curve
import scipy.fftpack as spfft

# Fit RidgeCV model
model = RidgeCV().fit(A, b)

# Define range of alphas
parameter_range = np.arange(0, 0.1, 0.01)

# Calculate validation curve for Ridge Regression
train_score, test_score = validation_curve(
    model, A, b,
    param_name="alphas", param_range=parameter_range, cv=10,
    scoring="neg_mean_squared_error"
)

# Calculate mean and standard deviation of training score
mean_train_score = np.mean(train_score, axis=1)
std_train_score = np.std(train_score, axis=1)

# Calculate mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis=1)
std_test_score = np.std(test_score, axis=1)

# Plot mean accuracy scores for training and testing scores
plt.plot(parameter_range, [abs(i) for i in mean_train_score], label="Cross Validation Score", color='b')

# Create the plot
plt.title("Validation Curve with Ridge Regression")
plt.xlabel("lambda")
plt.ylabel("Mean Squared Error")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

# Get the reconstructed image using Ridge Regression
Xat_ridge = np.array(model.coef_).reshape(nx, ny).T
Xb_ridge = idct2(Xat_ridge)

# Plot the image recovered using Ridge Regression
plt.title("Ridge Regression")
plt.imshow(Xb_ridge)
plt.show()

# Plot the image recovered using Lasso Regression for comparison
plt.title("Lasso Regression")
plt.imshow(Xa)
plt.show()


# Lasso Regression penalizes the sum of the absolute values of the coefficients (L1 penalty), while Ridge Regression penalizes the sum of squared coefficients (L2 penalty).
# 
# In this scenario, Lasso Regression performs significantly better because it serves as a variable selection technique, allowing for the identification of the most significant variables.
# 
# Lasso Regression is particularly useful when there are only a few features with high predictive power, and the rest are not useful. It effectively zeros out the unimportant variables, retaining only the most relevant subset.
# 
# On the other hand, Ridge Regression is beneficial when the predictive power of the dataset is spread across various features. It does not zero out any features but instead reduces the weight of most variables in the model.

# #References
# 
# Lecture notes, code, class slack and Ed discussion
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#     
# https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
#     
# https://www.youtube.com/watch?v=nUfYR5FBGZc
#     
# 

# In[ ]:




