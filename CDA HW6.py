#!/usr/bin/env python
# coding: utf-8

# # Question 1a
# To control the data-fit complexity in a regression tree, we adjust the tree size, which is the number of terminal nodes. This is crucial to avoid overfitting, where the model memorizes the training data but performs poorly on new data. Two key hyperparameters help control the tree size:
# 
# Min_samples_split: Determines the minimum samples needed to split a node. A small value leads to many leaves with smaller subsets of data, resulting in high training error but low data-fit complexity. A large value creates fewer leaves with larger subsets, reducing training error but increasing data-fit complexity.
# 
# Min_samples_leaf: Sets the minimum samples allowed in a terminal leaf node. A small value results in many leaves with small subsets, leading to high data-fit complexity but potentially low generalization error. A large value creates fewer leaves with larger subsets, reducing data-fit complexity but potentially increasing generalization error.
# 
# Choosing appropriate values, such as `min_samples_split` close to the square root of total samples and `min_samples_leaf` close to total samples, helps strike a balance between model complexity and accuracy on unseen data. This ensures the regression tree performs well by avoiding both underfitting and overfitting, improving its reliability and effectiveness.

# # Question 1b
# 
# Out-of-bag (OOB) errors are generated in the process of training a random forest with bagging. In this method, each tree is trained on a bootstrap sample, leaving some data out to form an out-of-bag sample. This OOB sample serves as the test data for the trees that were not trained on it. The final prediction is made by majority vote or average prediction from all trees.
# 
# The OOB error represents the number of incorrectly predicted rows from this out-of-bag sample. To determine the optimal number of trees for a random forest, we monitor when the OOB error stabilizes. Training can be stopped when the OOB error reaches a stable point.
# 
# Since the OOB sample is distinct from the data used to train the model, the OOB error serves as a form of test error rather than training error.

# # Question 1c
# 
# Random forest is a bagging technique, not a boosting technique. In boosting, the learning process is indeed boosted by iteratively learning from the errors of previous models.
# 
# In a random forest, the trees are trained independently and in parallel. There is no interaction between these trees during the building process. Once all trees are constructed, their predictions are combined through voting or averaging for classification or regression problems, respectively.
# 
# On the other hand, boosting algorithms like GBM (Gradient Boosting Machine) train trees sequentially. Consider the scenario where the first tree is trained and makes predictions on the training data. Not all predictions will be accurate. Let's say, out of 100 predictions, the first tree incorrectly predicts 10 observations. In the next step, these 10 misclassified observations are given more weight when training the second tree. This process boosts the learning of the second tree from the errors of the first tree. This sequential learning from past mistakes continues for subsequent trees in the boosting process, hence the term "boosting". Each tree is built to correct the mistakes of the previous trees, improving the overall model's predictive power as the process iterates.

# # Question 1d
# 
# 1.Hold-out (Data Split):
# Splitting the dataset into training and testing sets, often using an 80-20 ratio. Training the model on the training set to ensure good performance not only on seen data but also on unseen testing data. This method requires a sufficiently large dataset to maintain training effectiveness post-split.
# 
# 2.Cross-validation (Data Split):
# Dividing the dataset into k subsets (k-fold cross-validation) where each subset serves as the testing set once, with the rest used for training. This process repeats for each subset, ensuring all data is eventually used for training and testing. Cross-validation provides a comprehensive assessment of model generalization but is computationally more intensive than hold-out.
# 
# 3.Data Augmentation (Data Expansion):
# Increasing the dataset size artificially, especially when gathering more data is not feasible. Techniques such as image flipping, rotation, rescaling, or shifting are employed for tasks like image classification, effectively diversifying the training data.
# 
# 4.Feature Selection (Data Pruning):
# Identifying and utilizing only the most relevant features for training, particularly when faced with a limited sample size with numerous features. This avoids unnecessary complexity and potential overfitting. Various methods, including testing individual features or employing established feature selection algorithms, are used.
# 
# 5.L1 / L2 Regularization (Learning Algorithm):
# Introducing penalty terms in the cost function to control the complexity of the model. L1 regularization drives some weights to absolute zero, while L2 regularization encourages weights to approach zero. Both methods prevent extreme weight values, reducing overfitting tendencies.
# 
# 6.Reducing Model Complexity (Model Structure):
# Directly reducing the complexity of the model by:
# -Removing layers or reducing the number of layers to find an optimal balance between underfitting and overfitting.
# -Decreasing the number of neurons in fully-connected layers to simplify the model's learning process.
# 
# 7.Dropout (Model Regularization):
# Applying dropout regularization to layers, which randomly deactivates a subset of units during training. This technique reduces interdependent learning among units, discouraging overfitting. However, it often requires more training epochs for convergence.
# 
# 8.Early Stopping (Model Training):
# Training the model for a large number of epochs and monitoring the validation loss. When the loss stops decreasing and starts to increase, training is halted to prevent overfitting. This can be done by observing loss graphs or setting an early stopping trigger, ensuring the model's optimal generalization performance.
# 
# These methods collectively aim to curb overfitting in Classification and Regression Trees (CART), enhancing the model's ability to generalize well on unseen data while maintaining predictive accuracy.

# # Question 1e
# Bias is the simplifying assumptions made by the model to make the target function easier to approximate. Variance is the amount that the estimate of the target function will change given different training data. Trade-off is tension between the error introduced by the bias and the variance

# In[ ]:





# In[ ]:





# In[ ]:


import os
os.chdir('C:\\Users\\laury\\OneDrive\\Documents')
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


# #Question 2a
# ### First Iteration :
# 
# 1.Initialize weights:
# Set all weights $$D_1(i)$$ to $$\frac{1}{N}$$ where $$N$$ is the number of samples.
# $$D_1 = \left[\frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}\right] $$
# 
# 2.Train the first weak learner (decision stump) on the data:
# If we choose the threshold as -0.5: Points $$X1$$, $$X2$$, $$X3$$, and $$X7$$ are misclassified and Points $$X4$$, $$X5$$, $$X6$$, and $$X8$$ are correctly classified.
# The error $$\varepsilon_1$$ is the sum of weights for misclassified samples:
# $$\varepsilon_1 = D_1(1) + D_1(2) + D_1(3) + D_1(7) = \frac{1}{8} + \frac{1}{8} + \frac{1}{8} + \frac{1}{8} = \frac{1}{2} $$
# The weight of this stump's prediction:
# $$ \alpha_1 = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_1}{\varepsilon_1}\right) = \frac{1}{2} \ln\left(\frac{1 - \frac{1}{2}}{\frac{1}{2}}\right) = \frac{1}{2} \ln(1) = 0 $$
# The decision stump separates the data based on the threshold of -0.5 on the X-axis.
# 
# ### Decision Stump for  first Iteration :
# ```
#          |
#          |
#    -1 ___|___ 0
#          |
#          |
# ```
# 
# ###  Second Iteration :
# 
# 1. **Train the second weak learner (decision stump) on the updated weights \(D_2\):**
# Assume the decision stump splits based on the second feature (Y-axis).
# If we choose the threshold as 0: Points $$X1$$, $$X7$$, and $$X8$$ are misclassified and points $$X2$$, $$X3$$, $$X4$$, $$X5$$, and $$X6$$ are correctly classified.
# The error $$\varepsilon_2$$ is the sum of weights for misclassified samples:
# $$ \varepsilon_2 = D_2(1) + D_2(7) + D_2(8) = \frac{1}{8} + \frac{1}{8} + \frac{1}{8} = \frac{3}{8} $$
# The weight of this stump's predictionn is:
# $$ \alpha_2 = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_2}{\varepsilon_2}\right) = \frac{1}{2} \ln\left(\frac{1 - \frac{3}{8}}{\frac{3}{8}}\right) = \frac{1}{2} \ln\left(\frac{5}{3}\right) \approx 0.2027 $$
# Updatingthe weights:
# $$ Z_2 = 2 * \sqrt{\varepsilon_2 * (1 - \varepsilon_2)} = 2 * \sqrt{\frac{3}{8} * \frac{5}{8}} = \frac{1}{\sqrt{2}} $$
# $$ D_3(i) = \frac{D_2(i) * e^{-\alpha_2 * y_i * h_2(x_i)}}{Z_2} $$
# Where:
# $$ h_2(x) $$ is the prediction of the second decision stump.
# $$ y_i $$ is the label of sample $$ i $$.
# Calculating the updated weights:
# $$ D_3(1) = \frac{1}{8} * e^{-0.2027} \approx 0.1038 $$
# $$ D_3(7) = \frac{1}{8} * e^{-0.2027} \approx 0.1038 $$
# $$ D_3(8) = \frac{1}{8} * e^{-0.2027} \approx 0.1038 $$
# $$ D_3(2) = \frac{1}{8} * e^{0.2027} \approx 0.2411 $$
# $$ D_3(3) = \frac{1}{8} * e^{0.2027} \approx 0.2411 $$
# $$ D_3(4) = \frac{1}{8} * e^{0.2027} \approx 0.2411 $$
# $$ D_3(5) = \frac{1}{8} * e^{0.2027} \approx 0.2411 $$
# $$ D_3(6) = \frac{1}{8} * e^{0.2027} \approx 0.2411 $$
# 
# 
# ### Decision Stump for second Iteration:
# ```
#     -1 __|__ 0
#          |
#          |
#    -1 ___|___ 0
#          |
#          |
# ```
# 
# ###  Third Iteration:
# 
# Train the third weak learner (decision stump) on the updated weights $$D_3$$
# Assume the decision stump splits based on the first feature (X-axis) again.
# If we choose the threshold as -0.5: Points $$X1$$, $$X2$$, $$X3$$, and $$X7$$ are misclassified and Points $$X4$$, $$X5$$, $$X6$$, and $$X8$$ are correctly classified.
# The error $$\varepsilon_3$$ is the sum of weights for misclassified samples:
# $$ \varepsilon_3 = D_3(1) + D_3(2) + D_3(3) + D_3(7) = 0.1038 + 0.2411 + 0.2411 + 0.1038 = 0.6898 $$
# 
# The weight of this stump's prediction:
# $$ \alpha_3 = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_3}{\varepsilon_3}\right) = \frac{1}{2} \ln\left(\frac{1 - 0.6898}{0.6898}\right) \approx 0.2989 $$
# Updated weights:
# $$ Z_3 = 2 * \sqrt{\varepsilon_3 * (1 - \varepsilon_3)} \approx 1.4476 $$
# $$ D_4(i) = \frac{D_3(i) * e^{-\alpha_3 * y_i * h_3(x_i)}}{Z_3} $$
#    
#    
# $$ h_3(x) $$ is the prediction of the third decision stump.
# $$ y_i $$ is the label of sample $$ i $$.
# Calculating the updated weights:
# $$ D_4(1) \approx \frac{0.1038}{1.4476} \approx 0.0717 $$
# $$ D_4(7) \approx \frac{0.1038}{1.4476} \approx 0.0717 $$
# $$ D_4(2) \approx \frac{0.2411}{1.4476} \approx 0.1667 $$
# $$ D_4(3) \approx \frac{0.2411}{1.4476} \approx 0.1667 $$
# $$ D_4(4) \approx \frac{0.2411}{1.4476} \approx 0.1667 $$
# $$ D_4(5) \approx \frac{0.2411}{1.4476} \approx 0.1667 $$
# $$ D_4(6) \approx \frac{0.2411}{1.4476} \approx 0.1667 $$
# $$ D_4(8) \approx \frac{0.1038}{1.4476} \approx 0.0717 $$
#  
# 
# ### Decision Stump for  thirdIteration :
# ```
#          |
#          |
#    -1 ___|___ 0
#          |
#     1 ___|___
#          |
#          |
# ```
# 
# 

# # Question 2b
# 
# 
# | t  | ϵt    | αt     | Zt      | Dt(1) | Dt(2) | Dt(3) | Dt(4) | Dt(5) | Dt(6) | Dt(7) | Dt(8) |
# |----|-------|--------|---------|-------|-------|-------|-------|-------|-------|-------|-------|
# | 1  | 12    | 0      | -       | 18    | 18    | 18    | 18    | 18    | 18    | 18    | 18    |
# | 2  | 38    | 0.2027 | 1.63299 | 0.2411| 0.2411| 0.2411| 0.2411| 0.2411| 0.2411| 0.1038| 0.1038|
# | 3  | 0.6898| 0.2989 | 1.4476  | 0.0717| 0.1667| 0.1667| 0.1667| 0.1667| 0.1667| 0.0717| 0.0717|
# 
# ### Calculating Training Error:
# The training error of AdaBoost is the sum of the weights of the misclassified points divided by the sum of all weights.
# 
# Iteration 1:
# Misclassified points: $$ X1, X2, X3, X7 $$
# Total weight of misclassified points: $$ D_1(1) + D_1(2) + D_1(3) + D_1(7) = 18 + 18 + 18 + 18 = 72 $$
# 
# Iteration 2:
# Misclassified points: $$ X1, X7, X8 $$
# Total weight of misclassified points: $$ D_2(1) + D_2(7) + D_2(8) = 0.2411 + 0.1038 + 0.1038 = 0.4487 $$
# 
# Iteration 3:
# Misclassified points: $$ X1, X2, X3, X7 $$
# Total weight of misclassified points: $$ D_3(1) + D_3(2) + D_3(3) + D_3(7) = 0.1038 + 0.2411 + 0.2411 + 0.1038 = 0.6898 $$
# 
# ### Total Training Error:
# Total training error = Sum of misclassified points' weights at each iteration
# Total training error = $$ 72 + 0.4487 + 0.6898 = 73.1385 $$
# 
# ### Explanation of AdaBoost vs. Single Decision Stump:
# AdaBoost combines multiple weak learners (in this case, decision stumps) to create a strong learner. It does this by assigning weights to the classifiers based on their accuracy in each iteration. Here's why AdaBoost typically outperforms a single decision stump:
# 
# Error Reduction:
# AdaBoost focuses on correcting the mistakes of the previous models. It gives more weight to misclassified points, forcing subsequent models to focus on them.
# 
# Increased Model Complexity:
# By combining multiple weak learners, AdaBoost can create a more complex decision boundary. A single decision stump can only make simple, one-dimensional splits. AdaBoost, with its ensemble approach, can learn more intricate patterns in the data.
# 
# Robustness:
# AdaBoost is less prone to overfitting compared to a single decision stump. The ensemble nature of AdaBoost helps in generalizing better to unseen data.
# 
# Versatility:
# AdaBoost can be used with various base learners, not just decision stumps. This flexibility allows AdaBoost to adapt to different types of data and problems.
# 
# 

# In[19]:


# Question 3a

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
data =  pd.read_csv('spambase.data', sep=",", header=None)
data.columns = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over',
'word_freq_remove','word_freq_internet','word_freq_order','word_freq_mail','word_freq_receive','word_freq_will',
'word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business','word_freq_email',
'word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857',
'word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm',
'word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re','word_freq_edu',
'word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!','char_freq_$',
'char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam']
data.head()


# In[20]:


# Split data into test/train
X = data.iloc[:,0:57]
y = data.iloc[:,57:58]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1234)


clf = DecisionTreeClassifier(min_samples_leaf = 100)
clf = clf.fit(X_train, y_train.values.ravel())

# Predict labels
y_pred_train_tree = clf.predict(X_train)
y_pred_test_tree = clf.predict(X_test)


# Plot the tree
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(clf)
plt.savefig('Full_Tree.jpg')


# In[21]:


# Plot first 5 levels of tree
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(clf, max_depth = 4)
plt.savefig('Partial_Tree.jpg')


# In[22]:


# Questyion 3b

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(min_samples_leaf = 100, random_state=1234)
clf_rf.fit(X_train, y_train.values.ravel())

y_pred_train_rf = clf_rf.predict(X_train)
y_pred_test_rf = clf_rf.predict(X_test)


# Generate test error
from sklearn.metrics import accuracy_score
tree_error = 1 - accuracy_score(y_test, y_pred_test_tree)
rf_error = 1 - accuracy_score(y_test, y_pred_test_rf)

tree_error, rf_error


# In[23]:


# Plot Test Error vs Number of Trees for Random Forest
min_n = 1
max_n = 200

n = []
rf_error = []

# Loop through different number of trees
for i in np.arange(min_n, max_n + 1,2):
#     print(i)
    clf_rf_plt = RandomForestClassifier(min_samples_leaf = 100, random_state=1234, n_estimators=i)
    clf_rf_plt.fit(X_train, y_train.values.ravel())
    y_pred_test_rf_plt = clf_rf_plt.predict(X_test)
    rf_plt_error = 1 - accuracy_score(y_test, y_pred_test_rf_plt)
    n.append(i)
    rf_error.append(rf_plt_error)
    
cart_error = np.ones(len(n))*tree_error

plt.plot(n,rf_error)
plt.plot(n,cart_error)
plt.legend(['Random Forest', 'CART'], loc='upper right')
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.savefig('Error_Rate.jpg')


# In[27]:


#Question 3c
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


n_variables = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Range of values for ν

oob_errors = []
test_errors = []

for n_var in n_variables:
    clf_rf_n_var = RandomForestClassifier(min_samples_leaf=100, random_state=1234, max_features=n_var, oob_score=True)
    clf_rf_n_var.fit(X_train, y_train.values.ravel())
    oob_error = 1 - clf_rf_n_var.oob_score_
    test_pred = clf_rf_n_var.predict(X_test)
    test_error = 1 - accuracy_score(y_test, test_pred)
    oob_errors.append(oob_error)
    test_errors.append(test_error)


# Create a DataFrame
error_df = pd.DataFrame({
    'Number of Variables (ν)': n_variables,
    'OOB Error': oob_errors,
    'Test Error': test_errors
})

# Display the DataFrame
print(error_df)

    
# Plot OOB Error and Test Error against ν
plt.plot(n_variables, oob_errors, label='OOB Error')
plt.plot(n_variables, test_errors, label='Test Error')
plt.xlabel('Number of Variables (ν)')
plt.ylabel('Error Rate')
plt.title('Random Forest Sensitivity Analysis to Number of Variables')
plt.legend()
plt.savefig('Sensitivity_Analysis_RF.jpg')


# In[17]:


#Question 3d

X_train_nspam = X_train[y_train['spam']==0]
from sklearn.svm import OneClassSVM

clf_svm = OneClassSVM(kernel='rbf', gamma='auto').fit(X_train_nspam)
y_pred_svm = clf_svm.predict(X_test)
y_pred_svm[y_pred_svm==1]=0
y_pred_svm[y_pred_svm==-1]=1

svm_error = 1 - accuracy_score(y_test, y_pred_svm)

svm_error


# # Question 4a
# 
# We want to find the coefficients $$\beta_0$$ and $$\beta_1$$ that minimize the weighted sum of squared residuals. The weight $$W$$ is given by the Gaussian kernel function:
# 
# $$ Kh(z) = \frac{1}{(\sqrt{2\pi}h)^p} e^{-\frac{\|z\|^2}{2h^2}} $$
# 
# Defining the Matrices
# Let $$X$$ be the matrix of predictors where each row is a predictor vector $$x_i$$, augmented with a column of 1s for the intercept term:
# $$ X = \begin{bmatrix} 1 & x_1^T \\ 1 & x_2^T \\ \vdots & \vdots \\ 1 & x_n^T \end{bmatrix} $$
# 
# Let $$W$$ be the diagonal matrix of weights, where the $$i$$th diagonal element is given by the weight corresponding to the $$i$$th data point:
# $$ W = \text{diag}\left(Kh(x - x_i)\right) = \begin{bmatrix} Kh(x - x_1) & 0 & \cdots & 0 \\ 0 & Kh(x - x_2) & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & Kh(x - x_n) \end{bmatrix} $$
# 
# Let \(Y\) be the vector of target values:
# $$ Y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix} $$
# 
# The Loss Function
# 
# $$ L(\beta_0, \beta_1) = \sum_{i=1}^{n} W_{ii} (y_i - \beta_0 - (x - x_i)^T \beta_1)^2 $$
# 
# Minimize the Loss
# To find the coefficients $$\beta_0$$ and $$\beta_1$$ that minimize the loss function, we set the derivative of $$L$$ with respect to $$\beta_0$$ and $$\beta_1$$ to zero:
# 
# $$ \frac{\partial L}{\partial \beta_0} = -2 \sum_{i=1}^{n} W_{ii} (y_i - \beta_0 - (x - x_i)^T \beta_1) = 0 $$
# 
# $$ \frac{\partial L}{\partial \beta_1} = -2 \sum_{i=1}^{n} W_{ii} (y_i - \beta_0 - (x - x_i)^T \beta_1)(x - x_i) = 0 $$
# 
# Expressing in Matrix Form
# Let $$Z = X - x^T$$, a matrix where each row is $$x_i - x$$.
# Let $$D = \text{diag}(W)$$, the diagonal matrix with the weights.
# 
# The equations become:
# 
# $$ X^T W (Y - X \beta) = 0 $$
# $$ X^T W Z (Y - X \beta) = 0 $$
# 
# Expanding, we get:
# 
# $$ X^T W Y - X^T W X \beta = 0 $$
# $$ X^T W Z Y - X^T W Z X \beta = 0 $$
# 
# Solve for $$\beta$$
# Solving for $$\beta$$, we get:
# 
# $$ X^T W Y = X^T W X \beta $$
# $$ X^T W Z Y = X^T W Z X \beta $$
# 
# Multiplying both sides by $$(X^T W X)^{-1}$$, we get the desired form:
# 
# $$ \beta = (X^T W X)^{-1} X^T W Y $$
# 
# 

# In[11]:


#Question 4b

import scipy.io as sio

# Load data from the .mat file
data = sio.loadmat("data.mat")
#print (data)


# In[7]:


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

# Extract X and Y from the loaded data
X = data['data'][:, 0]  # Predictor variable
Y = data['data'][:, 1]  # Target variable

def local_linear_weighted_regression(X_train, Y_train, X_test, h):
    # Calculate weights
    weights = np.exp(-0.5 * ((X_train - X_test) / h) ** 2)
    weights /= np.sum(weights)

    # Design matrix
    X_design = np.vstack((np.ones_like(X_train), X_train)).T

    # Weighted linear regression
    beta = np.linalg.inv(X_design.T @ np.diag(weights) @ X_design) @ X_design.T @ (weights * Y_train)

    # Predict using the local linear model
    X_test_design = np.vstack((np.ones_like(X_test), X_test)).T
    Y_test_pred = X_test_design @ beta

    return Y_test_pred

# Define a range of bandwidth values to try
bandwidth_values = [0.1, 0.5, 1.0, 2.0, 5.0]

# Initialize variables to store average MSE for each bandwidth
avg_mse_per_bandwidth = []

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)

for h in bandwidth_values:
    mse_per_fold = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        Y_val_pred = []

        for x in X_val:
            y_pred = local_linear_weighted_regression(X_train, Y_train, x, h)
            Y_val_pred.append(y_pred)

        Y_val_pred = np.array(Y_val_pred)
        mse = np.mean((Y_val - Y_val_pred) ** 2)
        mse_per_fold.append(mse)

    avg_mse = np.mean(mse_per_fold)
    avg_mse_per_bandwidth.append(avg_mse)

# Find the index of the minimum average MSE
best_bandwidth_idx = np.argmin(avg_mse_per_bandwidth)
best_bandwidth = bandwidth_values[best_bandwidth_idx]

print("Best bandwidth parameter:", best_bandwidth)

# Question 4c

# Using the tuned hyper-parameter h to predict x = -1.5
x_to_predict = -1.5
prediction = local_linear_weighted_regression(X, Y, x_to_predict, best_bandwidth)

print("Prediction for x = -1.5:", prediction)


# In[10]:


# Original data
X = data['data'][:, 0]
Y = data['data'][:, 1]

# Tuned hyper-parameter
best_bandwidth = 0.1  # Update with your tuned hyper-parameter

# Predict for x = -1.5
x_to_predict = -1.5
prediction = local_linear_weighted_regression(X, Y, x_to_predict, best_bandwidth)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Local Linear Weighted Regression')
plt.axvline(x=x_to_predict, color='red', linestyle='--', label='Prediction for x=-1.5')
plt.plot(X, local_linear_weighted_regression(X, Y, X, best_bandwidth), color='green', label='Local Linear Weighted Regression Line')
plt.scatter(x_to_predict, prediction[0], color='red', marker='o', s=100, label=f'Prediction: {prediction[0]:.4f}')
plt.legend()
plt.grid(True)
plt.show()


# # References 
# Class videos and sample code
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 
# https://scikit-learn.org/stable/modules/tree.html
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html#sklearn.tree.plot_tree
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
# 
# https://www.geeksforgeeks.org/locally-weighted-linear-regression-using-python/
#     
# 

# In[ ]:




