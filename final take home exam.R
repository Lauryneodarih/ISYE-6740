# Load required libraries
library(caret)
library(e1071)
library(glmnet)
library(gam)
library(randomForest)
library(nnet)
library(xgboost)
library(mgcv)
library(boot)
library(class)

# Read Training Data
data <- read.table(file = "C:/Users/laury/OneDrive/Documents/7406test.csv", sep = ",", header = TRUE)
rawdata <- read.table(file = "C:/Users/laury/OneDrive/Documents/7406train.csv", sep = ",", header = TRUE)

# Extract relevant columns
X1 <- rawdata[, 1]
X2 <- rawdata[, 2]
Y_values <- rawdata[, 3:202]

# Calculate muhat = E(Y) and Vhat = Var(Y)
muhat <- apply(Y_values, 1, mean)
Vhat <- apply(Y_values, 1, var)

# Construct dataframe
df <- data.frame(X1 = X1, X2 = X2, muhat = muhat, Vhat = Vhat)

# Exploratory Data Analysis
data0 <- data.frame(X1 = X1, X2 = X2, muhat = muhat, Vhat = Vhat)

# Scatter plots
pairs(data0[, c("X1", "X2", "muhat", "Vhat")], main = "Scatterplot Matrix")

# Correlation matrix
cor_matrix <- cor(data0)
corrplot::corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
                   tl.col = "black", tl.srt = 45)

# Density plots
par(mfrow = c(2, 2))
hist(data0$X1, main = "Histogram of X1", xlab = "X1", col = "lightblue", border = "black")
hist(data0$X2, main = "Histogram of X2", xlab = "X2", col = "lightgreen", border = "black")
hist(data0$muhat, main = "Histogram of muhat", xlab = "muhat", col = "lightcoral", border = "black")
hist(data0$Vhat, main = "Histogram of Vhat", xlab = "Vhat", col = "lightgoldenrodyellow", border = "black")

# Scatter plot of X1 vs. X2 colored by muhat
plot(data0$X1, data0$X2, col = data0$muhat, pch = 19, main = "X1 vs. X2 (Colored by muhat)",
     xlab = "X1", ylab = "X2")
legend("topright", legend = "muhat", fill = colorRampPalette(c("blue", "red"))(100)[100 * (data0$muhat - min(data0$muhat)) / (max(data0$muhat) - min(data0$muhat)) + 1])

# Pair plot of X1, X2, muhat, Vhat
pairs(data0[, c("X1", "X2", "muhat", "Vhat")], main = "Pair Plot")

# Correlation between X1, X2, muhat, Vhat
cor_X1_muhat <- cor(data0$X1, data0$muhat)
cor_X2_muhat <- cor(data0$X2, data0$muhat)
cor_X1_Vhat <- cor(data0$X1, data0$Vhat)
cor_X2_Vhat <- cor(data0$X2, data0$Vhat)

cat("Correlation between X1 and muhat:", cor_X1_muhat, "\n")
cat("Correlation between X2 and muhat:", cor_X2_muhat, "\n")
cat("Correlation between X1 and Vhat:", cor_X1_Vhat, "\n")
cat("Correlation between X2 and Vhat:", cor_X2_Vhat, "\n")

# Test/train splits
set.seed(123)  # for reproducibility
trainflag <- sample(1:nrow(df), 0.90 * nrow(df))
train <- df[trainflag, ]
test <- df[-trainflag, ]

# Generate sample predictions (replace with your actual predictions)
set.seed(123)  # for reproducibility


# Linear Regression Model for muhat
lm_model_mu <- lm(muhat ~ X1 + X2, data = train)
lm_pred_mu <- predict(lm_model_mu, newdata = test, interval = "prediction")

# Linear Regression Model for Vhat
lm_model_var <- lm(Vhat ~ X1 + X2, data = train)
lm_pred_var <- predict(lm_model_var, newdata = test, interval = "prediction")

# SVM Model for muhat
svm_model_mu <- svm(muhat ~ X1 + X2, data = train)
svm_pred_mu <- predict(svm_model_mu, newdata = test)

# SVM Model for Vhat
svm_model_var <- svm(Vhat ~ X1 + X2, data = train)
svm_pred_var <- predict(svm_model_var, newdata = test)

# Generalized Additive Model (GAM) for muhat
gam_model_mu <- gam(muhat ~ s(X1) + s(X2), data = train)
gam_pred_mu <- predict(gam_model_mu, newdata = test, type = "response")

# Generalized Additive Model (GAM) for Vhat
gam_model_var <- gam(Vhat ~ s(X1) + s(X2), data = train)
gam_pred_var <- predict(gam_model_var, newdata = test, type = "response")

# Neural Network Model for muhat
#nn_model_mu <- neuralnet(muhat ~ X1 + X2, data = train, linear.output = TRUE)
#nn_pred_mu <- predict(nn_model_mu, newdata = test)

# Neural Network Model for Vhat
#nn_model_var <- neuralnet(Vhat ~ X1 + X2, data = train, linear.output = TRUE)
#nn_pred_var <- predict(nn_model_var, newdata = test)

# Random Forest Model for muhat
rf_model_mu <- randomForest(muhat ~ X1 + X2, data = train)
rf_pred_mu <- predict(rf_model_mu, newdata = test)

# Random Forest Model for Vhat
rf_model_var <- randomForest(Vhat ~ X1 + X2, data = train)
rf_pred_var <- predict(rf_model_var, newdata = test)

# K-Nearest Neighbors (KNN) Model for muhat
knn_model_mu <- knn(train[, c("X1", "X2")], test[, c("X1", "X2")], train$muhat, k = 5)
knn_pred_mu <- as.numeric(knn_model_mu)

# K-Nearest Neighbors (KNN) Model for Vhat
knn_model_var <- knn(train[, c("X1", "X2")], test[, c("X1", "X2")], train$Vhat, k = 5)
knn_pred_var <- as.numeric(knn_model_var)

# XGBoost Model for muhat
xgb_model_mu <- xgboost(data = as.matrix(train[, c("X1", "X2")]), label = train$muhat, nrounds = 10)
xgb_pred_mu <- predict(xgb_model_mu, as.matrix(test[, c("X1", "X2")]))

# XGBoost Model for Vhat
xgb_model_var <- xgboost(data = as.matrix(train[, c("X1", "X2")]), label = train$Vhat, nrounds = 10)
xgb_pred_var <- predict(xgb_model_var, as.matrix(test[, c("X1", "X2")]))

# Perform cross-validation
for (i in 1:num_folds) {
  # Subset the data for the current fold
  fold_indices <- which(folds$Fold != i)
  fold_train <- train[fold_indices, ]
  fold_test <- train[-fold_indices, ]
  
  # Check if there are enough observations for GAM
  if (nrow(fold_train) < 3) {
    warning("Not enough observations for GAM in fold ", i)
    next  # Skip to the next iteration
  }
  
  # Check for missing values in fold_train
  if (any(is.na(fold_train$muhat)) || any(is.na(fold_train$Vhat)) ||
      any(is.na(fold_train$X1)) || any(is.na(fold_train$X2))) {
    warning("Missing values detected in fold ", i)
    # Handle missing values (impute or exclude)
    # Example: fold_train <- na.omit(fold_train)
  }
  
  # Fit GAM model for the current fold for muhat
  fold_gam_model_mu <- gam(muhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_mu <- predict(fold_gam_model_mu, newdata = fold_test, type = "response")
  
  # Fit GAM model for the current fold for Vhat
  fold_gam_model_var <- gam(Vhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_var <- predict(fold_gam_model_var, newdata = fold_test, type = "response")
  
  # Store predictions for the current fold
  gam_pred_mu_cv[fold_indices] <- fold_pred_mu
  gam_pred_var_cv[fold_indices] <- fold_pred_var
}

# Calculate MSE for cross-validated GAM model
mse_gam_mu_cv <- mean((train$muhat - gam_pred_mu_cv)^2)
mse_gam_var_cv <- mean((train$Vhat - gam_pred_var_cv)^2)

# Print MSE results for GAM
cat("GAM Model - MSE for muhat (cross-validated):", mse_gam_mu_cv, "\n")
cat("GAM Model - MSE for Vhat (cross-validated):", mse_gam_var_cv, "\n")










# Perform cross-validation
for (i in 1:num_folds) {
  # Subset the data for the current fold
  fold_indices <- which(folds$Fold != i)
  fold_train <- train[fold_indices, ]
  fold_test <- train[-fold_indices, ]
  
  # Check if there are enough observations for GAM
  if (nrow(fold_train) < 3) {
    warning("Not enough observations for GAM in fold ", i)
    next  # Skip to the next iteration
  }
  
  # Check for missing values in fold_train
  if (any(is.na(fold_train$muhat)) || any(is.na(fold_train$Vhat)) ||
      any(is.na(fold_train$X1)) || any(is.na(fold_train$X2))) {
    warning("Missing values detected in fold ", i)
    # Handle missing values (impute or exclude)
    # Example: fold_train <- na.omit(fold_train)
  }
  
  # Fit GAM model for the current fold for muhat
  fold_gam_model_mu <- gam(muhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_mu <- predict(fold_gam_model_mu, newdata = fold_test, type = "response")
  
  # Fit GAM model for the current fold for Vhat
  fold_gam_model_var <- gam(Vhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_var <- predict(fold_gam_model_var, newdata = fold_test, type = "response")
  
  # Store predictions for the current fold
  gam_pred_mu_cv[fold_indices] <- fold_pred_mu
  gam_pred_var_cv[fold_indices] <- fold_pred_var
}

# Calculate MSE for cross-validated GAM model
mse_gam_mu_cv <- mean((train$muhat - gam_pred_mu_cv)^2)
mse_gam_var_cv <- mean((train$Vhat - gam_pred_var_cv)^2)

# Print MSE results for GAM
cat("GAM Model - MSE for muhat (cross-validated):", mse_gam_mu_cv, "\n")
cat("GAM Model - MSE for Vhat (cross-validated):", mse_gam_var_cv, "\n")
# Cross-validation for GAM
num_folds <- 5
folds <- createFolds(train$muhat, k = num_folds, list = TRUE)

# Create empty vectors to store predictions
gam_pred_mu_cv <- numeric(length(train$muhat))
gam_pred_var_cv <- numeric(length(train$Vhat))

# Perform cross-validation
for (i in 1:num_folds) {
  # Subset the data for the current fold
  fold_indices <- which(folds$Fold != i)
  fold_train <- train[fold_indices, ]
  fold_test <- train[-fold_indices, ]
  
  # Fit GAM model for the current fold for muhat
  fold_gam_model_mu <- gam(muhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_mu <- predict(fold_gam_model_mu, newdata = fold_test, type = "response")
  
  # Fit GAM model for the current fold for Vhat
  fold_gam_model_var <- gam(Vhat ~ s(X1) + s(X2), data = fold_train)
  fold_pred_var <- predict(fold_gam_model_var, newdata = fold_test, type = "response")
  
  # Store predictions for the current fold
  gam_pred_mu_cv[fold_indices] <- fold_pred_mu
  gam_pred_var_cv[fold_indices] <- fold_pred_var
}

# Calculate MSE for cross-validated GAM model
mse_gam_mu_cv <- mean((train$muhat - gam_pred_mu_cv)^2)
mse_gam_var_cv <- mean((train$Vhat - gam_pred_var_cv)^2)

# Print MSE results for GAM
cat("GAM Model - MSE for muhat:", mse_gam_mu_cv, "\n")
cat("GAM Model - MSE for Vhat:", mse_gam_var_cv, "\n")

# Cross-validation for Random Forest
ctrl <- trainControl(method = "cv", number = 5)

# Random Forest Model for muhat
rf_cv_mu <- train(muhat ~ X1 + X2, data = train, method = "rf", trControl = ctrl)
rf_pred_mu_cv <- predict(rf_cv_mu, newdata = test)

# Random Forest Model for Vhat
rf_cv_var <- train(Vhat ~ X1 + X2, data = train, method = "rf", trControl = ctrl)
rf_pred_var_cv <- predict(rf_cv_var, newdata = test)

# Calculate MSE for cross-validated Random Forest models
mse_rf_mu_cv <- mean((test$muhat - rf_pred_mu_cv)^2)
mse_rf_var_cv <- mean((test$Vhat - rf_pred_var_cv)^2)

# Print MSE results for Random Forest
cat("Random Forest Model - MSE for muhat:", mse_rf_mu_cv, "\n")
cat("Random Forest Model - MSE for Vhat:", mse_rf_var_cv, "\n")

# Set up cross-validation control
ctrl <- trainControl(method = "cv", number = 5)

# XGBoost Model for muhat with cross-validation
xgb_cv_mu <- train(muhat ~ X1 + X2, data = train, method = "xgbTree", trControl = ctrl)
xgb_pred_mu_cv <- predict(xgb_cv_mu, newdata = test)

# XGBoost Model for Vhat with cross-validation
xgb_cv_var <- train(Vhat ~ X1 + X2, data = train, method = "xgbTree", trControl = ctrl)
xgb_pred_var_cv <- predict(xgb_cv_var, newdata = test)

# Calculate MSE for cross-validated XGBoost models
mse_xgb_mu_cv <- mean((test$muhat - xgb_pred_mu_cv)^2)
mse_xgb_var_cv <- mean((test$Vhat - xgb_pred_var_cv)^2)

# Print MSE results for XGBoost
cat("XGBoost Model - MSE for muhat (cross-validated):", mse_xgb_mu_cv, "\n")
cat("XGBoost Model - MSE for Vhat (cross-validated):", mse_xgb_var_cv, "\n")


# Cross-validation for SVM
svm_cv_mu <- train(muhat ~ X1 + X2, data = train, method = "svmRadial", trControl = ctrl)
svm_pred_mu_cv <- predict(svm_cv_mu, newdata = test)

svm_cv_var <- train(Vhat ~ X1 + X2, data = train, method = "svmRadial", trControl = ctrl)
svm_pred_var_cv <- predict(svm_cv_var, newdata = test)

# Calculate MSE for cross-validated SVM models
mse_svm_mu_cv <- mean((test$muhat - svm_pred_mu_cv)^2)
mse_svm_var_cv <- mean((test$Vhat - svm_pred_var_cv)^2)

# Print MSE results for SVM
cat("SVM Model - MSE for muhat:", mse_svm_mu_cv, "\n")
cat("SVM Model - MSE for Vhat:", mse_svm_var_cv, "\n")

# Cross-validation for KNN
knn_cv_mu <- train(muhat ~ X1 + X2, data = train, method = "knn", trControl = ctrl)
knn_pred_mu_cv <- predict(knn_cv_mu, newdata = test)

knn_cv_var <- train(Vhat ~ X1 + X2, data = train, method = "knn", trControl = ctrl)
knn_pred_var_cv <- predict(knn_cv_var, newdata = test)

# Calculate MSE for cross-validated KNN models
mse_knn_mu_cv <- mean((test$muhat - knn_pred_mu_cv)^2)
mse_knn_var_cv <- mean((test$Vhat - knn_pred_var_cv)^2)

# Print MSE results for KNN
cat("K-Nearest Neighbors Model - MSE for muhat:", mse_knn_mu_cv, "\n")
cat("K-Nearest Neighbors Model - MSE for Vhat:", mse_knn_var_cv, "\n")

# Cross-validation for Linear Regression
lm_cv_mu <- train(muhat ~ X1 + X2, data = train, method = "lm", trControl = ctrl)
lm_pred_mu_cv <- predict(lm_cv_mu, newdata = test)

lm_cv_var <- train(Vhat ~ X1 + X2, data = train, method = "lm", trControl = ctrl)
lm_pred_var_cv <- predict(lm_cv_var, newdata = test)

# Calculate MSE for cross-validated Linear Regression models
mse_lm_mu_cv <- mean((test$muhat - lm_pred_mu_cv)^2)
mse_lm_var_cv <- mean((test$Vhat - lm_pred_var_cv)^2)

# Print MSE results for Linear Regression
cat("Linear Regression Model - MSE for muhat:", mse_lm_mu_cv, "\n")
cat("Linear Regression Model - MSE for Vhat:", mse_lm_var_cv, "\n")


# Combine predictions into a data matrix
predictions_matrix <- cbind(
  round(lm_pred_mu, 6), round(lm_pred_var, 6),
  round(svm_pred_mu, 6), round(svm_pred_var, 6),
  round(knn_pred_mu, 6), round(knn_pred_var, 6),
  round(rf_pred_mu, 6), round(rf_pred_var, 6),
  round(xgb_pred_mu, 6), round(xgb_pred_var, 6),
  round(gam_pred_mu_cv, 6), round(gam_pred_var_cv, 6)
)

# Create a data frame with rounded predictions
predictions_df <- as.data.frame(predictions_matrix)

# Add the first two columns from the test data
predictions_df <- cbind(test$X1, test$X2, predictions_df)
#predictions_df <- cbind(test_data$X1, test_data$X2, predictions_df)


# Extract the required columns (X1, X2, estimated mean, estimated variance)
predictions_df <- predictions_df[, c(1, 2, 11, 12)]

# Save the data matrix to a CSV file without headers
write.csv(predictions_df, file = "predictions_matrix.csv", row.names = FALSE, col.names = FALSE)




