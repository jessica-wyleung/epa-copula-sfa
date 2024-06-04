list.of.packages <- c('grf', 'parallel', 'glmnet', 'doParallel')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="http://cran.us.r-project.org")
library('grf')
library('parallel')
library('glmnet')
library('doParallel')

n_jobs = detectCores()
registerDoParallel(cores=n_jobs)

### Hyper-parameter optimization ###
#Import data
train_eps_W = as.matrix(read.csv('.//cross_sectional_SFA_RS2007_electricity_application_NN_train_eps_W.csv', header = TRUE))
train_u = as.matrix(read.csv('.//cross_sectional_SFA_RS2007_electricity_application_NN_train_u.csv', header = TRUE))
gaussian_copula_correlation_matrix = as.matrix(read.csv('.//cross_sectional_SFA_RS2007_electricity_application_gaussian_copula_correlation_matrix.csv', header = FALSE))
n = length(train_u)
n_inputs = 3

#Hyper-parameter grid search space
mtry_search_space = seq(1, ncol(train_eps_W), 2)
min_node_size_search_space = c(5, 10, 25, 50, 100)
honesty_fraction_search_space = c(0.5, 0.6, 0.7, 0.8)

CV_loss_matrix = matrix(ncol = 4)

#Pilot Lasso to select covariates for the local linear regression
adaptive_weights = 1/abs(gaussian_copula_correlation_matrix[1:nrow(gaussian_copula_correlation_matrix),1])
lasso.mod <- cv.glmnet(train_eps_W, train_u, alpha = 1, parallel = TRUE, standardize = TRUE, penalty.factor = adaptive_weights, nfolds=5)
lasso.coef <- predict(lasso.mod, type = "nonzero")
selected <- lasso.coef[,1]

#Loop over the hyper-parameter search space
i = 1
for (mtry_variable in mtry_search_space){
  for (min_node_size in min_node_size_search_space){
    for (honesty_fraction in honesty_fraction_search_space){
      
      #Fit a LL forest model 
      LL_forest_object = ll_regression_forest(train_eps_W, 
                                              train_u,
                                              num.trees = 2000,
                                              num.threads = n_jobs,
                                              min.node.size = min_node_size,
                                              honesty.fraction = honesty_fraction,
                                              mtry = mtry_variable, 
                                              enable.ll.split = TRUE)
      u_hat = predict(LL_forest_object, linear.correction.variables = selected)$predictions
      loss = mean((u_hat - train_u)^2)
      
      if (i==1){
        CV_loss_matrix[i,1:3] = c(mtry_variable, min_node_size, honesty_fraction)
        CV_loss_matrix[i,ncol(CV_loss_matrix)] = loss
      } else{
        CV_loss_matrix = rbind(CV_loss_matrix, c(mtry_variable, min_node_size, honesty_fraction, loss))
      }
      i = i+1
    }
  }
}

best_CV_loss = min(CV_loss_matrix[,ncol(CV_loss_matrix)])
best_CV_loss_idx = which(CV_loss_matrix[,ncol(CV_loss_matrix)] == best_CV_loss)
best_params = CV_loss_matrix[best_CV_loss_idx,1:ncol(CV_loss_matrix)-1]
if (!is.null(nrow(best_params))){
  sums = rowSums(best_params)
  best_idx = which(sums == min(sums))
  best_params = best_params[best_idx,]
}

#Import data
test_eps_W = as.matrix(read.csv('.//cross_sectional_SFA_RS2007_electricity_application_NN_test_eps_W.csv', header = TRUE))

#Fit a LL forest model 
LL_forest_object = ll_regression_forest(train_eps_W, 
                                        train_u,
                                        num.trees = 2000,
                                        num.threads = n_jobs,
                                        min.node.size = best_params[2],
                                        honesty.fraction = best_params[3],
                                        mtry = best_params[1],
                                        enable.ll.split = TRUE,
                                        seed = 1234)

results.llf.var = predict(LL_forest_object, test_eps_W, estimate.variance = TRUE, linear.correction.variables = selected)
u_hat = results.llf.var$predictions
V_u_hat = results.llf.var$variance.estimates

results = cbind(u_hat, V_u_hat)

#Save the MSE in the simulation results file 
write.csv(results, './/LLF_Gaussian_copula_u_hat.csv', row.names = FALSE)
