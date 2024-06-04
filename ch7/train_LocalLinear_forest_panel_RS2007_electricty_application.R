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
train_eps = as.matrix(read.csv('.//panel_SFA_RS2007_electricty_application_NN_train_eps.csv', header = TRUE))
train_u = as.matrix(read.csv('.//panel_SFA_RS2007_electricty_application_NN_train_u.csv', header = TRUE))
gaussian_copula_correlation_matrix = as.matrix(read.csv('.//panel_SFA_RS2007_electricty_application_gaussian_copula_correlation_matrix.csv', header = FALSE))
N = length(train_u)
T = ncol(train_u)

#Hyper-parameter grid search space
mtry_search_space = seq(1, ncol(train_eps), 2)
min_node_size_search_space = c(5, 10, 25, 50, 100)
honesty_fraction_search_space = c(0.5, 0.6, 0.7, 0.8)

CV_loss_matrix = matrix(ncol = 4)

#Loop over the hyper-parameter search space
i = 1
for (mtry_variable in mtry_search_space){
  for (min_node_size in min_node_size_search_space){
    for (honesty_fraction in honesty_fraction_search_space){
      
      u_hat = matrix(0, nrow(train_u), ncol(train_u))

      #Fit a LL forest model to predict u_it using all eps_i1,  ..., eps_iT
      for (t in c(1:T)){
        train_fold_yt = train_u[1:nrow(train_u), t]
        
        #Pilot Lasso to select co-variates for the local linear regression
        adaptive_weights = 1/abs(gaussian_copula_correlation_matrix[1:nrow(gaussian_copula_correlation_matrix),t])
        lasso.mod <- cv.glmnet(train_eps, train_fold_yt, alpha = 1, parallel = TRUE, standardize = TRUE, penalty.factor = adaptive_weights, nfolds=5)
        lasso.coef <- predict(lasso.mod, type = "nonzero")
        selected <- lasso.coef[,1]
        
        #The ll.lambda (ridge penalty for prediction is tuned automatically)
        LL_forest_object_t = ll_regression_forest(train_eps, 
                                                  train_fold_yt,
                                                  num.threads = n_jobs,
                                                  min.node.size = min_node_size,
                                                  honesty.fraction = honesty_fraction,
                                                  mtry = mtry_variable, 
                                                  enable.ll.split = TRUE)
        u_hat_t = predict(LL_forest_object_t, linear.correction.variables = selected)$predictions
        u_hat[1:nrow(train_u), t] = u_hat_t
      }
      
      loss = sum((u_hat - train_u)^2)/(N*T)
      
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

#Import observed empirical data 
test_eps = as.matrix(read.csv('.//panel_SFA_RS2007_electricty_application_NN_test_eps.csv', header = TRUE))
N = 72
T = ncol(test_eps)

u_hat = matrix(NaN, N, T)
V_u_hat = matrix(NaN, N, T)
#Fit a LL forest model to predict u_it using all eps_i1,  ..., eps_iT
for (t in c(1:T)){
  train_ut = train_u[1:nrow(train_u), t]
  
  #Pilot Lasso to select covariates for the local linear regression
  adaptive_weights = 1/abs(gaussian_copula_correlation_matrix[1:nrow(gaussian_copula_correlation_matrix),t])
  lasso.mod <- cv.glmnet(train_eps, train_ut, alpha = 1, parallel = TRUE, standardize = TRUE, penalty.factor = adaptive_weights, nfolds=5)
  lasso.coef <- predict(lasso.mod, type = "nonzero")
  selected <- lasso.coef[,1]

  #The ll.lambda (ridge penalty for prediction is tuned automatically)
  LL_forest_object_t = ll_regression_forest(train_eps, 
                                            train_ut,
                                            num.threads = n_jobs,
                                            min.node.size = best_params[2],
                                            honesty.fraction = best_params[3],
                                            mtry = best_params[1],
                                            enable.ll.split = TRUE,
                                            seed = 1234)

  #remove rows with nan
  n_NaNs = N - sum(complete.cases(test_eps))
  test_eps = test_eps[complete.cases(test_eps),]
  
  results_t.llf.var = predict(LL_forest_object_t, test_eps, estimate.variance = TRUE, linear.correction.variables = selected)
  u_hat_t = results_t.llf.var$predictions
  V_u_hat_t = results_t.llf.var$variance.estimates
  u_hat[1:length(u_hat_t), t] = u_hat_t
  V_u_hat[1:length(V_u_hat_t), t] = V_u_hat_t
}

#Save the MSE in the simulation results file 
write.csv(u_hat, './/RS2007_electricity_LLF_Gaussian_copula_u_hat.csv', row.names = FALSE)
write.csv(V_u_hat, './/RS2007_electricity_LLF_Gaussian_copula_V_u_hat.csv', row.names = FALSE)
