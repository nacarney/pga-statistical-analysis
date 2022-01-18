# The code below is mainly split into 2 sections. The first section covers the majority of the functionality of the code, and creates 
# the first set of Ridge, LASSO and Elastic Net GLM's. After the '#### RUN TO HERE ####' comment, another three of these models are 
# created having removed two variables contained in the first three models.


## LOADING IN PACKAGES 

#install.packages("ROCR")
library(ROCR)

#install.packages("pROC")
library('pROC')

library(ggplot2)

library(bayesplot)
theme_set(bayesplot::theme_default())

library(MASS)

require(dplyr)
#install.packages('caret', dependencies = T)
#install.packages("lubridate")
library('caret')

library(stats4)

#For reg subsets variable selection
library(leaps)

library(glmnet)

#install.packages('TeachingDemos')
library(TeachingDemos)

require(ggplot2)

require(sandwich)

require(msm)

#Package for correlation heatmap
#install.packages('reshape2') 
library(reshape2)

#Package for dropping levels
# install.packages('gdata')
library(gdata)

#Package for histogram in ggplot2
#install.packages('plyr')
library(plyr)

# Below is for printing two boxplots side by side
# Source: 
# https://stackoverflow.com/questions/1249548/side-by-side-plots-with-ggplot2
# install.packages('gridExtra')
require(gridExtra)

#install.packages('glmnet')
library(glmnet)

library(stats4)

#install.packages('expss')
library(expss)

#Below is for Exploratory analysis, sourced from https://boxuancui.github.io/DataExplorer/
#install.packages('data.table')
#install.packages('DataExplorer')
library('DataExplorer')

## START OF CODE

#Setting Working Directory
setwd('/Users/nathancarney/Documents/College/3rd Year/Stats Analysis/Final Project')

# setting seed so random splitting of data is consistent
set.seed(130)

## PART 1 -  READING IN THE DATA

raw <- as.data.frame(read.csv('pgaTourData.csv', header = TRUE))

# Getting an overall picture of the data using the DataExplorer Package
#introduce(raw)
#plot_intro(raw)

#introduce(raw)

# Setting NA values in the Wins column to 0 in the raw dataset
index <- which(is.na(raw$Wins), arr.ind = TRUE)

raw[index, 11] = 0

# Reading in the data frame
df <- as.data.frame(read.csv('pgaTourData.csv', header = TRUE))

## PART 2 - CLEANING THE DATA

# changing null win / top 10 values to 0's 

index <- which(is.na(df$Wins), arr.ind = TRUE)

df[index, 11] = 0

top_index <- which(is.na(df$Top.10), arr.ind = TRUE)

df[top_index, 12] = 0

#Number of NA's (No null values in remaining columns)

#sum(is.na(df))

# removing NA's

df <- na.omit(df)

# Removing ',' from Points and Money, as well as "$" from Money column, and making both columns integers

df$Rounds <- as.numeric(df$Rounds)

df$Points <- as.numeric(gsub(",","",df$Points))

df$Money <- gsub("\\$", "", df$Money)

df$Money <- gsub(",","",df$Money)

df$Money <- as.numeric(df$Money)

#Creating Table showing spread of Golfer's wins
#table(df$Wins)

# Removing Name and Year Column as these will not contribute to model (ID Columns)

df <- df[,-1]
df <- df[,-3]

# Removing Money, Top 10's and Points columns as these are directly associated with Wins
# Also removing SG Total and Average Score as these are an aggregate of other SG stats

df <- df[,c(-7, -8, -10, -12, -16)]

# Renaming columns for better presentation + consistency (Average = Avg)

colnames(df) = c("Rounds.Played", "Fairway.Percentage", "Avg.Driving.Distance", "GIR.Percentage", "Avg.Putts.Per.Round",
                 "Scrambling.Percentage", "Wins", "Avg.SG.Putting", "Avg.SG.OTT", "Avg.SG.APR",
                 "Avg.SG.ARG")

## OUTLIERS
# 2% of players had more than 1 win in a single season
#(nrow(df[which(df$Wins>1),]))/nrow(df)

# Removing outliers (players with one or more wins in a single season)
index <- which((df$Wins == 2) | (df$Wins == 3) | (df$Wins == 4) | (df$Wins == 5) , arr.ind = TRUE)

# Dataset now has 1633 observations
df <- df[-index,]

## PART 3 - EXPLORATORY ANALYSIS / VARIABLE SELECTION 

# CORRELATION HEATMAP

# Creating a correlation heatmap using the reshape2 package
# Code source: 
# http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization

cormat <- round(cor(df),2)

#Melting the correlation matrix so that it is in the correct format
melted_cormat <- melt(cormat)

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)

melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Heatmap

reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}

# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)

upper_tri <- get_upper_tri(cormat)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal") +
    guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# Converting Wins to a Factor for the model, new levels are 0 and 1 as players with 2 or more wins have been removed 

df$Wins <- factor(df$Wins, levels = c("0", "1"))

# Graphing the relationships between variables further by using the pairs function, with each set of observations color coded based on Wins
#pairs(df, col=df$Wins)

df$Wins <- revalue(df$Wins, c("0"="No_Win", "1"="Win"))

# DATA EXPLORER ANALYSIS

# View basic description for PGA data
introduce(df)

# Plot basic description for PGA data
plot_intro(df)

plot_histogram(df)

# View bivariate continuous distribution based on `wins`
plot_boxplot(df, by = "Wins")

plot_prcomp(df, variance_cap = 0.9)

# REGSUBSETS VISUALISATION

# RegSubsets Visualisation of Leaps algorithm exhaustive search result indicates that 
# Intercept, Rounds Played, Putts per Round, Scrambling percentage, Strokes Gained OTT, Strokes Gained APR 
# and Average Strokes Gained ARG are the most significant variables in terms of their contribution to 
# reducing the BIC of the model, however after removing these variables and recreating the models 
# it was found that both the AIC and AUC decreased for Ridge, LASSO and Elastic Net. Due to the fact that this model is 
# focused on prediction, AIC and AUC are more suitable benchmarks to adhere to than BIC alone. 
# There was also a decrease in the accuracy of the ridge and LASSO models when these variables were removed.

# Hence, the variables were not removed in the final model.
?regsubsets
plot(regsubsets(Wins ~ ., data = df, method = "exhaustive", nbest = 1, nvmax = 10), main = "Significant Variables")
a <- regsubsets(Wins ~ ., data = df, method = "exhaustive", nbest = 1)

## PART 4 - BUILDING REGRESSION MODELS: RIDGE, LASSO AND ELASTIC NET

# Generate training dataset
data <- sort(sample(nrow(df), nrow(df)*.7))

# Put 70% into training dataset
train <-df[data,]

#Splitting the remaining 30% of data into test and validation sets, each being 15% of full dataset 
leftover_data <-df[-data,]
test_val_index <- sort(sample(nrow(leftover_data), nrow(leftover_data)*.5))
test <-leftover_data[test_val_index,]
validation <- leftover_data[-test_val_index,]

# creating general model
model <- glm(formula = Wins ~ ., family = binomial, data = train)

summary(model)

model.fit <- predict(model, test, type="response")
plot(model.fit)

#create predictor matrix and outcome vector for training, validation, and test data

train.X <- as.matrix(within(train, rm(Wins)))
val.X <- as.matrix(within(validation, rm(Wins)))
test.X <- as.matrix(within(test, rm(Wins)))
train.y <- train$Wins
val.y <- validation$Wins
test.y <- test$Wins

## PARAMETER CROSS-VALIDATION / TUNING FOR RIDGE AND LASSO MODELS

#cross-validate to tune lambda for ridge and lasso
cvridge <- cv.glmnet(train.X, train.y, family="binomial", alpha=0, nlambda=20, type.measure="auc")
cvlasso <- cv.glmnet(train.X, train.y, family="binomial", alpha=1, nlambda=20, type.measure="auc")

#fit models with final lambda
ridgemod <- glmnet(train.X, train.y, family="binomial", alpha = 0, lambda = cvridge$lambda.1se)
lassomod <- glmnet(train.X, train.y, family="binomial", alpha = 1, lambda = cvlasso$lambda.1se)
train.stdX <-scale(train.X)

?cv.glmnet
  
## TUNING PARAMETERS (ALPHA AND LAMBDA) FOR ELASTIC NET MODEL USING K-FOLD CV (5 FOLDS)

# Set training control
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              verboseIter = TRUE)

# Train the model
elastic_grid <- train(train.stdX, train.y,
                      method = "glmnet",
                      tuneLength = 25,
                      trControl = train_control,
                      metric= "ROC",
                      family = "binomial",
                      standardize = FALSE)

#fit the model with best lambda and alpha
elasticmod <- glmnet(train.X, train.y, family="binomial", alpha = elastic_grid$bestTune$alpha, 
                     lambda = elastic_grid$bestTune$lambda)

# Looking at Coefs for final models
Intercepts <- cbind(ridgemod$a0,lassomod$a0,elasticmod$a0)
Coefs <- cbind(ridgemod$beta,lassomod$beta, elasticmod$beta)
Betas <-rbind(Intercepts, Coefs)
rownames(Betas)[1] = "(Intercept)"
colnames(Betas) = c("Ridge", "Lasso", "Elastic Net")
Betas

#fit ridge, lasso and elastic models
fit.ridge <- predict(ridgemod, val.X, type="response")
fit.lasso <- predict(lassomod, val.X, type="response")
fit.elastic <- predict(elasticmod, val.X, type="response")

## THRESHOLD SELECTION AND AUC -> ROCR CURVES FOR ALL 3 MODELS

# RIDGE

# ROCR Plot
ROCRfit.ridge = prediction(fit.ridge, val.y)
ROCRperf.tr.ridge = performance(ROCRfit.ridge, "tpr", "fpr")

plot(ROCRperf.tr.ridge, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                        main = "ROCR Plot for PGA Tour Win Logit Probability - RIDGE")

# ROC
ridge.roc <- roc(val.y, as.vector(fit.ridge))
coords(ridge.roc, "best", best.method="youden", transpose=TRUE)

# AUC of 0.8186
auc(ridge.roc)

# Method using visualisation over Grid
# accuracy 
cutoffs <- seq(min(fit.ridge),max(fit.ridge),(max(fit.ridge)-min(fit.ridge))/100)
accuracy <- NULL

for (i in seq(along = cutoffs)){
  prediction <- ifelse(fit.ridge >= cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  accuracy <- c(accuracy,length(which(val.y ==prediction))/length(prediction)*100)
}

ridge_ROCR_plot <- plot(cutoffs, accuracy, pch =19,type='l',col= "steelblue",
     main ="Logistic Regression - Ridge", xlab="Cutoff Level", ylab = "Accuracy %")

# LASSO

# ROCR Plot 
ROCRfit.lasso = prediction(fit.lasso, val.y)
ROCRperf.tr.lasso = performance(ROCRfit.lasso, "tpr", "fpr")

plot(ROCRperf.tr.lasso, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                        main = "ROCR Plot for PGA Tour Win Logit Probability - LASSO")

# ROC
lasso.roc <- roc(val.y, as.vector(fit.lasso))
coords(lasso.roc, "best", best.method="youden", transpose=TRUE)

# AUC of 0.8094
auc(lasso.roc)

# Method using visualisation over Grid
# accuracy 
lasso_cutoffs <- seq(min(fit.lasso),max(fit.lasso),(max(fit.lasso)-min(fit.lasso))/100)
lasso_accuracy <- NULL

for (i in seq(along = lasso_cutoffs)){
  lasso_prediction <- ifelse(fit.lasso >= lasso_cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  lasso_accuracy <- c(lasso_accuracy,length(which(val.y == lasso_prediction))/length(lasso_prediction)*100)
}

# Accuracy Plot
plot(lasso_cutoffs, lasso_accuracy, pch =19,type='l',col= "steelblue",
     main ="Logistic Regression - LASSO", xlab="Cutoff Level", ylab = "Accuracy %")

# ELASTIC

# ROCR Plot
ROCRfit.elastic = prediction(fit.elastic, val.y)
ROCRperf.tr.elastic = performance(ROCRfit.elastic, "tpr", "fpr")

elastic_rocplot <- plot(ROCRperf.tr.elastic, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                        main = "ROCR Plot for PGA Tour Win Logit Probability - ELASTIC NET")

# ROC
elastic.roc <- roc(val.y, as.vector(fit.elastic))
coords(elastic.roc, "best", best.method="youden", transpose=TRUE)

?coords
# AUC of 0.8099
auc(elastic.roc)

# Method using visualisation over Grid
# accuracy 
elastic_cutoffs <- seq(min(fit.elastic),max(fit.elastic),(max(fit.elastic)-min(fit.elastic))/100)
elastic_accuracy <- NULL

for (i in seq(along = elastic_cutoffs)){
  elastic_prediction <- ifelse(fit.elastic >= elastic_cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  elastic_accuracy <- c(elastic_accuracy,length(which(val.y == elastic_prediction))/length(elastic_prediction)*100)
}

# Accuracy Plot
plot(elastic_cutoffs, elastic_accuracy, pch =19,type='l',col= "steelblue",
     main ="Logistic Regression - ELASTIC", xlab="Cutoff Level", ylab = "Accuracy %")

## FINDING OPTIMAL THRESHOLDS USING YOUDEN METHOD IN COORDS FUNCTION

# Ridge Threshold = 0.1555
thresh.r <- coords(roc(val.y, as.vector(fit.ridge)), "best", best.method="youden", transpose=TRUE, ret="threshold")

# Lasso Threshold = 0.1923
thresh.l <- coords(roc(val.y, as.vector(fit.lasso)), "best", best.method="youden", transpose=TRUE, ret="threshold")

# Elastic Threshold = 0.1565
thresh.e <- coords(roc(val.y, as.vector(fit.elastic)), "best", best.method="youden", transpose=TRUE, ret="threshold")

## FINDING OPTIMAL THRESHOLDS USING CLOSEST.TOPLEFT METHOD IN COORDS FUNCTION

# Ridge Threshold = 0.1618
close_thresh.r <- coords(roc(val.y, as.vector(fit.ridge)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

# Lasso Threshold = 0.1923
close_thresh.l <- coords(roc(val.y, as.vector(fit.lasso)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

# Elastic Threshold = 0.1565
close_thresh.e <- coords(roc(val.y, as.vector(fit.elastic)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

## CHOICE OF THRESHOLD

# There was total agreement between the thresholds from both the topleft and youden methods for the lasso and elastic net models.
# However, the topleft method had a higher threshold for the ridge method which is favourable as the model favours a lower false positive
# rate and higher sensitivity. It can be seen below that the closest topleft thresholds yielded better results for the accuracy of the ridge
# model, so these were used

## PART 5 TESTING FINAL MODELS

#predict classifications in test data
final.r <- predict(ridgemod, test.X, type="response")
final.l <- predict(lassomod, test.X, type="response")
final.e <- predict(elasticmod, test.X, type="response")

## PART 6 ASSESSING ACCURACY AND COMPLEXITY OF MODELS USING CLOSEST TOPLEFT THRESHOLDS

#use caret to see various measures of performance using confusion matrices

# 1. Ridge - 69.8% accuracy, 74 misclassifications
class.ridge <- as.factor(ifelse(final.r <= close_thresh.r, "No_Win", "Win"))
ridge_confusion <- confusionMatrix(class.ridge, test.y, positive = "Win")

# 2. Lasso - 74.69% accuracy, 62 misclassifications
class.lasso <- as.factor(ifelse(final.l <= thresh.l, "No_Win", "Win"))
lasso_confusion <- confusionMatrix(class.lasso, test.y, positive = "Win")

# 3. Elastic - 68.16% accuracy, 78 misclassifications
class.elastic <- as.factor(ifelse(final.e <= thresh.e, "No_Win", "Win"))
elastic_confusion <- confusionMatrix(class.elastic, test.y, positive = "Win")

# Ridge AIC = -68.17
ridge_tLL <- ridgemod$nulldev - deviance(ridgemod)
ridge_k <- ridgemod$df
ridge_n <- ridgemod$nobs
ridge_AICc <- -ridge_tLL+2*ridge_k+2*ridge_k*(ridge_k+1)/(ridge_n-ridge_k-1)
ridge_AICc

# LASSO AIC = -95.97
lasso_tLL <- lassomod$nulldev - deviance(lassomod)
lasso_k <- lassomod$df
lasso_n <- lassomod$nobs
lasso_AICc <- -lasso_tLL+2*lasso_k+2*lasso_k*(lasso_k+1)/(lasso_n-lasso_k-1)
lasso_AICc

# Elastic AIC = -98.77
elastic_tLL <- elasticmod$nulldev - deviance(elasticmod)
elastic_k <- elasticmod$df
elastic_n <- elasticmod$nobs
elastic_AICc <- -elastic_tLL+2*elastic_k+2*elastic_k*(elastic_k+1)/(elastic_n-elastic_k-1)
elastic_AICc

## CHOICE OF MODEL

# LASSO appears to be the most favourable model, as both the 'closest topleft' and ROCR visual thresholds are in approximate 
# agreement at ~0.195, hence it is less likely that a player will be predicted to get a win compared with the other two models
# which have a lower threshold. The AUC was relatively high, and very close to that of Elastic and Ridge models. 
# The AIC was higher than the ridge model and lower than the elastic model, but was not significantly less than the elastic model. 
# It seems to represent a suitable compromise between both ridge and elastic. 

# Predicting Abraham Ancer's probability of getting a win using his stats from his 2020 season in which he did not get a win 
# on the PGA Tour (data pulled from shotlink data page - www.pga.com )
ancer <- Matrix(data = c(56, 64.44, 297.5, 66.97, 28.65, 64.77, 0.328, 0.328, 0.335, -0.93), nrow = 1, ncol = 10)

# Logit probability = 0.036, threshold for lasso is 0.1923, hence model has proven correct in this case
predict(lassomod, newx = ancer, type = "response")


# Predicting Bryson DeChambeau's probability of getting a win using his stats from his best season so far in the PGA Tour, 2018, 
# in which he won 3 times (data pulled from shotlink data page - www.pga.com )
bryson <- Matrix(data = c(78, 62.23, 305.7, 69.65, 29.18, 59.84, 0.346, 0.586, 0.556, 0.07), nrow = 1, ncol = 10)

# Logit probability = 0.3097, threshold for lasso is 0.1923, hence model has, again, proven correct in this case
predict(lassomod, newx = bryson, type = "response")




                                                #### RUN TO HERE ####


                                      #### END OF FIRST SET OF MODELS ######


      #### BELOW I HAVE REMOVED THE DRIVING DISTANCE AND FAIRWAY PERCENTAGE VARIABLES AND REPEATED THE ABOVE PROCESSES ####






## LASSO RESULTS 

# The creation of a LASSO model with all 10 variables indicated (in its coefficients) that Driving Distance and Fairway percentage 
# were not significant, the final elastic net model also indicated that Driving Distance should be fully removed and Fairway percentage 
# should take a very small value (0.0019). This is in agreement with the correlation heatmap and regsubsets visualisation. Hence, 
# these variables were removed and the models were recreated accordingly.

df <- df[,c(-2, -3)]

#Generate training dataset
data <- sort(sample(nrow(df), nrow(df)*.7))

#Put 70% into training dataset
train <-df[data,]

#Splitting the remaining 30% of data into test and validation sets, each being 15% of full dataset 
leftover_data <-df[-data,]
test_val_index <- sort(sample(nrow(leftover_data), nrow(leftover_data)*.5))
test <-leftover_data[test_val_index,]
validation <- leftover_data[-test_val_index,]

#create predictor matrix and outcome vector for training, validation, and test data

train.X <- as.matrix(within(train, rm(Wins)))
val.X <- as.matrix(within(validation, rm(Wins)))
test.X <- as.matrix(within(test, rm(Wins)))
train.y <- train$Wins
val.y <- validation$Wins
test.y <- test$Wins

## PARAMETER CROSS-VALIDATION / TUNING FOR RIDGE AND LASSO MODELS

#cross-validate to tune lambda for ridge and lasso
cvridge <- cv.glmnet(train.X, train.y, family="binomial", alpha=0, nlambda=20, type.measure="auc")
cvlasso <- cv.glmnet(train.X, train.y, family="binomial", alpha=1, nlambda=20, type.measure="auc")

#fit models with final lambda
ridgemod <- glmnet(train.X, train.y, family="binomial", alpha = 0, lambda = cvridge$lambda.1se)
lassomod <- glmnet(train.X, train.y, family="binomial", alpha = 1, lambda = cvlasso$lambda.1se)
train.stdX <-scale(train.X)

## TUNING PARAMETERS (ALPHA AND LAMBDA) FOR ELASTIC NET MODEL

# Set training control
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary,
                              verboseIter = TRUE)

# Train the model
elastic_grid <- train(train.stdX, train.y,
                      method = "glmnet",
                      tuneLength = 25,
                      trControl = train_control,
                      metric= "ROC",
                      family = "binomial",
                      standardize = FALSE)

#fit the model with best lambda and alpha
elasticmod <- glmnet(train.X, train.y, family="binomial", alpha = elastic_grid$bestTune$alpha, 
                     lambda = elastic_grid$bestTune$lambda)

# Looking at Coefs for final models
Intercepts <- cbind(ridgemod$a0,lassomod$a0,elasticmod$a0)
Coefs <- cbind(ridgemod$beta,lassomod$beta, elasticmod$beta)
Betas <-rbind(Intercepts, Coefs)
rownames(Betas)[1] = "(Intercept)"
colnames(Betas) = c("Ridge", "Lasso", "Elastic Net")
Betas

#fit ridge, lasso and elastic models
fit.ridge <- predict(ridgemod, val.X, type="response")
fit.lasso <- predict(lassomod, val.X, type="response")
fit.elastic <- predict(elasticmod, val.X, type="response")

## ROCR CURVES FOR ALL 3 MODELS TO DECIDE ON APPROPRIATE THRESHOLDS

# RIDGE

# ROCR Plot
ROCRfit.ridge = prediction(fit.ridge, val.y)
ROCRperf.tr.ridge = performance(ROCRfit.ridge, "tpr", "fpr")

plot(ROCRperf.tr.ridge, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                      main = "ROCR Plot for PGA Tour Win Logit Probability - RIDGE")

ridge.roc <- roc(val.y, as.vector(fit.ridge))
#coords(ridge.roc, "best", best.method="youden", transpose=TRUE)

# AUC of 0.7454
auc(ridge.roc)

# Method using visualisation over Grid
# accuracy 
cutoffs <- seq(min(fit.ridge),max(fit.ridge),(max(fit.ridge)-min(fit.ridge))/100)
accuracy <- NULL

for (i in seq(along = cutoffs)){
  prediction <- ifelse(fit.ridge >= cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  accuracy <- c(accuracy,length(which(val.y ==prediction))/length(prediction)*100)
}

# Accuracy Plot
plot(cutoffs, accuracy, pch =19,type='l',col= "steelblue",
                        main ="Logistic Regression - Ridge", xlab="Cutoff Level", ylab = "Accuracy %")

# LASSO

#ROCR Plot
ROCRfit.lasso = prediction(fit.lasso, val.y)
ROCRperf.tr.lasso = performance(ROCRfit.lasso, "tpr", "fpr")

plot(ROCRperf.tr.lasso, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                      main = "ROCR Plot for PGA Tour Win Logit Probability - LASSO")

# ROC
lasso.roc <- roc(val.y, as.vector(fit.lasso))
coords(lasso.roc, "best", best.method="youden", transpose=TRUE)

# AUC of 0.7273
auc(lasso.roc)

# Method using visualisation over Grid
# accuracy 
lasso_cutoffs <- seq(min(fit.lasso),max(fit.lasso),(max(fit.lasso)-min(fit.lasso))/100)
lasso_accuracy <- NULL

for (i in seq(along = lasso_cutoffs)){
  lasso_prediction <- ifelse(fit.lasso >= lasso_cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  lasso_accuracy <- c(lasso_accuracy,length(which(val.y == lasso_prediction))/length(lasso_prediction)*100)
}

# Accuracy Plot
plot(lasso_cutoffs, lasso_accuracy, pch =19,type='l',col= "steelblue",
                        main ="Logistic Regression - LASSO", xlab="Cutoff Level", ylab = "Accuracy %")

# ELASTIC

# ROCR Plot
ROCRfit.elastic = prediction(fit.elastic, val.y)
ROCRperf.tr.elastic = performance(ROCRfit.elastic, "tpr", "fpr")

plot(ROCRperf.tr.elastic, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.01), text.adj=c(-0.2,1.7), 
                        main = "ROCR Plot for PGA Tour Win Logit Probability - ELASTIC NET")

# ROC
elastic.roc <- roc(val.y, as.vector(fit.elastic))
coords(elastic.roc, "best", best.method="youden", transpose=TRUE)

# AUC of 0.7772
auc(elastic.roc)

# Method using visualisation over Grid
# accuracy 
elastic_cutoffs <- seq(min(fit.elastic),max(fit.elastic),(max(fit.elastic)-min(fit.elastic))/100)
elastic_accuracy <- NULL

for (i in seq(along = elastic_cutoffs)){
  elastic_prediction <- ifelse(fit.elastic >= elastic_cutoffs[i], "Win", "No_Win") #Predicting for cut-off
  elastic_accuracy <- c(elastic_accuracy,length(which(val.y == elastic_prediction))/length(elastic_prediction)*100)
}

# Accuracy Plot
plot(elastic_cutoffs, elastic_accuracy, pch =19,type='l',col= "steelblue",
                          main ="Logistic Regression - ELASTIC", xlab="Cutoff Level", ylab = "Accuracy %")

## FINDING OPTIMAL THRESHOLDS USING YOUDEN METHOD IN COORDS FUNCTION

# Ridge Threshold = 0.1429
thresh.r <- coords(roc(val.y, as.vector(fit.ridge)), "best", best.method="youden", transpose=TRUE, ret="threshold")

# Lasso Threshold = 0.1695
thresh.l <- coords(roc(val.y, as.vector(fit.lasso)), "best", best.method="youden", transpose=TRUE, ret="threshold")

# Elastic Threshold = 0.1928
thresh.e <- coords(roc(val.y, as.vector(fit.elastic)), "best", best.method="youden", transpose=TRUE, ret="threshold")

## FINDING OPTIMAL THRESHOLDS USING CLOSEST.TOPLEFT METHOD IN COORDS FUNCTION

# Ridge Threshold = 0.1457
close_thresh.r <- coords(roc(val.y, as.vector(fit.ridge)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

# Lasso Threshold = 0.1695
close_thresh.l <- coords(roc(val.y, as.vector(fit.lasso)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

# Elastic Threshold = 0.1928
close_thresh.e <- coords(roc(val.y, as.vector(fit.elastic)), "best", best.method="closest.topleft", transpose=TRUE, ret="threshold")

## CHOICE OF THRESHOLD

# same as in original model creation above 

## TESTING FINAL MODELS

#predict classifications in test data
final.r <- predict(ridgemod, test.X, type="response")
final.l <- predict(lassomod, test.X, type="response")
final.e <- predict(elasticmod, test.X, type="response")

## ASSESSING ACCURACY AND COMPLEXITY OF MODELS USING CLOSEST TOP LEFT THRESHOLDS

#use caret to see various measures of performance using confusion matrices

# 1. Ridge - 66.53% accuracy, 72 misclassifications
class.ridge <- as.factor(ifelse(final.r <= close_thresh.r, "No_Win", "Win"))
ridge_confusion <- confusionMatrix(class.ridge, test.y, positive = "Win")

# 2. Lasso - 71.84% accuracy, 69 misclassifications
class.lasso <- as.factor(ifelse(final.l <= thresh.l, "No_Win", "Win"))
lasso_confusion <- confusionMatrix(class.lasso, test.y, positive = "Win")

# 3. Elastic - 70.2% accuracy, 73 misclassifications
class.elastic <- as.factor(ifelse(final.e <= thresh.e, "No_Win", "Win"))
elastic_confusion <- confusionMatrix(class.elastic, test.y, positive = "Win")

# Ridge AIC = -53.52
ridge_tLL <- ridgemod$nulldev - deviance(ridgemod)
ridge_k <- ridgemod$df
ridge_n <- ridgemod$nobs
ridge_AICc <- -ridge_tLL+2*ridge_k+2*ridge_k*(ridge_k+1)/(ridge_n-ridge_k-1)
ridge_AICc

# LASSO AIC = -62.02
lasso_tLL <- lassomod$nulldev - deviance(lassomod)
lasso_k <- lassomod$df
lasso_n <- lassomod$nobs
lasso_AICc <- -lasso_tLL+2*lasso_k+2*lasso_k*(lasso_k+1)/(lasso_n-lasso_k-1)
lasso_AICc

# Elastic AIC = -93.24
elastic_tLL <- elasticmod$nulldev - deviance(elasticmod)
elastic_k <- elasticmod$df
elastic_n <- elasticmod$nobs
elastic_AICc <- -elastic_tLL+2*elastic_k+2*elastic_k*(elastic_k+1)/(elastic_n-elastic_k-1)
elastic_AICc

### By removing the above two variables, the AIC and AUC was decreased for all models, and the accuracy of the Elastic Net model was 
# increased along with its threshold. However, the accuracy of both the Ridge and LASSO models decreased. 


