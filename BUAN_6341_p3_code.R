
# set directory
setwd("E:/AML - BUAN 6341")

# install necessary packages if missing
if(!require("DMwR")){
  cat(" \n DMwR package not found.. Hence installing...")
  install.packages("DMwR")
}

if(!require("caret")){
  cat(" \n caret package not found.. Hence installing...")
  install.packages("caret")
}

if(!require("unbalanced")){
  cat(" \n unbalanced package not found.. Hence installing...")
  install.packages("unbalanced")
}

# ROC Curve: requires the ROCR package.
if(!require("ROCR")){
  cat(" \n ROCR package not found.. Hence installing...")
  install.packages("ROCR")
}

#Data partitioning and modeling

if(!require("dplyr")){
  cat(" \n dplyr package not found.. Hence installing...")
  install.packages("dplyr")
}
if(!require("outliers")){
  cat(" \n outliers package not found.. Hence installing...")
  install.packages("outliers")
}

#Plotting Area under curve
if(!require("AUC")){
  cat(" \n AUC package not found.. Hence installing...")
  install.packages("AUC")
}
library(AUC)
library(outliers)
library(dplyr)

library(ROCR)
library(DMwR) #for SMOTE
library(caret)
library(caTools)
library(unbalanced)

# load dataset
df <- read.csv("Attrition.csv", header = TRUE)

#omit missing value
df <- na.omit(df)

#Dropping categorical variable with just 1 level
df = select(df, -EmployeeCount,-StandardHours,-Over18)


#Converting to categorical variables
df$Education <- as.factor(df$Education) 
df$EnvironmentSatisfaction <- as.factor(df$EnvironmentSatisfaction)
df$JobInvolvement <- as.factor(df$JobInvolvement)
df$JobLevel <- as.factor(df$JobLevel)
df$JobSatisfaction <- as.factor(df$JobSatisfaction)
df$PerformanceRating <- as.factor(df$PerformanceRating)
df$RelationshipSatisfaction <- as.factor(df$RelationshipSatisfaction)
df$StockOptionLevel <- as.factor(df$StockOptionLevel)
df$TrainingTimesLastYear <- as.factor(df$TrainingTimesLastYear)
df$WorkLifeBalance <- as.factor(df$WorkLifeBalance)

#Scaling Continous Features
if(!require("BBmisc")){
  cat(" \n BBmisc package not found.. Hence installing...")
  install.packages("BBmisc")
}
library(BBmisc)
nums <- sapply(df, is.numeric)
numeric <- df[,nums]

#Finding the categorical variables
categ <- sapply(df, is.factor)
categorical <- df[,categ]


#Dummy variable conversion
X <- select(categorical,-Attrition)
dmy <- dummyVars(" ~ . ", data = X ) 
df3 <- data.frame(predict(dmy, newdata = X))
str(df3)


#combining Categorical and Numeric
df5 <- cbind.data.frame(df3, numeric)

#Scaling whole Dataset
scaled_df <- normalize(df5, method = "standardize", range = c(0, 1))

df6 <-  cbind.data.frame(scaled_df, df$Attrition)
colnames(df6)[86] <- "Attrition"

levels(df6$Attrition) <- c(0, 1)
data<-ubBalance(X= df6[1:85], Y=df6$Attrition, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
dt1_smoted<-cbind(data$X,data$Y)
colnames(dt1_smoted)[86]<-'Attrition'

levels(dt1_smoted$Attrition) <- c("No", "Yes")

# Split the data into training and test
set.seed(1000)
intrain_dt1 <- createDataPartition(y = dt1_smoted$Attrition, p= 0.5, list = FALSE)

train_dt1 <- dt1_smoted[intrain_dt1,]
test_dt1 <- dt1_smoted[-intrain_dt1,]

numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))

fit2 <- caret::train(x= train_dt1[1:85],y=train_dt1$Attrition, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1 <- predict(fit2, newdata=train_dt1)
conf1 <- confusionMatrix(results1, train_dt1$Attrition)
cat("\nFor Train Dataset\n")
print(conf1)
pred_dt1_tr <- prediction(predictions = as.numeric(results1),labels = as.numeric(train_dt1$Attrition))
perf_dt1_tr <- performance(pred_dt1_tr, "tpr", "fpr")
auc_dt1_tr <- performance(pred_dt1_tr,"auc")
auc_dt1_tr <- round(as.numeric(auc_dt1_tr@y.values),5)
auccuracy_dt1_tr <- conf1$overall["Accuracy"]
print(paste('AUC of Model for Train:',auc_dt1_tr))
cat("\n Model accuracy for Train dataset:", auccuracy_dt1_tr)
#plot(fit2)

#tr.rmse <- sqrt(mean((results1 - train_dt1$Attrition)^2))

results2 <- predict(fit2, newdata=test_dt1)
conf2 <- confusionMatrix(results2, test_dt1$Attrition)
cat("\nFor Test Dataset\n")
print(conf2)
pred_dt1_ts <- prediction(predictions = as.numeric(results2),labels = as.numeric(test_dt1$Attrition))
perf_dt1_ts <- performance(pred_dt1_ts, "tpr", "fpr")
auccuracy_dt1_ts <- conf2$overall["Accuracy"]
auc_dt1_ts <- performance(pred_dt1_ts,"auc")
auc_dt1_ts <- round(as.numeric(auc_dt1_ts@y.values),5)
print(paste('AUC of Model for Test:',auc_dt1_ts))
cat("\n Model accuracy for test dataset:", auccuracy_dt1_ts)


probs <- predict(fit2, newdata=test_dt1, type='prob')

# install necessary packages if missing
if(!require("devtools")){
  cat(" \n devtools package not found.. Hence installing...")
  install.packages("devtools")
}
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(fit2)

cat("\n Experimentation\n ")
cat("\n Varying Activation function ")
fit2_act_sigmoid <- caret::train(x= train_dt1[1:85],y=train_dt1$Attrition, method = 'nnet', act.fct= 'sigmoid',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_sigmoid <- predict(fit2_act_sigmoid , newdata=train_dt1)
conf1_act_sigmoid  <- confusionMatrix(results1_act_sigmoid , train_dt1$Attrition)
accuracy_act_sigmoid <- conf1_act_sigmoid$overall["Accuracy"]

fit2_act_tanh <- caret::train(x= train_dt1[1:85],y=train_dt1$Attrition, method = 'nnet', act.fct= 'tanh',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_tanh <- predict(fit2_act_tanh , newdata=train_dt1)
conf1_act_tanh  <- confusionMatrix(results1_act_tanh , train_dt1$Attrition)
accuracy_act_tanh <- conf1_act_tanh$overall["Accuracy"]

cat("\n Model Accuracy with Sigmoid activation function", accuracy_act_sigmoid)
cat("\n Model Accuracy with tanh activation function", accuracy_act_tanh)

cat("\n Varying Hidden layers, nodes keeping activation function tanh ")

dt1.grid <- expand.grid(.decay = c(0.1), .size = c(15,10))
fit2_act_tanh <- caret::train(x= train_dt1[1:85],y=train_dt1$Attrition, method = 'nnet', act.fct= 'tanh',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_tanh <- predict(fit2_act_tanh , newdata=train_dt1)
conf1_act_tanh  <- confusionMatrix(results1_act_tanh , train_dt1$Attrition)
accuracy_act_tanh <- conf1_act_tanh$overall["Accuracy"]

cat("\n Model Accuracy increasing nodes, hidden layers and decreasing weight to 0.1 is : ", accuracy_act_tanh)


results2_exp <- predict(fit2_act_tanh, newdata=test_dt1)
conf2_test <- confusionMatrix(results2_exp, test_dt1$Attrition)
accuracy_test <- conf2_test$overall["Accuracy"]
cat("\n Accuracy for Test Dataset")
cat("\nAUC for Test Dataset for IBM ANN", accuracy_test)
print(auc(roc(results2_exp,test_dt1$Attrition)))

cat("\n For Test Dataset \n")
print(conf2_test)


plot(perf_dt1_tr, main = "ROC curves for Neural Network", col='blue')
plot(perf_dt1_ts,add=TRUE, col='red')
plot(roc(results2_exp,test_dt1$Attrition), add=TRUE, col='green')
legend('bottom', c("ROC Train", "ROC Test Before Exp", "ROC Test After Exp"), fill = c('blue','red', 'green'), bty='n')

cat("\n Model Accuracy for Test before experimentation", auccuracy_dt1_ts)
cat("\n Model Accuracy for Test after experimentation", accuracy_test)
cat("\n increasing the hidden layers nodes resulted in minor decrease in Test model accuracy")

cat(" \n KNN for IBM Attrition")
ctrl <- trainControl(method="repeatedcv",repeats = 10)
knnFit <- caret::train(Attrition ~ ., data=train_dt1, method = "knn", trControl = ctrl, tuneLength = 10)
#Output of kNN fit
print(knnFit)
plot(knnFit)
knnPredict <- predict(knnFit,newdata = test_dt1)
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, test_dt1$Attrition )
#Plotting Area under curve
plot(roc(knnPredict,test_dt1$Attrition))
print(auc(roc(knnPredict,test_dt1$Attrition)))

