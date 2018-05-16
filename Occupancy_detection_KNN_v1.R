#Data partitioning and modeling
setwd("C:/Users/Karthik/Desktop/SVM/project3")

library(outliers)
library(caret)
library(dplyr)

#Setting the working directory
#setwd('C:/Users/Aditya/Documents/Fall 3rd Semester/Applied Machine Learning/Project 3/occupancy_data')

#Reading in the data
df <- read.csv('datatraining.csv')

#Removing the Date column
df <- df[-1]

#Normalizing the Dataset
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the room data
df_n <- as.data.frame(lapply(df[1:5], normalize))
df$Occupancy <- as.factor(df$Occupancy )
df_n <- cbind(df_n, df$Occupancy)
colnames(df_n )[6] <- "Occupancy"




# create training and test data
df_train <- df[1:4071, ]
df_test <- df[4072:8143, ]

# # create labels for training and test data
# df_train_labels <- df[1:4071,6 ]
# df_test_labels <- df[4072:8143, 6]


set.seed(100)
ctrl <- trainControl(method="repeatedcv",repeats = 5)
knnFit <- caret::train(Occupancy ~ ., data = df_train, method = "knn", trControl = ctrl, tuneLength = 5)

#Output of kNN fit
knnFit

plot(knnFit)


knnPredict <- predict(knnFit,newdata = df_test )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, df_test$Occupancy )

#Plotting Area under curve
library(AUC)

plot(roc(knnPredict,df_test$Occupancy))

auc(roc(knnPredict,df_test$Occupancy))


# Neural Networks 

numFolds <- trainControl(method = 'cv', number = 10, classProbs = TRUE, verboseIter = TRUE, summaryFunction = twoClassSummary, preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

#Tuning parameters:
#size (#Hidden Units)
#decay (Weight Decay)
dt1.grid <- expand.grid(.decay = c(0.5), .size = c(10))

levels(df_train$Occupancy) <- c("yes", "no")
levels(df_test$Occupancy) <- c("yes", "no")

pred<-df_train[1:5]
res<-df_train[,6]

fit2 <- caret::train(x= pred,y=res, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1 <- predict(fit2, newdata=df_train)
conf1 <- confusionMatrix(results1, df_train$Occupancy)
cat("\nFor Train Dataset\n")
print(conf1)
pred_dt1_tr <- prediction(predictions = as.numeric(results1),labels = as.numeric(df_train$Occupancy))
perf_dt1_tr <- performance(pred_dt1_tr, "tpr", "fpr")
auc_dt1_tr <- performance(pred_dt1_tr,"auc")
auc_dt1_tr <- round(as.numeric(auc_dt1_tr@y.values),5)
print(paste('AUC of Model for Train:',auc_dt1_tr))
plot(perf_dt1_tr,type ="o",col="blue")
  

results2 <- predict(fit2, newdata=df_test)
conf2 <- confusionMatrix(results2, df_test$Occupancy)
cat("\nFor Test Dataset\n")
print(conf2)
pred_dt1_ts <- prediction(predictions = as.numeric(results2),labels = as.numeric(df_test$Occupancy))
perf_dt1_ts <- performance(pred_dt1_ts, "tpr", "fpr")
auc_dt1_ts <- conf2$overall["Accuracy"]
print(paste('AUC of Model for Test:',auc_dt1_ts))
plot(perf_dt1_ts,type ="o",col="blue")



cat("\n Experimentation\n ")
cat("\n Varying Activation function ")
fit2_act_sigmoid <- caret::train(x= pred,y=res, method = 'nnet', act.fct= 'sigmoid',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_sigmoid <- predict(fit2_act_sigmoid , newdata=df_train)
conf1_act_sigmoid  <- confusionMatrix(results1_act_sigmoid , df_train$Occupancy)
accuracy_act_sigmoid <- conf1_act_sigmoid$overall["Accuracy"]

fit2_act_tanh <- caret::train(x= pred,y=res, method = 'nnet', act.fct= 'tanh',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_tanh <- predict(fit2_act_tanh , newdata=df_train)
conf1_act_tanh  <- confusionMatrix(results1_act_tanh , df_train$Occupancy)
accuracy_act_tanh <- conf1_act_tanh$overall["Accuracy"]

cat("\n Model Accuracy with Sigmoid activation function", accuracy_act_sigmoid)
cat("\n Model Accuracy with tanh activation function", accuracy_act_tanh)


cat("\n Varying Hidden layers, nodes keeping activation function sigmoid ")

dt1.grid <- expand.grid(.decay = c(0.1), .size = c(15,10))
fit2_act_tanh <- caret::train(x= pred,y=res, method = 'nnet', act.fct= 'sigmoid',preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=dt1.grid)
results1_act_tanh <- predict(fit2_act_tanh , newdata=df_train)
conf1_act_tanh  <- confusionMatrix(results1_act_tanh , df_train$Occupancy)
accuracy_act_tanh <- conf1_act_tanh$overall["Accuracy"]

cat("\n Model Accuracy increasing nodes, hidden layers and decreasing weight to 0.1", accuracy_act_tanh)

cat("\n Accuracy for Test Dataset")
results2_exp <- predict(fit2_act_tanh, newdata=df_test)
conf2_test <- confusionMatrix(results2_exp, df_test$Occupancy)
accuracy_test <- conf2_test$overall["Accuracy"]
cat("\n For Test Dataset \n")
print(conf2_test)



#Plotting Area under curve
if(!require("AUC")){
  cat(" \n AUC package not found.. Hence installing...")
  install.packages("AUC")
}
library(AUC)

plot(perf_dt1_tr, main = "ROC curves for Neural Network", col='blue')
plot(perf_dt1_ts,add=TRUE, col='red')
legend('bottom', c("ROC Train", "ROC Test Before Exp"), fill = c('blue','red'), bty='n')

cat("\n Model Accuracy for Test before experimentation", auc_dt1_ts)
cat("\n Model Accuracy for Test after experimentation", accuracy_test)
cat("\n increasing the hidden layers nodes resulted in minor decrease in Test model accuracy")

