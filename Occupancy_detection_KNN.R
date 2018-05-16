#Data partitioning and modeling
library(outliers)
library(caret)
library(dplyr)

#Setting the working directory
setwd('C:/Users/Aditya/Documents/Fall 3rd Semester/Applied Machine Learning/Project 3/occupancy_data')

#Reading in the data
df <- read.csv('datatraining.csv')

table(df$Occupancy)

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
ctrl <- trainControl(method="repeatedcv",repeats = 10)
knnFit <- train(Occupancy ~ ., data = df_train, method = "knn", trControl = ctrl, tuneLength = 10)

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


